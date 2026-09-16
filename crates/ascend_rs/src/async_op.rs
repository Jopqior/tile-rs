//! Composable async ops over `AclStream` / `AclEvent` (experimental).
//!
//! Build a graph of `DeviceOp`s, then `.await` (or `submit_and_wait`).
//! The graph is lazy: nothing hits ACL until the op is driven.
//!
//! Inspired by NVIDIA cuda-oxide's `DeviceOperation` host-side model
//! (lazy ops + stream-pool scheduling + native `.await`), adapted to the
//! Ascend `AclStream`/`AclEvent` primitives. Cross-op edges become
//! `record` + `stream_wait`; the `Future` impl can either poll
//! `aclrtQueryEvent` (busy-poll fallback) or, when a `ReportPump` is
//! attached via `StreamPool::with_callback_pump`, register a one-shot
//! `aclrtLaunchCallback` that wakes the task exactly once on completion.
//!
//! Single-threaded (no `Send` bounds) because `AclStream` / `AclEvent`
//! wrap raw handles without `Send` impls. Use under `LocalSet` /
//! `futures::executor::block_on`. A future revision may add
//! `unsafe impl Send` once ACL thread-safety is audited.

use std::cell::RefCell;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

use crate::context::AclContext;
use crate::errors::{AclResult, ToAclResult};
use crate::event::AclEvent;
use crate::stream::AclStream;

// ---- Pipe tags (phantom-only; mirror device-side pipes for future fusion) ----

/// Memory transfer engine 2 (host-to-device / load).
pub struct Mte2;
/// Memory transfer engine 3 (device-to-host / store).
pub struct Mte3;
/// Vector pipe.
pub struct V;
/// Cube (matmul) pipe.
pub struct M;
/// Pipe-agnostic — produced by `join` and other multi-pipe ops.
pub struct Any;

// ---- Report pump (ACL callback dispatcher thread) ----

/// Dispatcher thread that drains `aclrtProcessReport` so that
/// `aclrtLaunchCallback`-registered host wake closures run.
///
/// One pump per `StreamPool`. Each stream in the pool is bound to the
/// pump's thread via `aclrtSubscribeReport` at construction and
/// unbound via `aclrtUnSubscribeReport` at teardown.
pub struct ReportPump {
    thread_id: u64,
    join: Option<std::thread::JoinHandle<()>>,
    exit: std::sync::Arc<std::sync::atomic::AtomicBool>,
    bound_streams: RefCell<Vec<ascend_sys::core::aclrtStream>>,
}

// `aclrtStream` is `*mut c_void` (not Send). We hand-roll Send/Sync for the
// pump's interior because the only thread that mutates the bound-streams list
// is the owning (host) thread; the pump thread only reads ACL queues.
unsafe impl Send for ReportPump {}
unsafe impl Sync for ReportPump {}

impl ReportPump {
    /// Spawn a pump bound to `ctx`. The pump thread sets its current context
    /// and loops on `aclrtProcessReport(timeout_ms)` until `Drop`.
    pub fn new(ctx: &AclContext<'_>) -> AclResult<Self> {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::{Arc, Barrier};

        let exit = Arc::new(AtomicBool::new(false));
        let tid_slot = Arc::new(AtomicU64::new(0));
        let ready = Arc::new(Barrier::new(2));

        let raw_ctx = ctx.to_raw() as usize;
        let exit_c = exit.clone();
        let tid_c = tid_slot.clone();
        let ready_c = ready.clone();

        let join = std::thread::Builder::new()
            .name("ascend-rs-report-pump".into())
            .spawn(move || {
                // SAFETY: raw_ctx is an aclrtContext (opaque *mut c_void)
                // captured before spawn; the parent thread keeps it valid
                // for the pump's lifetime via `ctx`'s lifetime in the API.
                let ctx_ptr = raw_ctx as ascend_sys::core::aclrtContext;
                unsafe {
                    let _ = ascend_sys::core::aclrtSetCurrentContext(ctx_ptr);
                }
                // Publish our pthread_self() so the parent can subscribe.
                // SAFETY: pthread_self always succeeds.
                let tid = unsafe { libc::pthread_self() } as u64;
                tid_c.store(tid, Ordering::SeqCst);
                ready_c.wait();
                while !exit_c.load(Ordering::Acquire) {
                    // 50ms timeout — short enough to notice exit promptly,
                    // long enough to avoid busy-spinning in userspace.
                    unsafe {
                        let _ = ascend_sys::core::aclrtProcessReport(50);
                    }
                }
            })
            .expect("spawn report-pump thread");

        ready.wait();
        let thread_id = tid_slot.load(Ordering::SeqCst);
        Ok(Self {
            thread_id,
            join: Some(join),
            exit,
            bound_streams: RefCell::new(Vec::new()),
        })
    }

    /// Bind a stream's callback queue to this pump.
    pub fn subscribe(&self, stream: &AclStream<'_>) -> AclResult<()> {
        unsafe {
            ascend_sys::core::aclrtSubscribeReport(self.thread_id, stream.to_raw()).to_result()?;
        }
        self.bound_streams.borrow_mut().push(stream.to_raw());
        Ok(())
    }

    /// Thread id this pump uses (the pthread_self of the dispatcher).
    pub fn thread_id(&self) -> u64 {
        self.thread_id
    }
}

impl Drop for ReportPump {
    fn drop(&mut self) {
        // Unsubscribe each stream first so ACL stops queuing callbacks to us.
        for stream in self.bound_streams.borrow().iter() {
            unsafe {
                let _ = ascend_sys::core::aclrtUnSubscribeReport(self.thread_id, *stream);
            }
        }
        // Signal the pump to exit and join.
        self.exit.store(true, std::sync::atomic::Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

// ---- Stream pool ----

/// Pool of `AclStream`s with round-robin selection. Optionally hosts a
/// `ReportPump` so completion futures can use real (one-shot) wakers
/// instead of busy-polling.
pub struct StreamPool<'c> {
    streams: Vec<Rc<AclStream<'c>>>,
    next: RefCell<usize>,
    pump: Option<Rc<ReportPump>>,
}

impl<'c> StreamPool<'c> {
    pub fn new(streams: Vec<AclStream<'c>>) -> Self {
        assert!(
            !streams.is_empty(),
            "StreamPool requires at least one stream"
        );
        Self {
            streams: streams.into_iter().map(Rc::new).collect(),
            next: RefCell::new(0),
            pump: None,
        }
    }

    /// Attach a freshly spawned `ReportPump` for `ctx` and subscribe every
    /// stream in the pool to it. After this call, any `DeviceOpHandle` whose
    /// originating stream lives in this pool wakes via `aclrtLaunchCallback`
    /// rather than busy-polling `aclrtQueryEvent`.
    pub fn with_callback_pump(mut self, ctx: &AclContext<'_>) -> AclResult<Self> {
        let pump = ReportPump::new(ctx)?;
        for s in &self.streams {
            pump.subscribe(s)?;
        }
        self.pump = Some(Rc::new(pump));
        Ok(self)
    }

    /// Round-robin pick. Replace with least-loaded later.
    pub fn pick(&self) -> Rc<AclStream<'c>> {
        let mut i = self.next.borrow_mut();
        let s = self.streams[*i % self.streams.len()].clone();
        *i = i.wrapping_add(1);
        s
    }

    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// `true` if completions are delivered via ACL callbacks rather than
    /// query-based busy-poll.
    pub fn has_callback_pump(&self) -> bool {
        self.pump.is_some()
    }

    fn pump(&self) -> Option<Rc<ReportPump>> {
        self.pump.clone()
    }
}

// ---- DeviceOp: lazy plan ----

/// Closure that enqueues work on a stream and returns `(value, completion_event)`.
type Enqueue<'c, T> = Box<dyn for<'a> FnOnce(&'a AclStream<'c>) -> AclResult<(T, AclEvent)> + 'c>;

/// Lazy GPU operation. Polymorphic over a pipe tag `P` (phantom) and an
/// output value `T`. Nothing executes until `.await` / `submit` / `submit_and_wait`.
pub struct DeviceOp<'c, P, T> {
    enqueue: Enqueue<'c, T>,
    pool: Rc<StreamPool<'c>>,
    _pipe: PhantomData<fn() -> P>,
}

impl<'c, P: 'static, T: 'c> DeviceOp<'c, P, T> {
    /// Wrap a one-shot enqueue closure as a `DeviceOp`.
    pub fn new<F>(pool: Rc<StreamPool<'c>>, f: F) -> Self
    where
        F: for<'a> FnOnce(&'a AclStream<'c>) -> AclResult<(T, AclEvent)> + 'c,
    {
        Self {
            enqueue: Box::new(f),
            pool,
            _pipe: PhantomData,
        }
    }

    /// Sequence: run `next(self_output)` on a fresh stream after `self` completes.
    /// Uses an event to chain without host blocking.
    pub fn then<Q: 'static, U: 'c, F>(self, next: F) -> DeviceOp<'c, Q, U>
    where
        F: for<'a> FnOnce(T, &'a AclStream<'c>) -> AclResult<(U, AclEvent)> + 'c,
    {
        let pool = self.pool.clone();
        DeviceOp::new(pool.clone(), move |consumer_stream| {
            let producer_stream = pool.pick();
            let (val, prod_evt) = (self.enqueue)(&producer_stream)?;
            prod_evt.stream_wait(consumer_stream)?;
            next(val, consumer_stream)
        })
    }

    /// Fork-join two independent ops; both run, results paired.
    pub fn join<Q: 'static, U: 'c>(self, other: DeviceOp<'c, Q, U>) -> DeviceOp<'c, Any, (T, U)> {
        let pool = self.pool.clone();
        DeviceOp::new(pool.clone(), move |sink_stream| {
            let s1 = pool.pick();
            let s2 = pool.pick();
            let (a, ea) = (self.enqueue)(&s1)?;
            let (b, eb) = (other.enqueue)(&s2)?;
            ea.stream_wait(sink_stream)?;
            eb.stream_wait(sink_stream)?;
            let done = AclEvent::new()?;
            done.record(sink_stream)?;
            Ok(((a, b), done))
        })
    }

    /// Flush onto a freshly picked stream. Returns a pollable handle.
    pub fn submit(self) -> AclResult<DeviceOpHandle<'c, T>> {
        let stream = self.pool.pick();
        let pump = self.pool.pump();
        let (val, evt) = (self.enqueue)(&stream)?;
        Ok(DeviceOpHandle {
            value: Some(val),
            event: evt,
            stream,
            pump,
            waker_slot: None,
            callback_armed: false,
        })
    }

    /// Convenience: submit and block until complete.
    pub fn submit_and_wait(self) -> AclResult<T> {
        let mut h = self.submit()?;
        h.event.synchronize()?;
        Ok(h.value.take().unwrap())
    }
}

// ---- Waker plumbing (ACL callback) ----

/// Shared waker slot used by the host-side callback to wake the Future.
type WakerSlot = std::sync::Mutex<Option<std::task::Waker>>;

/// `extern "C"` callback installed via `aclrtLaunchCallback`. The userdata
/// pointer is a `*const WakerSlot` cloned from an `Arc<WakerSlot>` whose
/// other clone lives in the `DeviceOpHandle`. The callback takes the
/// current waker (if any) and calls `wake()`, then drops its Arc clone.
unsafe extern "C" fn wake_cb(user_data: *mut std::os::raw::c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: This pointer was created by `Arc::into_raw` on an
    // `Arc<WakerSlot>` and handed to `aclrtLaunchCallback`. ACL calls us
    // exactly once per launch.
    let arc: std::sync::Arc<WakerSlot> =
        unsafe { std::sync::Arc::from_raw(user_data as *const WakerSlot) };
    if let Ok(mut guard) = arc.lock() {
        if let Some(w) = guard.take() {
            w.wake();
        }
    }
    // arc dropped here
}

// ---- Awaitable handle ----

/// Pending handle returned by `submit`. When the originating `StreamPool`
/// has a `ReportPump` attached, registers a one-shot `aclrtLaunchCallback`
/// on first `poll` and parks; otherwise falls back to `aclrtQueryEvent`
/// busy-poll. The fallback exists so the API works on streams that were
/// not subscribed (e.g. tests using `StreamPool::new` without
/// `.with_callback_pump(...)`).
pub struct DeviceOpHandle<'c, T> {
    value: Option<T>,
    event: AclEvent,
    stream: Rc<AclStream<'c>>,
    pump: Option<Rc<ReportPump>>,
    waker_slot: Option<std::sync::Arc<WakerSlot>>,
    callback_armed: bool,
}

impl<'c, T: Unpin> Future for DeviceOpHandle<'c, T> {
    type Output = AclResult<T>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Fast path: completed.
        match self.event.query() {
            Ok(true) => return Poll::Ready(Ok(self.value.take().unwrap())),
            Err(e) => return Poll::Ready(Err(e)),
            Ok(false) => {}
        }

        // No pump → fall back to the original busy-poll behavior.
        if self.pump.is_none() {
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }

        // With a pump: register the waker via aclrtLaunchCallback exactly
        // once. On subsequent polls (e.g. spurious wakeups before the
        // callback fires) just refresh the slot in case the task moved.
        if !self.callback_armed {
            let slot: std::sync::Arc<WakerSlot> =
                std::sync::Arc::new(std::sync::Mutex::new(Some(cx.waker().clone())));
            // Hand a strong reference to ACL; the callback drops it.
            let raw = std::sync::Arc::into_raw(slot.clone()) as *mut std::os::raw::c_void;
            let ret = unsafe {
                ascend_sys::core::aclrtLaunchCallback(
                    Some(wake_cb),
                    raw,
                    ascend_sys::core::aclrtCallbackBlockType_ACL_CALLBACK_NO_BLOCK,
                    self.stream.to_raw(),
                )
            };
            if let Err(e) = ret.to_result() {
                // Reclaim the leaked Arc so we don't leak memory on error.
                unsafe {
                    let _ = std::sync::Arc::from_raw(raw as *const WakerSlot);
                }
                // Surface the failure (do not silently busy-loop).
                return Poll::Ready(Err(e));
            }
            self.waker_slot = Some(slot);
            self.callback_armed = true;
        } else if let Some(ref slot) = self.waker_slot {
            if let Ok(mut g) = slot.lock() {
                *g = Some(cx.waker().clone());
            }
        }

        Poll::Pending
    }
}

/// IntoFuture so users can `.await` a `DeviceOp` directly.
impl<'c, P: 'static, T: Unpin + 'c> std::future::IntoFuture for DeviceOp<'c, P, T> {
    type Output = AclResult<T>;
    type IntoFuture = DeviceOpFut<'c, T>;
    fn into_future(self) -> DeviceOpFut<'c, T> {
        DeviceOpFut {
            inner: Some(self.submit()),
        }
    }
}

pub struct DeviceOpFut<'c, T> {
    inner: Option<AclResult<DeviceOpHandle<'c, T>>>,
}

impl<'c, T: Unpin> Future for DeviceOpFut<'c, T> {
    type Output = AclResult<T>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.inner.take() {
            Some(Ok(h)) => {
                let mut h = h;
                let p = Pin::new(&mut h);
                match p.poll(cx) {
                    Poll::Ready(out) => Poll::Ready(out),
                    Poll::Pending => {
                        self.inner = Some(Ok(h));
                        Poll::Pending
                    }
                }
            }
            Some(Err(e)) => Poll::Ready(Err(e)),
            None => Poll::Pending,
        }
    }
}

// ---- Builders ----

/// Record an event right after submitting some closure-supplied work.
/// Helper for building leaf ops without repeating the event boilerplate.
pub fn enqueue<'c, P: 'static, T: 'c, F>(pool: Rc<StreamPool<'c>>, work: F) -> DeviceOp<'c, P, T>
where
    F: for<'a> FnOnce(&'a AclStream<'c>) -> AclResult<T> + 'c,
{
    DeviceOp::new(pool, move |stream| {
        let v = work(stream)?;
        let evt = AclEvent::new()?;
        evt.record(stream)?;
        Ok((v, evt))
    })
}
