//! Runtime tests for the experimental `async_op` module.
//!
//! All ACL-touching tests are `#[ignore]` so `cargo test` on a non-NPU host
//! still passes. To run on a 910c box:
//!
//! ```bash
//! source /usr/local/Ascend/cann-8.5.0/set_env.sh
//! cargo test -p ascend_rs --features async_op -- --ignored --nocapture
//! ```

#![cfg(feature = "async_op")]

use std::rc::Rc;

use ascend_rs::async_op::{DeviceOp, StreamPool, V, enqueue};
use ascend_rs::prelude::*;

/// Round-robin sanity — no ACL required (StreamPool::pick just rotates an index),
/// but we still need an empty pool guard via assertion.
#[test]
#[should_panic(expected = "StreamPool requires at least one stream")]
fn stream_pool_rejects_empty() {
    let _ = StreamPool::new(Vec::<AclStream<'static>>::new());
}

// ---- helpers ----

fn make_pool<'c>(ctx: &'c AclContext<'c>, n: usize) -> anyhow::Result<StreamPool<'c>> {
    let streams: Vec<AclStream<'c>> = (0..n)
        .map(|_| AclStream::new(ctx))
        .collect::<Result<_, _>>()?;
    Ok(StreamPool::new(streams))
}

fn make_pool_with_pump<'c>(ctx: &'c AclContext<'c>, n: usize) -> anyhow::Result<StreamPool<'c>> {
    let streams: Vec<AclStream<'c>> = (0..n)
        .map(|_| AclStream::new(ctx))
        .collect::<Result<_, _>>()?;
    Ok(StreamPool::new(streams).with_callback_pump(ctx)?)
}

/// Tiny no-alloc executor: poll, wake (which just sets a flag), poll again.
fn block_on<F: std::future::Future>(mut fut: F) -> F::Output {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct FlagWake(AtomicBool);
    impl Wake for FlagWake {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let flag = Arc::new(FlagWake(AtomicBool::new(true)));
    let waker = Waker::from(flag.clone());
    let mut cx = Context::from_waker(&waker);
    let mut fut = unsafe { std::pin::Pin::new_unchecked(&mut fut) };
    loop {
        if flag.0.swap(false, Ordering::SeqCst) {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => continue,
            }
        }
        std::hint::spin_loop();
    }
}

/// Split H2D and D2H across `.then(...)` so two different streams in the pool
/// are exercised, and validate the readback bit-exactly.
///
/// Here we keep `host_in` and `host_out` in `Rc<RefCell<_>>` so each closure
/// can borrow without moving — this is how a real `h2d`/`d2h` builder would
/// look.
#[test]
#[ignore = "needs NPU hardware"]
fn async_op_then_chain_validates_bytes() -> anyhow::Result<()> {
    use std::cell::RefCell;

    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let ctx = AclContext::new(&device)?;
    let pool = Rc::new(make_pool(&ctx, 2)?);

    let n = 2048usize;
    let host_in: Rc<Vec<f32>> = Rc::new((0..n).map(|i| (i as f32).sin()).collect());
    let host_out: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0f32; n]));
    let dev: Rc<RefCell<DeviceBuffer<f32>>> = Rc::new(RefCell::new(unsafe {
        DeviceBuffer::<f32>::uninitialized_with_policy(n, AclrtMemMallocPolicy::HugeFirst)?
    }));

    let host_in_c = host_in.clone();
    let dev_c1 = dev.clone();
    let upload: DeviceOp<V, ()> = enqueue(pool.clone(), move |stream| {
        let dev = dev_c1.borrow();
        unsafe {
            ascend_rs::memory::host_to_device_async(
                host_in_c.as_ptr(),
                dev.as_device_ptr(),
                n,
                stream,
            )?;
        }
        Ok(())
    });

    let dev_c2 = dev.clone();
    let host_out_c = host_out.clone();
    let pipeline = upload.then::<V, (), _>(move |(), stream| {
        let dev = dev_c2.borrow();
        let mut out = host_out_c.borrow_mut();
        unsafe {
            ascend_rs::memory::device_to_host_async(
                dev.as_device_ptr(),
                out.as_mut_ptr(),
                n,
                stream,
            )?;
        }
        let evt = AclEvent::new()?;
        evt.record(stream)?;
        Ok(((), evt))
    });

    pipeline.submit_and_wait()?;

    let out = host_out.borrow();
    for i in 0..n {
        let expect = (i as f32).sin();
        assert!(
            (out[i] - expect).abs() < 1e-6,
            "mismatch at {}: got {}, want {}",
            i,
            out[i],
            expect
        );
    }
    Ok(())
}

/// `.join()` runs two independent H2Ds on different streams in parallel;
/// the sink stream waits on both and the future resolves once both are done.
#[test]
#[ignore = "needs NPU hardware"]
fn async_op_join_two_h2d() -> anyhow::Result<()> {
    use std::cell::RefCell;

    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let ctx = AclContext::new(&device)?;
    let pool = Rc::new(make_pool(&ctx, 3)?);

    let n = 512usize;
    let xa: Rc<Vec<f32>> = Rc::new((0..n).map(|i| i as f32).collect());
    let xb: Rc<Vec<f32>> = Rc::new((0..n).map(|i| -(i as f32)).collect());
    let da: Rc<RefCell<DeviceBuffer<f32>>> = Rc::new(RefCell::new(unsafe {
        DeviceBuffer::<f32>::uninitialized(n)?
    }));
    let db: Rc<RefCell<DeviceBuffer<f32>>> = Rc::new(RefCell::new(unsafe {
        DeviceBuffer::<f32>::uninitialized(n)?
    }));

    let xa_c = xa.clone();
    let da_c = da.clone();
    let op_a: DeviceOp<V, ()> = enqueue(pool.clone(), move |stream| {
        let d = da_c.borrow();
        unsafe {
            ascend_rs::memory::host_to_device_async(xa_c.as_ptr(), d.as_device_ptr(), n, stream)?;
        }
        Ok(())
    });

    let xb_c = xb.clone();
    let db_c = db.clone();
    let op_b: DeviceOp<V, ()> = enqueue(pool.clone(), move |stream| {
        let d = db_c.borrow();
        unsafe {
            ascend_rs::memory::host_to_device_async(xb_c.as_ptr(), d.as_device_ptr(), n, stream)?;
        }
        Ok(())
    });

    let joined = op_a.join(op_b);
    joined.submit_and_wait()?;

    // Read back through the synchronous to_host path to confirm both uploads landed.
    let a_back = da.borrow().to_host()?;
    let b_back = db.borrow().to_host()?;
    let a_back = a_back.as_slice();
    let b_back = b_back.as_slice();
    for i in 0..n {
        assert_eq!(a_back[i], i as f32);
        assert_eq!(b_back[i], -(i as f32));
    }
    Ok(())
}

/// Drive a `DeviceOp` through `.await` using the inline `block_on` executor
/// instead of `submit_and_wait`. Validates that the `IntoFuture`/`DeviceOpFut`
/// poll loop terminates on event completion.
#[test]
#[ignore = "needs NPU hardware"]
fn async_op_await_path() -> anyhow::Result<()> {
    use std::cell::RefCell;

    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let ctx = AclContext::new(&device)?;
    let pool = Rc::new(make_pool(&ctx, 2)?);

    let n = 256usize;
    let host_in: Rc<Vec<f32>> = Rc::new((0..n).map(|i| i as f32 * 2.0).collect());
    let host_out: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0f32; n]));
    let dev: Rc<RefCell<DeviceBuffer<f32>>> = Rc::new(RefCell::new(unsafe {
        DeviceBuffer::<f32>::uninitialized(n)?
    }));

    let host_in_c = host_in.clone();
    let dev_c1 = dev.clone();
    let upload: DeviceOp<V, ()> = enqueue(pool.clone(), move |stream| {
        let d = dev_c1.borrow();
        unsafe {
            ascend_rs::memory::host_to_device_async(
                host_in_c.as_ptr(),
                d.as_device_ptr(),
                n,
                stream,
            )?;
        }
        Ok(())
    });

    let dev_c2 = dev.clone();
    let host_out_c = host_out.clone();
    let pipeline = upload.then::<V, (), _>(move |(), stream| {
        let d = dev_c2.borrow();
        let mut out = host_out_c.borrow_mut();
        unsafe {
            ascend_rs::memory::device_to_host_async(
                d.as_device_ptr(),
                out.as_mut_ptr(),
                n,
                stream,
            )?;
        }
        let evt = AclEvent::new()?;
        evt.record(stream)?;
        Ok(((), evt))
    });

    block_on(async move {
        pipeline.await.unwrap();
    });

    let out = host_out.borrow();
    for i in 0..n {
        assert_eq!(out[i], i as f32 * 2.0);
    }
    Ok(())
}

/// Drives a pipeline against a `StreamPool` that has a `ReportPump` attached,
/// so completion goes through `aclrtLaunchCallback` rather than the
/// `aclrtQueryEvent` busy-poll fallback. Validates that:
///   1. the pump-spawn and stream-subscribe machinery succeeds,
///   2. `.await` returns once the stream callback fires (no busy-poll),
///   3. the device→host readback bytes are correct,
///   4. teardown (unsubscribe + thread join) cleans up without hanging.
#[test]
#[ignore = "needs NPU hardware"]
fn async_op_callback_pump_await() -> anyhow::Result<()> {
    use ascend_rs::async_op::StreamPool;
    use std::cell::RefCell;

    let acl = Acl::new()?;
    let device = Device::new(&acl)?;
    let ctx = AclContext::new(&device)?;
    let pool = Rc::new(make_pool_with_pump(&ctx, 2)?);
    assert!(
        pool.has_callback_pump(),
        "expected the pool to have a callback pump attached"
    );

    let n = 1024usize;
    let host_in: Rc<Vec<f32>> = Rc::new((0..n).map(|i| (i as f32) * 0.5).collect());
    let host_out: Rc<RefCell<Vec<f32>>> = Rc::new(RefCell::new(vec![0.0f32; n]));
    let dev: Rc<RefCell<DeviceBuffer<f32>>> = Rc::new(RefCell::new(unsafe {
        DeviceBuffer::<f32>::uninitialized(n)?
    }));

    let host_in_c = host_in.clone();
    let dev_c1 = dev.clone();
    let upload: DeviceOp<V, ()> = enqueue(pool.clone(), move |stream| {
        let d = dev_c1.borrow();
        unsafe {
            ascend_rs::memory::host_to_device_async(
                host_in_c.as_ptr(),
                d.as_device_ptr(),
                n,
                stream,
            )?;
        }
        Ok(())
    });

    let dev_c2 = dev.clone();
    let host_out_c = host_out.clone();
    let pipeline = upload.then::<V, (), _>(move |(), stream| {
        let d = dev_c2.borrow();
        let mut out = host_out_c.borrow_mut();
        unsafe {
            ascend_rs::memory::device_to_host_async(
                d.as_device_ptr(),
                out.as_mut_ptr(),
                n,
                stream,
            )?;
        }
        let evt = AclEvent::new()?;
        evt.record(stream)?;
        Ok(((), evt))
    });

    block_on(async move {
        pipeline.await.unwrap();
    });

    let out = host_out.borrow();
    for i in 0..n {
        assert_eq!(out[i], (i as f32) * 0.5);
    }
    // pool's pump is unsubscribed + joined when `pool` drops at scope exit.
    let _ = StreamPool::<'_>::has_callback_pump; // type-level sanity touch
    Ok(())
}
