# Metal on free GitHub-hosted runners: what the primary sources support

Research artifact for the wayfinder ticket **查证免费 GitHub 托管 runner 的 Metal 执行能力**
(https://github.com/Jopqior/tile-rs/issues/3), part of the map **Metal 正确性 CI：为精简重构建立可信验证设计**
(https://github.com/Jopqior/tile-rs/issues/2).

- **Research date:** 2026-09-14 (UTC). All "current" statements below are as of this date.
- **Hard constraint (given, not re-litigated):** only **free GitHub-hosted runners** — no
  self-hosted runners and no paid/larger runners.
- **Question:** which free runners can actually **compile and execute a Metal compute kernel**,
  not merely "provide macOS" or "emit MSL"? Separate architecture, GPU-device visibility,
  Metal-toolchain availability, runner labels, and public-repo cost conditions.
- **Method:** primary sources only — GitHub docs source (`github/docs`), the first-party runner
  image repository (`actions/runner-images`), Apple developer documentation/transcripts, and the
  vendor docs for the virtualization stack the runner images are built on. Third-party CI runs
  are used **only** where they are public, dated, and corroborated by a GitHub staff reply; they
  are labelled as third-party evidence, not as our own experiment.
- **Not done here:** no workflow was added, dispatched, or run. Nothing in this document is a
  first-party experiment by tile-rs. Items that can only be settled by running something are
  listed in §7 as required follow-up experiments.

---

## 1. Bottom line

1. **The only free-hosted hardware that can plausibly execute Metal is the standard arm64 macOS
   runner** (Apple silicon, documented as 3 vCPU "M1" / 7 GB RAM): labels `macos-15`, `macos-26`,
   `macos-latest`, and `macos-14` (retiring). It is free and unlimited for public repositories.
2. **Intel macOS standard runners have no working Metal path that any primary source supports.**
   GitHub staff stated twice (2020, 2023) that macOS runners get no GPU exposure/passthrough, and
   the Anka virtualization stack GitHub builds its images with requires an explicit display
   setting (`set display -c pg`) to do paravirtualized graphics on Intel — no evidence that GitHub
   enables it. Treat `macos-15-intel` / `macos-26-intel` as **no Metal** until an experiment says
   otherwise.
3. **GitHub hosts no macOS GPU runner at all.** The only "GPU-powered" GitHub-hosted runner SKUs
   are larger runners (`linux_4_core_gpu`, `windows_4_core_gpu`), and larger runners are *always*
   billed even for public repositories. The macOS larger runner is an M2 Pro (`macos_xl`), also
   billed. All are excluded by the hard constraint.
4. **Metal execution on the arm64 runner is real but only partially documented.** GitHub's own
   docs removed the "MPS is not supported" disclaimer between 2025-09-26 and 2026-02-12, and a
   GitHub staff reply in 2026-08 states that pinning `macos-15` "runs your existing code on MPS,
   no changes needed". Public dated CI logs show MPS-gated PyTorch tests *running* (not skipping)
   on `macos-15-arm64`. This is evidence of a working Metal compute device, not a support promise.
5. **Compiling Metal on the free arm64 runner has a second, separate dependency:** since Xcode 26
   the Metal toolchain is an optional download. The runner image build script explicitly installs
   it for every Xcode ≥ 26, but it is known to be missing intermittently (~1/50 runs) and is
   currently missing on the `xcode-27` preview label.
6. **What cannot be guaranteed:** device identity/feature set (physical M1 GPU vs Apple
   paravirtualized GPU), working-set size, that arbitrary MSL (rather than MPS/MPSGraph) runs
   correctly, that runtime (`newLibraryWithSource`) compilation works without the Xcode Metal
   toolchain, or that any of this stays true across image updates. See §7.

---

## 2. Runner labels and architecture (as of 2026-09-14)

Source: `github/docs` `data/reusables/actions/supported-github-runners.md`
@pinned `c448e8ddef9f2d33e7f2dbc98007cbe2ddd6e74c` (2026-08-20), rendered live at
https://docs.github.com/en/actions/reference/runners/github-hosted-runners (retrieved 2026-09-14).

Standard runners for **public** repositories, macOS rows:

| VM | CPU | RAM | SSD | Architecture | Workflow labels |
|---|---|---|---|---|---|
| macOS | 4 | 14 GB | 14 GB | Intel | `macos-15-intel`, `macos-26-intel` |
| macOS | 3 (M1) | 7 GB | 14 GB | arm64 | `macos-latest`, `macos-14`, `macos-15`, `macos-26`, `xcode-27` (public preview) |

Notes verified in the same source and in live docs HTML:

- **`macos-13` no longer appears** in the standard-runner table. macOS 13 is deprecated
  (`github/docs` @`f1d891c56`, 2026-02-12, commit message: "macOS-13, Ubuntu-20.04 were deprecated").
- **`macos-latest` is arm64** and tracks the newest stable arm64 image. It is a moving target:
  it resolved to `macos-15-arm64` on 2026-06-21 and to `macos-26-arm64` on 2026-06-29 in the same
  public repository (see §5.4). GitHub also warns: "The `-latest` runner images are the latest
  stable images that GitHub provides, and might not be the most recent version of the operating
  system available from the operating system vendor."
- **`macos-26-intel` exists**, i.e. the Intel line is still alive next to arm64.
- **`xcode-27` is a preview label**, and preview/beta images are explicitly excluded from the SLA.

The "M1" hardware is stated by GitHub in the official announcement of the arm64 runner
(`actions/runner-images` issue #9254, author `Steve-Glass` with `author_association: CONTRIBUTOR`,
2024-01-30): "The new M1 runner is available for all plans, free in public repositories, and
eligible to consume included free plan minutes in private repositories. The new runner operates
exclusively on macOS 14 and to use it, simply update the `runs-on` key ... to `macos-14`."
Changelog: https://github.blog/changelog/2024-01-30-github-actions-introducing-the-new-m1-macos-runner-available-to-open-source/

**macOS runners are virtual machines.** GitHub docs: "The Linux and macOS virtual machines both
run using passwordless `sudo`" and "macOS runners are hosted in GitHub's own macOS cloud". The
arm64 limitations list says "Nested-virtualization is not supported due to the limitation of
Apple's Virtualization Framework."

**Which hypervisor:** the first-party image definitions are Anka-based. In
`actions/runner-images` @`bac22751eb7d886e12c6063685275299469e9e5b` (2026-09-11),
`images/macos/templates/` contains `macOS-14.anka.pkr.hcl`, `macOS-14.arm64.anka.pkr.hcl`,
`macOS-15.anka.pkr.hcl`, `macOS-15.arm64.anka.pkr.hcl`, `macOS-26.anka.pkr.hcl`,
`macOS-26.arm64.anka.pkr.hcl`, `xcode-27.arm64.anka.pkr.hcl`, all requiring the
`veertuinc/veertu-anka` packer plugin (>= v3.2.0) and builder type `veertu-anka-vm-clone`.
This is the fact that ties GitHub's runner capability to Veertu Anka's documented behaviour (§5.3).

---

## 3. Free/public-repo conditions

- `github/docs` `content/billing/concepts/product-billing/github-actions.md`
  @`7c06b1d8a446c632b1df2cf4dac7a1da70627bbb` (2026-07-09):
  - "The use of standard GitHub-hosted runners is free: **In public repositories** ..."
  - Note: "**Larger runners are always charged for, even when used by public repositories or when
    you have quota available from your plan.**"
  - "If your account does not have a valid payment method on file, usage is blocked once you use
    up your quota. Usage of larger runners is always blocked until you set up a payment method."
- `supported-github-runners.md` (pinned above): "**Use of the standard GitHub-hosted runners is
  free and unlimited on public repositories.**"
- The consumer repository is public: `gh api repos/Jopqior/tile-rs` → `"visibility": "public"`
  (retrieved 2026-09-14), so the free condition is satisfiable without granting anything.

**There is no free GPU product to fall back on.**
`github/docs` `content/billing/reference/actions-runner-pricing.md`
@`c518c6014d39e332be1556ce53c8840a5a8f1664` (2026-07-09) lists the only GPU-powered hosted
runners as larger-runner SKUs:

| OS | SKU | Rate |
|---|---|---|
| Linux 4-core | `linux_4_core_gpu` | $0.052/min |
| Windows 4-core | `windows_4_core_gpu` | $0.102/min |

The macOS larger runner in the same table is "macOS 5-core (M2 Pro)", SKU `macos_xl`, $0.102/min.
So even the plausible M2 Pro GPU path is paid and out of scope. There is **no macOS GPU SKU**
and no free GPU SKU.

---

## 4. Metal toolchain (compile-side availability)

### 4.1 Apple: Xcode 26+ makes the Metal toolchain an optional download

Apple, *Downloading and installing additional Xcode components* (retrieved 2026-09-14,
markdown endpoint `.../downloading-and-installing-additional-xcode-components.md`):

> "### Download and install the Metal Toolchain — To build your Metal apps, download and install
> the optional Metal Toolchain for the platforms your app targets."

and the command-line form:

```
xcodebuild -downloadComponent metalToolchain
```

Consequence for CI: on Xcode ≥ 26, `xcrun metal` / `xcrun metallib` are **not** present unless the
component was downloaded. On earlier Xcode (e.g. the 15.x/16.x set on the macos-14 image) the
toolchain shipped inside Xcode.

### 4.2 The runner image installs it for Xcode ≥ 26 — but it is not listed and it flakes

First-party build code, `actions/runner-images` @`bac22751`:

- `images/macos/scripts/helpers/Xcode.Installer.psm1`:
  `Install-XcodeAdditionalComponents` runs
  `$xcodeBuildPath -downloadComponent MetalToolchain`.
- `images/macos/scripts/build/Install-Xcode.ps1` lines 37-41: for every Xcode whose version
  matches `^(\d+)\.(\d+)(?:\.(\d+))?$` with major ≥ 26, it calls
  `Install-XcodeAdditionalComponents` and then `Update-DyldCache`.

Maintainer confirmation and known problems (all in `actions/runner-images`):

| Issue | State | What it establishes |
|---|---|---|
| #12856 (2025-08-20) | closed | Xcode 26 beta missing Metal toolchain; request to preinstall. |
| #13014 (2025-09-13) | open | Xcode 26 release still missing it; `error: cannot execute tool 'metal' due to missing Metal Toolchain`. Maintainer points at `Install-Xcode.ps1#L38` as the fix (2026-02-01). |
| #14013 (2026-05-06) | closed | On `macos-26`, the MetalToolchain component is **occasionally** missing (~1/50 runs, maintainer estimate); selecting the latest installed Xcode avoided it in 300 attempts. |
| #14717 (2026-09-11) | closed as dup of #13014 | "framework 'Metal' not found" linker failure in a Rust build — on a **larger** runner. |
| #14548 (2026-08-12) | open | **Metal toolchain missing on the `xcode-27` preview label** (reporter shows it working on `macos-26`). Avoid the preview label. |

Documentation gap: the image READMEs do **not** list the Metal toolchain. Grepping
`macos-14-arm64-Readme.md`, `macos-15-arm64-Readme.md`, `macos-26-arm64-Readme.md` (image versions
`20260831.0302.1`, `20260907.0337.1`, `20260907.0351.1`) for `metal` finds only announcement
headers — nothing in "Included Software". Treat the toolchain as present-by-build-script, not as
documented-software.

### 4.3 Xcode versions on the arm64 images (retrieved 2026-09-14)

| Image | Image version | Xcode versions |
|---|---|---|
| `macos-14-arm64` | 20260831.0302.1 | 16.2, 16.1, **15.4 (default)**, 15.3 → Metal toolchain bundled in Xcode, not a separate component |
| `macos-15-arm64` | 20260907.0337.1 | 26.3, 26.2, 26.1.1, 26.0.1 → separate Metal toolchain installed by image build |
| `macos-26-arm64` | 20260907.0351.1 | 26.6 (default), 26.5, 26.4.1, 26.3 → separate Metal toolchain installed by image build |

---

## 5. Device availability and actual execution

### 5.1 GitHub's official position changed shape over time

| Date | Source (first-party) | Statement |
|---|---|---|
| 2020-10-12 | `actions/runner-images` #1779, staff reply | "our current virtualization approach for macOS doesn't allow us to provide GPU passthrough due to several hardware and software limitations ... no plans to make GPU functions available for macOS runners in the nearest future." |
| 2023-02-15 | #7085, staff reply | "It is currently impossible due to runners' design, there is no any ETA on feature like this." |
| 2023-03-05 | #7235, staff reply | "unfortunately, we do not provide GPU exposure for macOS agents at the moment" |
| 2024-06-05 | #9918, staff reply (about MPS OOM on arm64) | "Unfortunately it's an expected behaviour for macOS arm64 runners." (linked the arm64 limitations doc) |
| ≤2025-09-26 | `github/docs` arm64 limitations reusable @`e2f8af949` (and earlier @`c6408df62`, 2024-06-03) | "**Nested-virtualization and Metal Performance Shaders (MPS) are not supported** due to the limitation of Apple's Virtualization Framework." |
| 2026-02-12 | same file @`f1d891c56` | clause reduced to "Nested-virtualization is not supported ..." — **the MPS disclaimer is gone**. Live docs still show the shortened text (retrieved 2026-09-14). |
| 2026-08-06 | #14380, staff reply | "**Pin `runs-on: macos-15` — runs your existing code on MPS, no changes needed.**" |

Caveat: the 2026-02 removal happened inside the commit
"macOS-13, Ubuntu-20.04 were deprecated; Ubuntu-slim runs in unprivileged mode (#59590)", and the
referenced PR number is not retrievable via the public API (404). **Removing a disclaimer is not
a support statement**; it is only evidence that the old blanket "MPS is not supported" claim no
longer reflects reality.

### 5.2 The mechanism: Apple paravirtualized graphics inside a macOS VM

Apple, *Paravirtualized Graphics* framework documentation (retrieved 2026-09-14):

> "The ParavirtualizedGraphics framework implements hardware-accelerated graphics for macOS
> running in a virtual machine ... The operating system provides a graphics driver that runs
> inside the guest, communicating with the framework in the host operating system to take
> advantage of **Metal-accelerated graphics**."

Apple, WWDC22 session 10002 *Create macOS VMs with Virtualization framework* (transcript,
retrieved 2026-09-14):

> "A first cool capability is GPU acceleration. We have built a graphic device that exposes the
> GPU capabilities to the virtual Mac. **This means you can run Metal in the virtual machine**,
> and get great graphics performance in macOS."

Apple's `VZMacGraphicsDeviceConfiguration` (macOS 12+) is the configuration object used for this
device; the framework documentation describes it as "Configuration for a display attached to a
Mac graphics device".

So a macOS guest VM under Apple's Virtualization framework **can** expose a Metal device. Whether
GitHub's VMs do depends on the hypervisor configuration, which is Anka's.

### 5.3 The virtualization vendor's own statement (Anka)

GitHub's macOS images are Anka-based (§2). Veertu's Anka documentation,
"Graphics Acceleration / Apple Metal" (page date 2019-12-12; source pinned at
`veertuinc/anka-docs` @`dbd6885b2e5d0df322af717fef77d26c3a8a4838`, 2023-08-18; retrieved
2026-09-14):

> "Using Apple's Metal inside of your Anka VMs.
> **MPS users: Support may be limited. Please be sure test your apps.**
> Big Sur Users: The host macOS should always be either newer or the same version as VMs running
> PG to avoid incompatibilities between the Apple metal APIs.
> **Intel** ... Set the display for the VM with `anka modify {vmName} set display -c pg`
> **ARM**: For ARM Anka VMs, PG is enabled by default."

Two readings matter for tile-rs:

- On **Apple silicon** Anka VMs, paravirtualized graphics (hence a Metal device) is the default;
  on **Intel** it is opt-in and GitHub never claimed to opt in.
- The vendor itself warns that **MPS support "may be limited"** on this path, i.e. the Metal
  device here is a paravirtualized one, not a bare-metal GPU.

### 5.4 Third-party public CI evidence that Metal compute actually executes on arm64 runners

These are **other people's runs**, dated and public, corroborated by GitHub staff. They are the
strongest available substitute for an experiment of our own; they are *not* our experiment.

1. **PyTorch MPS on `macos-15-arm64` (2026-06).** `actions/runner-images` #14380
   (filed 2026-07-13, closed 2026-08-06): tests pass on `macos-15` runners, fail on `macos-26`
   runners. GitHub staff (CONTRIBUTOR, 2026-08-06): the cause is Apple's MetalPerformanceShadersGraph
   version shipped with macOS (5.6.2 works on macos-15; 6.4.2/6.5.1 broken on macos-26), it is an
   OS-level issue, and the workaround is to pin `runs-on: macos-15`.
2. **The public logs behind that report.** Repository `lanl/ldrd_neat_ml` (public, verified
   2026-09-14) uses standard runners and gates its `[mps]` tests on
   `torch.backends.mps.is_available()`.
   - Run `27912836013`, job `82592825874` (2026-06-21, `runs-on: macos-latest`):
     runner banner reports `Image: macos-15-arm64`, `Version: 20260610.0126.1`; pytest ends
     `85 passed, 2 skipped`; the warnings sections list the `[mps]` parametrizations, i.e. those
     tests were executed, not skipped.
   - Run `28387722150`, job `84106618743` (2026-06-29, same label): banner reports
     `Image: macos-26-arm64`, `Version: 20260623.0192.1`; the same `[mps]` tests ran and failed,
     matching the report above.
   Both jobs therefore had `MTLCreateSystemDefaultDevice()` return a device and executed
   Metal-backed kernels on a **free standard arm64 runner**.
3. **Direct MTLDevice/buffer test on `macos-14-arm64` (2024-09).** `actions/runner-images` #10663:
   the reporter's `MTLCreateSystemDefaultDevice` + `newBufferWithBytesNoCopy` repro "worked as
   expected" on the macOS 14 arm runner and failed on macOS 12 x86; a GitHub staff member replied
   with an alignment fix and stated it works across macOS images. The original CI logs have since
   expired and could not be re-read, so this one rests on the issue thread alone.
4. **MPS working-set limit on arm64 (2024-05).** #9918 reports
   `MPS backend out of memory ... max allowed: 7.93 GB` on `macos-13/14-arm64` — consistent with
   the 7 GB VM RAM being the Metal working-set ceiling, and a reminder that execution capacity on
   this runner is small.
5. **No evidence found** of Metal working on `macos-15-intel` / `macos-26-intel`, and the only
   Intel-era statements from GitHub are negative (§5.1). Searching `actions/runner-images` for
   Metal-related Intel reports returns nothing recent.

### 5.5 What the free runner can and cannot be claimed to guarantee

Can be claimed (defensible with the sources above):

- A free, public-repo, standard **arm64 macOS** runner is available on demand (subject to queue),
  with an Apple-silicon CPU and a Metal device that at least MPS/MPSGraph uses on macOS 15.
- `macos-15` and `macos-26` arm64 images carry Xcode ≥ 26 with the Metal toolchain installed by
  the image build, so `xcrun metal` should work (with the flakiness caveats in §4.2).
- Everything above is reachable without paying and without self-hosting.

Cannot be claimed:

- That GitHub officially supports or SLAs Metal/GPU on standard runners. No doc promises it, and
  the staff statements in §5.1 were negative for years.
- That the Metal device is the physical M1 GPU. On this stack it is Apple's paravirtualized
  (Anka "PG") GPU path, whose vendor warns MPS support "may be limited".
- Feature-set parity with real Apple silicon: `MTLDevice.supportsFamily`, maximum threadgroup
  sizes, buffer/heap limits, unified-memory behaviour, and specific MPSGraph/MPS kernels may
  differ. Unknown until probed (§7).
- Stability across time: `macos-14` (arm64) **retires 2026-11-02** (#13518: deprecation from
  2026-07-06, unsupported from 2026-11-02, with brownout windows), `macos-latest` moves between
  major versions, and Xcode 26+ toolchain availability has a known ~1/50 failure mode.
- Anything about Intel macOS runners.

---

## 6. Consequences for the Metal correctness CI design (facts only, no design decisions)

These are the constraints this research hands to the design tickets; the decisions themselves
belong to the map's grilling tickets.

- Candidate free runner set: `macos-15` (arm64) first, `macos-26` (arm64) as a parallel signal.
  `macos-latest` is usable but floats across macOS majors — pin an explicit label if the design
  wants a stable oracle.
- `macos-14` must not be a load-bearing part of the design: it is unsupported from 2026-11-02.
- Intel labels are not viable Metal execution targets on current evidence.
- Build-side (offline `xcrun metal` → `.metallib`) and run-side (source → library at runtime) are
  two independent capabilities with different failure modes: the first depends on an
  intermittently-missing 700 MB Xcode component, the second is unverified (§7).
- The macOS 26 MPS/MPSGraph regression is a *pre-existing, OS-level* source of numerical
  divergence on that image. Any correctness oracle that runs there has to distinguish tile-rs
  regressions from that known upstream divergence.
- Resource envelope of the free runner is small: 3 vCPU, 7 GB RAM shared with the Metal working
  set. A correctness CI on this runner should not assume desktop-scale buffers.
- `xcode-27` preview is not usable today for Metal (open toolchain issue), and preview images are
  outside the SLA anyway.

---

## 7. Unknowns and the sharp follow-up experiment needed

**This research cannot confirm by documentation** (no primary source states it) — these need an
actual dispatch, which this ticket deliberately did not perform:

1. **Device identity and limits on the current images.** On `macos-15-arm64` and
   `macos-26-arm64`: `MTLCopyAllDevices()` / `MTLCreateSystemDefaultDevice()` return value;
   `device.name` (is it `Apple Paravirtual device`?); `supportsFamily(.appleN)`; `hasUnifiedMemory`;
   `maxThreadsPerThreadgroup`; `maxBufferLength`; `recommendedMaxWorkingSetSize`.
2. **Does arbitrary MSL compute (not MPS) compile and execute correctly on that device?**
   Minimum probe: a kernel that writes `thread_position_in_grid` into a shared buffer, read back
   and compared on the host; plus a numeric kernel (e.g. `sqrt`/rsqrt) compared against a CPU
   reference. Ideally the probe uses MSL of the same shape as tile-rs's generated output.
3. **Runtime compilation without the Xcode toolchain.** Does `device.newLibraryWithSource(...)`
   (the `metal` Rust crate's `Device::new_library_with_source`) succeed on a runner where the
   Xcode Metal toolchain is absent? If yes, CI can avoid the flaky 700 MB component entirely; if
   no, CI must depend on it.
4. **Offline compilation on the image's default Xcode.** `xcrun -sdk macosx metal -c` +
   `xcrun metallib` + loading the `.metallib`, on `macos-15` and `macos-26` default Xcode,
   across repeated runs (to size the ~1/50 flakiness).
5. **Paravirtual-device validation errors.** `MTL_DEBUG_LAYER=1` on the probe kernels, to surface
   unsupported features before a design builds a gate on them.
6. **Determinism across repeated runs and contention.** Same kernel, N runs, same image version:
   bitwise/numeric stability of results; whether shared M1 hosts make results flaky.
7. **Intel contrast (low priority, cheap to include).** The same probe on `macos-15-intel` to
   confirm `MTLCreateSystemDefaultDevice()` is nil, closing the Intel question factually.

Sharpest single experiment (worth one ticket): a single throwaway workflow on `macos-15`
(arm64) that prints the runner banner + device properties, compiles one MSL compute kernel both
offline (`xcrun metal`/`metallib`) and at runtime (`newLibraryWithSource`), executes it, and
compares the output against a CPU reference — with `macos-26` and `macos-15-intel` as extra
matrix legs. That is what separates "documented-plausible" from "actually works".

---

## 8. Sources (retrieved 2026-09-14, with pinned revisions where the source is a repo)

GitHub (first-party):

| Ref | Pinned revision | URL |
|---|---|---|
| Hosted-runner reference + public-repo label table | `github/docs` @`c448e8ddef9f2d33e7f2dbc98007cbe2ddd6e74c` (2026-08-20) | https://github.com/github/docs/blob/c448e8ddef9f2d33e7f2dbc98007cbe2ddd6e74c/data/reusables/actions/supported-github-runners.md · live: https://docs.github.com/en/actions/reference/runners/github-hosted-runners |
| arm64 limitations (current, no MPS clause) | `github/docs` @`f1d891c561fb4bb91b24cd7551a3d19c813e41a1` (2026-02-12) | https://github.com/github/docs/blob/f1d891c561fb4bb91b24cd7551a3d19c813e41a1/data/reusables/actions/macos-runner-limitations.md |
| arm64 limitations (historical, with MPS clause) | `github/docs` @`e2f8af949` (2025-09-26) and @`c6408df62` (2024-06-03) | https://github.com/github/docs/blob/e2f8af949/data/reusables/actions/macos-runner-limitations.md |
| Actions billing / free public repos / larger runners always charged | `github/docs` @`7c06b1d8a446c632b1df2cf4dac7a1da70627bbb` (2026-07-09) | https://github.com/github/docs/blob/7c06b1d8a446c632b1df2cf4dac7a1da70627bbb/content/billing/concepts/product-billing/github-actions.md |
| Runner pricing incl. GPU SKUs | `github/docs` @`c518c6014d39e332be1556ce53c8840a5a8f1664` (2026-07-09) | https://github.com/github/docs/blob/c518c6014d39e332be1556ce53c8840a5a8f1664/content/billing/reference/actions-runner-pricing.md |
| Anka packer templates (image virtualization stack) | `actions/runner-images` @`bac22751eb7d886e12c6063685275299469e9e5b` (2026-09-11) | https://github.com/actions/runner-images/tree/bac22751eb7d886e12c6063685275299469e9e5b/images/macos/templates |
| Metal toolchain install in image build | same revision | https://github.com/actions/runner-images/blob/bac22751eb7d886e12c6063685275299469e9e5b/images/macos/scripts/build/Install-Xcode.ps1 · .../scripts/helpers/Xcode.Installer.psm1 |
| Image software manifests | image versions `20260831.0302.1` (macos-14-arm64), `20260907.0337.1` (macos-15-arm64), `20260907.0351.1` (macos-26-arm64) | https://github.com/actions/runner-images/blob/main/images/macos/{macos-14-arm64,macos-15-arm64,macos-26-arm64}-Readme.md |
| M1 runner announcement | issue #9254 (GitHub staff, 2024-01-30) | https://github.com/actions/runner-images/issues/9254 · https://github.blog/changelog/2024-01-30-github-actions-introducing-the-new-m1-macos-runner-available-to-open-source/ |
| No GPU passthrough statements | issues #1779 (2020), #7085 (2023), #7235 (2023), #9918 (2024) | https://github.com/actions/runner-images/issues/1779 · /7085 · /7235 · /9918 |
| MPS works on macos-15, broken on macos-26 (OS-level) | issue #14380 (2026-07-13 → 2026-08-06) | https://github.com/actions/runner-images/issues/14380 |
| Metal buffer/device on macos-14 arm | issue #10663 (2024-09) | https://github.com/actions/runner-images/issues/10663 |
| Metal toolchain missing/flaky | issues #12856, #13014, #14013, #14548, #14717 | https://github.com/actions/runner-images/issues/12856 · /13014 · /14013 · /14548 · /14717 |
| macOS 14 deprecation/retirement | issue #13518 (2026-01-11) | https://github.com/actions/runner-images/issues/13518 |
| Public CI logs (third-party) | `lanl/ldrd_neat_ml` run 27912836013 job 82592825874 (2026-06-21); run 28387722150 job 84106618743 (2026-06-29) | https://github.com/lanl/ldrd_neat_ml/actions/runs/27912836013 · /28387722150 |

Apple (first-party):

| Ref | URL |
|---|---|
| Paravirtualized Graphics framework (Metal-accelerated guest graphics) | https://developer.apple.com/documentation/paravirtualizedgraphics |
| Virtualization framework | https://developer.apple.com/documentation/virtualization |
| `VZMacGraphicsDeviceConfiguration` | https://developer.apple.com/documentation/virtualization/vzmacgraphicsdeviceconfiguration |
| `MTLCreateSystemDefaultDevice()` | https://developer.apple.com/documentation/metal/mtlcreatesystemdefaultdevice() |
| Downloading and installing additional Xcode components (Metal Toolchain) | https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components |
| WWDC22 session 10002, *Create macOS VMs with Virtualization framework* | https://developer.apple.com/videos/play/wwdc2022/10002/ |

Vendor (the virtualization stack GitHub's images are built on):

| Ref | Pinned revision | URL |
|---|---|---|
| Anka — Graphics Acceleration / Apple Metal | `veertuinc/anka-docs` @`dbd6885b2e5d0df322af717fef77d26c3a8a4838` (2023-08-18; page date 2019-12-12) | https://docs.veertu.com/anka/anka-virtualization-cli/graphics-acceleration-apple-metal/ |

---

## 9. Explicit uncertainty register

- **U1** Device identity/limits on current arm64 images unknown (no doc states them).
- **U2** Whether arbitrary (non-MPS) MSL compute executes correctly here is unverified by us.
- **U3** Whether runtime MSL compilation (`newLibraryWithSource`) needs the Xcode 26 Metal
  toolchain is unknown.
- **U4** Intel labels: assumed no Metal from historical statements; no 2024-2026 primary
  confirmation.
- **U5** The 2026-02 removal of the "MPS not supported" doc clause has no public rationale
  (PR #59590 is not retrievable via the public API), so official support status remains unclear.
- **U6** Flakiness of the Metal toolchain and of the paravirtual GPU under load is only
  characterised by third-party reports, not measured.
- **U7** Whether `macos-26-arm64` can serve as the correctness oracle given the MPSGraph 6.4/6.5
  regression is a design question this artifact does not answer.
