//! Ending the process without running C++ static destructors.
//!
//! ONNX Runtime requires `ReleaseEnv` to happen before the static destructors inside
//! `libonnxruntime.so` run. The C++ this tool replaced satisfied that by construction: its
//! `Ort::Env` was an object member, destroyed at the end of `main`, long before any static
//! destructor. This crate cannot do the same. The `ort` crate keeps the environment in a private
//! global for the whole process -- ONNX Runtime permits `CreateEnv` only once, so an environment
//! that could be dropped and rebuilt would crash on the second session -- and defers `ReleaseEnv`
//! to a handler it installs in `.fini_array`.
//!
//! That handler and the shared library's own C++ destructors then run in the same process-exit
//! phase, and their order is the dynamic linker's business rather than ours. When it comes out
//! wrong, a destructor destroys a mutex that is still reachable, and the next lock of it is a
//! use-after-destroy.
//!
//! Which platform notices is the reason this went unseen for two releases. Bionic checks
//! `pthread_mutex_lock` against destroyed mutexes and aborts the process (`FORTIFY:
//! pthread_mutex_lock called on destroyed mutex`); glibc performs no such check, so the identical
//! defect passes silently on Linux. Every containerised linux-aarch64 run was therefore green
//! while every Android run aborted after printing its report.
//!
//! So the process leaves before that phase begins. Nothing is lost by it: the report file is
//! written and closed, the profiler trace is closed by `profile::finish`, the session is dropped
//! normally on the way out of `run`, and everything else is memory the kernel reclaims. What is
//! skipped is only the teardown whose ordering cannot be guaranteed.
//!
//! The causal chain above is inferred by reading the `ort` and ONNX Runtime sources, not observed
//! in a debugger: the abort reproduces only on a physical device, and this crate is developed
//! without one. What is directly established is narrower -- the process aborts during exit, after
//! the report has been printed, on Android and never on glibc. Two other explanations were
//! examined against the sources and ruled out: ending the profiler twice (`Profiler::EndProfiling`
//! checks `enabled_` before taking its mutex) and link-time removal of the crate's exit handler
//! (the binary's `.fini_array` relocation does resolve into `.text`). Treat this as the best
//! available account rather than a proven one, and check it against a real backtrace before
//! building on it.
//!
//! This is also a workaround at the wrong layer -- the real fix belongs in `ort`, whose
//! `.fini_array` approach is documented as validated on Linux and Windows -- so it should be
//! revisited when that crate is upgraded.

use std::io::Write;

/// Ends the process with `code`, skipping atexit handlers and C++ static destructors.
pub fn immediately(code: i32) -> ! {
    // `_exit` flushes nothing, so anything still buffered has to go out first. Both streams are
    // global handles, so this covers every `println!` and `anstream::println!` written anywhere.
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    // SAFETY: one of the two sanctioned unsafe blocks in this crate -- see the
    // `unsafe_code = "warn"` note in Cargo.toml, and `host::peak_rss_bytes` for the other.
    // `_exit` takes no pointer and reads no memory: it terminates the process without unwinding,
    // without running handlers, and without returning, so there is no state for it to invalidate.
    #[allow(unsafe_code)]
    unsafe {
        libc::_exit(code)
    }
}
