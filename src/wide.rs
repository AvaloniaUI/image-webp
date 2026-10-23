//! Runtime AVX2 dispatch. x86-64 builds target the SSE2 baseline, so 256-bit code runs behind
//! a CPU check: `IMAGE_WEBP_NO_AVX2` set in the environment keeps it off, so the tests can hold
//! both paths to the same output.
#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
static AVX2: AtomicU8 = AtomicU8::new(0);

/// Whether the 256-bit kernels may run: the CPU has AVX2 and `IMAGE_WEBP_NO_AVX2` is unset.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn has_avx2() -> bool {
    match AVX2.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => detect(),
    }
}

/// Reads the CPU's features and `IMAGE_WEBP_NO_AVX2` now. The first decode or encode does it
/// otherwise, on its own thread, where a `getenv` races a `setenv` on another.
pub fn detect_cpu() {
    #[cfg(target_arch = "x86_64")]
    detect();
}

#[cfg(target_arch = "x86_64")]
#[cold]
fn detect() -> bool {
    let on = std::is_x86_feature_detected!("avx2") && std::env::var_os("IMAGE_WEBP_NO_AVX2").is_none();
    AVX2.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    on
}

/// Runs `f` compiled for AVX2 when [`has_avx2`], so the loops it inlines vectorize 256 bits wide
/// instead of the baseline's 128. Only what `f` inlines is compiled so: its kernels are
/// `#[inline(always)]`.
#[inline(always)]
pub(crate) fn wide<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: the CPU supports AVX2.
        return unsafe { avx2(f) };
    }
    f()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2<R>(f: impl FnOnce() -> R) -> R {
    f()
}
