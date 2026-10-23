//! SSE2 versions of the spatial predictor residuals (libwebp's `PredictorSub*_SSE2`),
//! bit-exact with the scalar ones. They handle whole groups of four pixels and return
//! how many they did; the caller finishes the rest.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn load(p: &[u32], i: usize) -> __m128i {
    _mm_loadu_si128(p.as_ptr().add(i).cast())
}

#[inline(always)]
unsafe fn average2(a: __m128i, b: __m128i) -> __m128i {
    let one = _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1));
    _mm_sub_epi8(_mm_avg_epu8(a, b), one)
}

#[inline(always)]
unsafe fn abs_diff(a: __m128i, b: __m128i) -> __m128i {
    _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a))
}

#[inline(always)]
unsafe fn pixel_sums(dl: __m128i, dt: __m128i) -> __m128i {
    let s = _mm_madd_epi16(_mm_sub_epi16(dl, dt), _mm_set1_epi16(1));
    _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01))
}

/// Per pixel: `t` if sum(|l - tl|) <= sum(|t - tl|) over the four channels, else `l`.
#[inline(always)]
unsafe fn select(t: __m128i, l: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let dl = abs_diff(l, tl);
    let dt = abs_diff(t, tl);
    let lo = pixel_sums(_mm_unpacklo_epi8(dl, zero), _mm_unpacklo_epi8(dt, zero));
    let hi = pixel_sums(_mm_unpackhi_epi8(dl, zero), _mm_unpackhi_epi8(dt, zero));
    let s = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(lo),
        _mm_castsi128_ps(hi),
        0b10_00_10_00,
    ));
    let use_l = _mm_cmpgt_epi32(s, zero);
    _mm_or_si128(_mm_and_si128(use_l, l), _mm_andnot_si128(use_l, t))
}

#[inline(always)]
unsafe fn clamped_add_subtract_full(l: __m128i, t: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let lo = _mm_sub_epi16(
        _mm_add_epi16(_mm_unpacklo_epi8(l, zero), _mm_unpacklo_epi8(t, zero)),
        _mm_unpacklo_epi8(tl, zero),
    );
    let hi = _mm_sub_epi16(
        _mm_add_epi16(_mm_unpackhi_epi8(l, zero), _mm_unpackhi_epi8(t, zero)),
        _mm_unpackhi_epi8(tl, zero),
    );
    _mm_packus_epi16(lo, hi)
}

#[inline(always)]
unsafe fn add_half_difference(a: __m128i, b: __m128i) -> __m128i {
    let d = _mm_sub_epi16(a, b);
    _mm_add_epi16(
        a,
        _mm_srai_epi16(_mm_add_epi16(d, _mm_srli_epi16(d, 15)), 1),
    )
}

#[inline(always)]
unsafe fn clamped_add_subtract_half(l: __m128i, t: __m128i, tl: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();
    let ave = average2(l, t);
    let lo = add_half_difference(_mm_unpacklo_epi8(ave, zero), _mm_unpacklo_epi8(tl, zero));
    let hi = add_half_difference(_mm_unpackhi_epi8(ave, zero), _mm_unpackhi_epi8(tl, zero));
    _mm_packus_epi16(lo, hi)
}

#[inline(always)]
unsafe fn predict<const M: usize>(l: __m128i, tl: __m128i, t: __m128i, tr: __m128i) -> __m128i {
    match M {
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => average2(average2(l, tr), t),
        6 => average2(l, tl),
        7 => average2(l, t),
        8 => average2(tl, t),
        9 => average2(t, tr),
        10 => average2(average2(l, tl), average2(t, tr)),
        11 => select(t, l, tl),
        12 => clamped_add_subtract_full(l, t, tl),
        13 => clamped_add_subtract_half(l, t, tl),
        _ => _mm_set1_epi32(0xff000000u32 as i32),
    }
}

/// Residuals of `cur[i + 1]` for `i` in `0..out.len()`, with left `cur[i]` and top-left,
/// top and top-right `up[i..i + 3]`.
pub(super) fn sub_batch<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        i = unsafe { super::avx2::sub_batch::<M>(cur, up, out) };
    }
    // SAFETY: SSE2 is part of x86_64; the loads read cur[i..i + 5] and up[i..i + 6] and the
    // store writes out[i..i + 4], all in bounds for i + 4 <= n by the assert above.
    unsafe {
        while i < n {
            let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
            let res = _mm_sub_epi8(load(cur, i + 1), pred);
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), res);
            i += 4;
        }
    }
    n
}

/// `sub_batch` for groups of four pixels none of which is fully transparent, stopping at
/// the first group that has one. Returns how many pixels it did.
pub(super) fn sub_batch_opaque<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !3;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    if crate::wide::has_avx2() {
        // SAFETY: the CPU supports AVX2.
        i = unsafe { super::avx2::sub_batch_opaque::<M>(cur, up, out) };
    }
    // SAFETY: as in `sub_batch`.
    unsafe {
        while i < n {
            let x = load(cur, i + 1);
            let alpha_zero = _mm_cmpeq_epi32(_mm_srli_epi32(x, 24), _mm_setzero_si128());
            if _mm_movemask_epi8(alpha_zero) != 0 {
                break;
            }
            let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
            _mm_storeu_si128(out.as_mut_ptr().add(i).cast(), _mm_sub_epi8(x, pred));
            i += 4;
        }
    }
    i
}

/// `(changed, nonzero)`: bit `j` set where `v[j + 1] != v[j]`, and where `v[j + 1] != 0`.
pub(super) fn run_masks(v: &[u32; 65]) -> (u64, u64) {
    let (mut changed, mut nonzero) = (0u64, 0u64);
    // SAFETY: SSE2 is part of x86_64; the loads read v[16 * k..16 * k + 17] for k < 4.
    unsafe {
        let zero = _mm_setzero_si128();
        for k in 0..4 {
            let at = |i: usize| load(v, 16 * k + i);
            let (c0, c1, c2, c3) = (at(1), at(5), at(9), at(13));
            let eq = _mm_packs_epi16(
                _mm_packs_epi32(_mm_cmpeq_epi32(c0, at(0)), _mm_cmpeq_epi32(c1, at(4))),
                _mm_packs_epi32(_mm_cmpeq_epi32(c2, at(8)), _mm_cmpeq_epi32(c3, at(12))),
            );
            let is_zero = _mm_packs_epi16(
                _mm_packs_epi32(_mm_cmpeq_epi32(c0, zero), _mm_cmpeq_epi32(c1, zero)),
                _mm_packs_epi32(_mm_cmpeq_epi32(c2, zero), _mm_cmpeq_epi32(c3, zero)),
            );
            changed |= u64::from(!_mm_movemask_epi8(eq) as u16) << (16 * k);
            nonzero |= u64::from(!_mm_movemask_epi8(is_zero) as u16) << (16 * k);
        }
    }
    (changed, nonzero)
}
