//! AVX2 versions of the SSE2 kernels in `sse2.rs`, eight pixels at a time. Every 256-bit
//! operation here works within 128-bit lanes, so each lane computes what the SSE2 kernel does.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

#[inline(always)]
unsafe fn load(p: &[u32], i: usize) -> __m256i {
    _mm256_loadu_si256(p.as_ptr().add(i).cast())
}

#[inline(always)]
unsafe fn average2(a: __m256i, b: __m256i) -> __m256i {
    let one = _mm256_and_si256(_mm256_xor_si256(a, b), _mm256_set1_epi8(1));
    _mm256_sub_epi8(_mm256_avg_epu8(a, b), one)
}

#[inline(always)]
unsafe fn abs_diff(a: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a))
}

#[inline(always)]
unsafe fn pixel_sums(dl: __m256i, dt: __m256i) -> __m256i {
    let s = _mm256_madd_epi16(_mm256_sub_epi16(dl, dt), _mm256_set1_epi16(1));
    _mm256_add_epi32(s, _mm256_shuffle_epi32(s, 0b10_11_00_01))
}

#[inline(always)]
unsafe fn select(t: __m256i, l: __m256i, tl: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let dl = abs_diff(l, tl);
    let dt = abs_diff(t, tl);
    let lo = pixel_sums(_mm256_unpacklo_epi8(dl, zero), _mm256_unpacklo_epi8(dt, zero));
    let hi = pixel_sums(_mm256_unpackhi_epi8(dl, zero), _mm256_unpackhi_epi8(dt, zero));
    let s = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(lo),
        _mm256_castsi256_ps(hi),
        0b10_00_10_00,
    ));
    let use_l = _mm256_cmpgt_epi32(s, zero);
    _mm256_or_si256(_mm256_and_si256(use_l, l), _mm256_andnot_si256(use_l, t))
}

#[inline(always)]
unsafe fn clamped_add_subtract_full(l: __m256i, t: __m256i, tl: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let lo = _mm256_sub_epi16(
        _mm256_add_epi16(_mm256_unpacklo_epi8(l, zero), _mm256_unpacklo_epi8(t, zero)),
        _mm256_unpacklo_epi8(tl, zero),
    );
    let hi = _mm256_sub_epi16(
        _mm256_add_epi16(_mm256_unpackhi_epi8(l, zero), _mm256_unpackhi_epi8(t, zero)),
        _mm256_unpackhi_epi8(tl, zero),
    );
    _mm256_packus_epi16(lo, hi)
}

#[inline(always)]
unsafe fn add_half_difference(a: __m256i, b: __m256i) -> __m256i {
    let d = _mm256_sub_epi16(a, b);
    _mm256_add_epi16(
        a,
        _mm256_srai_epi16(_mm256_add_epi16(d, _mm256_srli_epi16(d, 15)), 1),
    )
}

#[inline(always)]
unsafe fn clamped_add_subtract_half(l: __m256i, t: __m256i, tl: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let ave = average2(l, t);
    let lo = add_half_difference(_mm256_unpacklo_epi8(ave, zero), _mm256_unpacklo_epi8(tl, zero));
    let hi = add_half_difference(_mm256_unpackhi_epi8(ave, zero), _mm256_unpackhi_epi8(tl, zero));
    _mm256_packus_epi16(lo, hi)
}

#[inline(always)]
unsafe fn predict<const M: usize>(l: __m256i, tl: __m256i, t: __m256i, tr: __m256i) -> __m256i {
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
        _ => _mm256_set1_epi32(0xff000000u32 as i32),
    }
}

/// libwebp's `PredictorSub{M}_AVX2`: `sse2::sub_batch` for whole groups of eight pixels.
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sub_batch<const M: usize>(cur: &[u32], up: &[u32], out: &mut [u32]) -> usize {
    let n = out.len() & !7;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    // SAFETY: the loads read cur[i..i + 9] and up[i..i + 10], the store writes out[i..i + 8], all in
    // bounds for i + 8 <= n by the assert above.
    while i < n {
        let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
        let res = _mm256_sub_epi8(load(cur, i + 1), pred);
        _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), res);
        i += 8;
    }
    n
}

/// `sse2::sub_batch_opaque` for groups of eight pixels.
///
/// # Safety
/// The CPU supports AVX2.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn sub_batch_opaque<const M: usize>(
    cur: &[u32],
    up: &[u32],
    out: &mut [u32],
) -> usize {
    let n = out.len() & !7;
    assert!(cur.len() > n && up.len() >= n + 2);
    let mut i = 0;
    // SAFETY: as in `sub_batch`.
    while i < n {
        let x = load(cur, i + 1);
        let alpha_zero = _mm256_cmpeq_epi32(_mm256_srli_epi32(x, 24), _mm256_setzero_si256());
        if _mm256_movemask_epi8(alpha_zero) != 0 {
            break;
        }
        let pred = predict::<M>(load(cur, i), load(up, i), load(up, i + 1), load(up, i + 2));
        _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), _mm256_sub_epi8(x, pred));
        i += 8;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossless::encoder::entropy::sub_pixels;
    use crate::lossless::encoder::predictor::predict;

    fn check_avx2<const M: usize>(cur: &[u32], up: &[u32]) {
        let n = cur.len() - 1;
        let (mut want, mut got) = (vec![0; n], vec![0; n]);
        for i in 0..n {
            want[i] = sub_pixels(cur[i + 1], predict::<M>(cur[i], up[i], up[i + 1], up[i + 2]));
        }
        // SAFETY: the caller checked for AVX2.
        let done = unsafe { sub_batch::<M>(cur, up, &mut got) };
        assert_eq!(done, n & !7);
        assert_eq!(want[..done], got[..done], "mode {M} n {n}");
        let mut got = vec![0; n];
        // SAFETY: as above.
        let done = unsafe { sub_batch_opaque::<M>(cur, up, &mut got) };
        let first_clear = (1..=n).find(|&x| cur[x] >> 24 == 0).map_or(n, |x| x - 1);
        assert_eq!(done, first_clear.min(n) & !7, "opaque mode {M} n {n}");
        assert_eq!(want[..done], got[..done], "opaque mode {M} n {n}");
    }

    #[test]
    fn avx2_predictor_batches_match_scalar() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut seed = 0x2545_f491u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for n in [7usize, 8, 9, 15, 16, 17, 33, 70] {
            for clear_at in [usize::MAX, 3, 12, 40] {
                let cur: Vec<u32> = (0..=n)
                    .map(|i| if i == clear_at { rnd() & 0x00ff_ffff } else { rnd() | 0x0100_0000 })
                    .collect();
                let up: Vec<u32> = (0..n + 2).map(|_| rnd()).collect();
                check_avx2::<0>(&cur, &up);
                check_avx2::<1>(&cur, &up);
                check_avx2::<2>(&cur, &up);
                check_avx2::<3>(&cur, &up);
                check_avx2::<4>(&cur, &up);
                check_avx2::<5>(&cur, &up);
                check_avx2::<6>(&cur, &up);
                check_avx2::<7>(&cur, &up);
                check_avx2::<8>(&cur, &up);
                check_avx2::<9>(&cur, &up);
                check_avx2::<10>(&cur, &up);
                check_avx2::<11>(&cur, &up);
                check_avx2::<12>(&cur, &up);
                check_avx2::<13>(&cur, &up);
            }
        }
    }
}
