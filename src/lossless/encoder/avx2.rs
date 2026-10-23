//! AVX2 versions of the SSE2 kernels in `sse2.rs`, eight pixels at a time. Every 256-bit
//! operation here works within 128-bit lanes, so each lane computes what the SSE2 kernel does.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::entropy::fast_slog2;
use super::tables::{LOG2_TABLE, SLOG2_TABLE};

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
    let lo = pixel_sums(
        _mm256_unpacklo_epi8(dl, zero),
        _mm256_unpacklo_epi8(dt, zero),
    );
    let hi = pixel_sums(
        _mm256_unpackhi_epi8(dl, zero),
        _mm256_unpackhi_epi8(dt, zero),
    );
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
    let lo = add_half_difference(
        _mm256_unpacklo_epi8(ave, zero),
        _mm256_unpacklo_epi8(tl, zero),
    );
    let hi = add_half_difference(
        _mm256_unpackhi_epi8(ave, zero),
        _mm256_unpackhi_epi8(tl, zero),
    );
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

/// `fast_slog2` of the eight lanes of `v`, added to the four 64-bit lanes of `acc`, except
/// for lanes of 65536 and above (libwebp takes `log` of those in floating point): their bits
/// are returned for the caller to add.
#[inline(always)]
unsafe fn add_slog2(v: __m256i, acc: &mut __m256i) -> u32 {
    let small = _mm256_cmpeq_epi32(v, _mm256_min_epu32(v, _mm256_set1_epi32(255)));
    let large = _mm256_cmpeq_epi32(v, _mm256_max_epu32(v, _mm256_set1_epi32(65536)));
    let small_bits = _mm256_movemask_ps(_mm256_castsi256_ps(small)) as u32;
    let large_bits = _mm256_movemask_ps(_mm256_castsi256_ps(large)) as u32;
    if small_bits != 0 {
        // Lanes that are not small look up SLOG2_TABLE[0], which is 0.
        let idx = _mm256_and_si256(v, small);
        let table = SLOG2_TABLE.as_ptr().cast::<i64>();
        let lo = _mm256_i32gather_epi64::<8>(table, _mm256_castsi256_si128(idx));
        let hi = _mm256_i32gather_epi64::<8>(table, _mm256_extracti128_si256::<1>(idx));
        *acc = _mm256_add_epi64(*acc, _mm256_add_epi64(lo, hi));
    }
    if (small_bits | large_bits) != 0xff {
        // v * (LOG2_TABLE[v >> log_cnt] + (log_cnt << 23)) + LOG_2_RECIPROCAL_FIXED * (v & (y - 1))
        // with log_cnt = floor(log2(v)) - 7 and y = 1 << log_cnt; the other lanes are zeroed.
        let v = _mm256_andnot_si256(_mm256_or_si256(small, large), v);
        let exponent = _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(v)));
        let log_cnt = _mm256_sub_epi32(exponent, _mm256_set1_epi32(127 + 7));
        let idx = _mm256_srlv_epi32(v, log_cnt);
        let log2 = _mm256_i32gather_epi32::<4>(LOG2_TABLE.as_ptr().cast(), idx);
        let term = _mm256_add_epi32(log2, _mm256_slli_epi32::<23>(log_cnt));
        let y1 = _mm256_sub_epi32(
            _mm256_sllv_epi32(_mm256_set1_epi32(1), log_cnt),
            _mm256_set1_epi32(1),
        );
        let corr = _mm256_mullo_epi32(_mm256_and_si256(v, y1), _mm256_set1_epi32(12102203));
        let even = _mm256_add_epi64(
            _mm256_mul_epu32(v, term),
            _mm256_and_si256(corr, _mm256_set1_epi64x(0xffff_ffff)),
        );
        let odd = _mm256_add_epi64(
            _mm256_mul_epu32(_mm256_srli_epi64::<32>(v), _mm256_srli_epi64::<32>(term)),
            _mm256_srli_epi64::<32>(corr),
        );
        *acc = _mm256_add_epi64(*acc, _mm256_add_epi64(even, odd));
    }
    large_bits
}

#[inline(always)]
unsafe fn add_large(v: __m256i, mut bits: u32) -> u64 {
    let mut lanes = [0u32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast(), v);
    let mut sum = 0u64;
    while bits != 0 {
        sum = sum.wrapping_add(fast_slog2(lanes[bits.trailing_zeros() as usize]));
        bits &= bits - 1;
    }
    sum
}

#[inline(always)]
unsafe fn hsum32(v: __m256i) -> u32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b01_00_11_10>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0b10_11_00_01>(s));
    _mm_cvtsi128_si32(s) as u32
}

#[inline(always)]
unsafe fn hsum64(v: __m256i) -> u64 {
    let s = _mm_add_epi64(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    _mm_cvtsi128_si64(s) as u64
}

/// `SlogHisto::combined_entropy`'s sum over the bins where `x` is nonzero: `fast_slog2(x)
/// + fast_slog2(x + y) - slog[i]` each, and the sum of `x`.
#[target_feature(enable = "avx2")]
unsafe fn combined_entropy_avx2(x: &[u32; 256], y: &[u32; 256], slog: &[u64; 256]) -> (u64, u32) {
    let zero = _mm256_setzero_si256();
    let (mut sum_x, mut acc, mut sub) = (zero, zero, zero);
    let mut large = 0u64;
    for i in (0..256).step_by(8) {
        let xv = load(x, i);
        if _mm256_testz_si256(xv, xv) != 0 {
            continue;
        }
        let is_zero = _mm256_cmpeq_epi32(xv, zero);
        let xy = _mm256_andnot_si256(is_zero, _mm256_add_epi32(xv, load(y, i)));
        sum_x = _mm256_add_epi32(sum_x, xv);
        let bits = add_slog2(xv, &mut acc);
        if bits != 0 {
            large = large.wrapping_add(add_large(xv, bits));
        }
        let bits = add_slog2(xy, &mut acc);
        if bits != 0 {
            large = large.wrapping_add(add_large(xy, bits));
        }
        let lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(is_zero));
        let hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(is_zero));
        let s = slog[i..i + 8].as_ptr().cast::<__m256i>();
        sub = _mm256_add_epi64(sub, _mm256_andnot_si256(lo, _mm256_loadu_si256(s)));
        sub = _mm256_add_epi64(sub, _mm256_andnot_si256(hi, _mm256_loadu_si256(s.add(1))));
    }
    (
        hsum64(_mm256_sub_epi64(acc, sub)).wrapping_add(large),
        hsum32(sum_x),
    )
}

/// `combined_entropy_avx2`, when the CPU has AVX2.
pub(super) fn combined_entropy(
    x: &[u32; 256],
    y: &[u32; 256],
    slog: &[u64; 256],
) -> Option<(u64, u32)> {
    if !crate::wide::has_avx2() {
        return None;
    }
    // SAFETY: the CPU supports AVX2.
    Some(unsafe { combined_entropy_avx2(x, y, slog) })
}

#[target_feature(enable = "avx2")]
unsafe fn chunk_stats_avx2(v: &[u32; 64]) -> (u32, u32, u64) {
    let (mut sum, mut max) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    let mut acc = _mm256_setzero_si256();
    let mut large = 0u64;
    for i in (0..64).step_by(8) {
        let x = load(v, i);
        sum = _mm256_add_epi32(sum, x);
        max = _mm256_max_epu32(max, x);
        if _mm256_testz_si256(x, x) == 0 {
            let bits = add_slog2(x, &mut acc);
            if bits != 0 {
                large = large.wrapping_add(add_large(x, bits));
            }
        }
    }
    let m = _mm_max_epu32(
        _mm256_castsi256_si128(max),
        _mm256_extracti128_si256::<1>(max),
    );
    let m = _mm_max_epu32(m, _mm_shuffle_epi32::<0b01_00_11_10>(m));
    let m = _mm_max_epu32(m, _mm_shuffle_epi32::<0b10_11_00_01>(m));
    (
        hsum32(sum),
        _mm_cvtsi128_si32(m) as u32,
        hsum64(acc).wrapping_add(large),
    )
}

/// The wrapping sum, the maximum and the sum of `fast_slog2` of `v`, when the CPU has AVX2.
pub(super) fn chunk_stats(v: &[u32; 64]) -> Option<(u32, u32, u64)> {
    if !crate::wide::has_avx2() {
        return None;
    }
    // SAFETY: the CPU supports AVX2.
    Some(unsafe { chunk_stats_avx2(v) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lossless::encoder::entropy::{sub_pixels, SlogHisto};
    use crate::lossless::encoder::predictor::predict;

    fn check_avx2<const M: usize>(cur: &[u32], up: &[u32]) {
        let n = cur.len() - 1;
        let (mut want, mut got) = (vec![0; n], vec![0; n]);
        for i in 0..n {
            want[i] = sub_pixels(
                cur[i + 1],
                predict::<M>(cur[i], up[i], up[i + 1], up[i + 2]),
            );
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
                    .map(|i| {
                        if i == clear_at {
                            rnd() & 0x00ff_ffff
                        } else {
                            rnd() | 0x0100_0000
                        }
                    })
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

    #[test]
    fn avx2_slog2_sums_match_scalar() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut seed = 0x6a09_e667u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let edges = [
            0u32,
            1,
            2,
            255,
            256,
            257,
            511,
            512,
            4095,
            4096,
            65535,
            65536,
            65537,
            u32::MAX,
        ];
        for round in 0..3000 {
            let mut pick = || match rnd() % 6 {
                0 => 0,
                1 => rnd() % 256,
                2 => 256 + rnd() % (65536 - 256),
                3 => edges[(rnd() % edges.len() as u32) as usize],
                4 => rnd() % (1 << 20),
                _ => {
                    if round % 2 == 0 {
                        0
                    } else {
                        rnd()
                    }
                }
            };
            let v: [u32; 64] = core::array::from_fn(|_| pick());
            let want = v.iter().fold((0u32, 0u32, 0u64), |(s, m, e), &x| {
                (s.wrapping_add(x), m.max(x), e.wrapping_add(fast_slog2(x)))
            });
            assert_eq!(chunk_stats(&v), Some(want), "{v:?}");
            let mut x: [u32; 256] = core::array::from_fn(|_| pick() >> (round % 8));
            if round % 3 == 0 {
                // Sparse, as a tile's histogram is: most groups of eight are all zero.
                for (i, v) in x.iter_mut().enumerate() {
                    if (i * 37 + round) % 64 >= round % 5 {
                        *v = 0;
                    }
                }
            }
            let y: [u32; 256] = core::array::from_fn(|_| pick() >> (round % 5));
            let (mut want, mut sx, mut sxy) = (0u64, 0u32, 0u32);
            for (&a, &b) in x.iter().zip(&y) {
                let ab = a.wrapping_add(b);
                sx = sx.wrapping_add(a);
                sxy = sxy.wrapping_add(ab);
                want = want
                    .wrapping_add(fast_slog2(a))
                    .wrapping_add(fast_slog2(ab));
            }
            let want = fast_slog2(sx)
                .wrapping_add(fast_slog2(sxy))
                .wrapping_sub(want);
            let mut h = SlogHisto::default();
            h.update(&y);
            assert_eq!(h.combined_entropy(&x), want);
        }
    }
}
