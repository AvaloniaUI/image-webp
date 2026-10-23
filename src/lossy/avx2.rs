//! AVX2 decoder kernels: libwebp has none for VP8 decoding, so these run the SSE2 kernels of
//! `sse2.rs` 256 bits wide, bit-exact with them. Callers check [`crate::wide::has_avx2`] first.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

/// 16 bytes at `p` in the low lane, 16 at `p + 32` in the high one.
#[inline(always)]
unsafe fn load_lanes(p: *const u8) -> __m256i {
    _mm256_loadu2_m128i(p.add(32).cast(), p.cast())
}

/// `yuv444_to_rgb` of `sse2.rs` on 16 pixels whose luma and chroma are in the high bytes.
#[inline(always)]
unsafe fn yuv444_to_rgb(y0: __m256i, u0: __m256i, v0: __m256i) -> [__m256i; 3] {
    let y1 = _mm256_mulhi_epu16(y0, _mm256_set1_epi16(19077));
    let r0 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(26149));
    let r2 = _mm256_add_epi16(_mm256_sub_epi16(y1, _mm256_set1_epi16(14234)), r0);
    let g0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(6419));
    let g1 = _mm256_mulhi_epu16(v0, _mm256_set1_epi16(13320));
    let g4 = _mm256_sub_epi16(
        _mm256_add_epi16(y1, _mm256_set1_epi16(8708)),
        _mm256_add_epi16(g0, g1),
    );
    let b0 = _mm256_mulhi_epu16(u0, _mm256_set1_epi16(33050u16 as i16));
    let b2 = _mm256_subs_epu16(_mm256_adds_epu16(b0, y1), _mm256_set1_epi16(17685));
    [
        _mm256_srai_epi16::<6>(r2),
        _mm256_srai_epi16::<6>(g4),
        _mm256_srli_epi16::<6>(b2),
    ]
}

#[inline(always)]
unsafe fn planar_to_24b_step(i: [__m256i; 6]) -> [__m256i; 6] {
    let m = _mm256_set1_epi16(0x00ff);
    let even = |a, b| _mm256_packus_epi16(_mm256_and_si256(a, m), _mm256_and_si256(b, m));
    let odd = |a, b| _mm256_packus_epi16(_mm256_srli_epi16::<8>(a), _mm256_srli_epi16::<8>(b));
    [
        even(i[0], i[1]),
        even(i[2], i[3]),
        even(i[4], i[5]),
        odd(i[0], i[1]),
        odd(i[2], i[3]),
        odd(i[4], i[5]),
    ]
}

#[inline(always)]
unsafe fn store_lo(p: *mut u8, x: __m256i) {
    _mm_storeu_si128(p.cast(), _mm256_castsi256_si128(x));
}

#[inline(always)]
unsafe fn store_hi(p: *mut u8, x: __m256i) {
    _mm_storeu_si128(p.cast(), _mm256_extracti128_si256::<1>(x));
}

/// `sse2::yuv_to_rgb_row` 64 (RGB) or 16 (RGBA) pixels at a time: RGB runs libwebp's
/// `VP8YuvToRgb32_SSE2` on two blocks of 32 pixels, one in each lane.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn yuv_to_rgb_row<const BPP: usize>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u8],
) -> usize {
    let n = y.len().min(u.len()).min(v.len()).min(out.len() / BPP);
    let mut i = 0;
    let (y, u, v, out) = (y.as_ptr(), u.as_ptr(), v.as_ptr(), out.as_mut_ptr());
    let zero = _mm256_setzero_si256();
    if BPP == 3 {
        while i + 64 <= n {
            let [ya, ua, va] = [y, u, v].map(|p| load_lanes(p.add(i)));
            let [yb, ub, vb] = [y, u, v].map(|p| load_lanes(p.add(i + 16)));
            let lo = |x| _mm256_unpacklo_epi8(zero, x);
            let hi = |x| _mm256_unpackhi_epi8(zero, x);
            let [r0, g0, b0] = yuv444_to_rgb(lo(ya), lo(ua), lo(va));
            let [r1, g1, b1] = yuv444_to_rgb(hi(ya), hi(ua), hi(va));
            let [r2, g2, b2] = yuv444_to_rgb(lo(yb), lo(ub), lo(vb));
            let [r3, g3, b3] = yuv444_to_rgb(hi(yb), hi(ub), hi(vb));
            let mut c = [
                _mm256_packus_epi16(r0, r1),
                _mm256_packus_epi16(r2, r3),
                _mm256_packus_epi16(g0, g1),
                _mm256_packus_epi16(g2, g3),
                _mm256_packus_epi16(b0, b1),
                _mm256_packus_epi16(b2, b3),
            ];
            for _ in 0..5 {
                c = planar_to_24b_step(c);
            }
            for (k, x) in c.into_iter().enumerate() {
                store_lo(out.add(3 * i + 16 * k), x);
                store_hi(out.add(3 * i + 96 + 16 * k), x);
            }
            i += 64;
        }
    } else {
        let alpha = _mm256_set1_epi16(255);
        let load = |p: *const u8| {
            _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(_mm_loadu_si128(p.cast())))
        };
        while i + 16 <= n {
            let [r, g, b] = yuv444_to_rgb(load(y.add(i)), load(u.add(i)), load(v.add(i)));
            let rb = _mm256_packus_epi16(r, b);
            let ga = _mm256_packus_epi16(g, alpha);
            let rg = _mm256_unpacklo_epi8(rb, ga);
            let ba = _mm256_unpackhi_epi8(rb, ga);
            let (lo, hi) = (_mm256_unpacklo_epi16(rg, ba), _mm256_unpackhi_epi16(rg, ba));
            let p = out.add(4 * i);
            store_lo(p, lo);
            store_lo(p.add(16), hi);
            store_hi(p.add(32), lo);
            store_hi(p.add(48), hi);
            i += 16;
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    #[test]
    fn yuv_rows_match_sse2() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(9);
        for len in [0usize, 15, 16, 17, 63, 64, 65, 100, 128, 191, 300] {
            let [y, u, v] = [0; 3].map(|_| (0..len).map(|_| rng.gen()).collect::<Vec<u8>>());
            for bpp in [3, 4] {
                let mut a = vec![7u8; len * bpp];
                let mut b = vec![7u8; len * bpp];
                // SAFETY: the CPU has AVX2.
                let done = unsafe {
                    match bpp {
                        3 => super::yuv_to_rgb_row::<3>(&y, &u, &v, &mut a),
                        _ => super::yuv_to_rgb_row::<4>(&y, &u, &v, &mut a),
                    }
                };
                let step = if bpp == 3 { 64 } else { 16 };
                assert_eq!(done, len / step * step);
                match bpp {
                    3 => super::super::sse2::yuv_to_rgb_row_from::<3>(&y, &u, &v, &mut b, 0),
                    _ => super::super::sse2::yuv_to_rgb_row_from::<4>(&y, &u, &v, &mut b, 0),
                };
                assert_eq!(a[..done * bpp], b[..done * bpp], "bpp {bpp} len {len}");
            }
        }
    }
}
