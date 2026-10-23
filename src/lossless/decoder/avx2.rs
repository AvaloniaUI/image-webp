//! AVX2 kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_avx2.c`, run behind
//! [`crate::wide::has_avx2`]. Each takes whole groups of eight pixels, leaving the rest to the
//! `sse2` kernel it widens. Pixels are RGBA bytes.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::x86_64::*;

use super::sse2;

#[inline(always)]
unsafe fn load8(p: *const u8) -> __m256i {
    _mm256_loadu_si256(p.cast())
}

#[inline(always)]
unsafe fn store8(p: *mut u8, v: __m256i) {
    _mm256_storeu_si256(p.cast(), v)
}

/// libwebp's `TransformColorInverse_AVX2` over the first whole eights of the `n` pixels at `p`,
/// with red and blue where RGBA keeps them.
#[inline(always)]
unsafe fn color_transform8(p: *mut u8, n: usize, red_to_blue: u8, green_to_blue: u8, green_to_red: u8) -> usize {
    let cst = |m: u8| i32::from(((m as i8 as i16) << 3) as u16);
    let mults_rb = _mm256_set1_epi32(cst(green_to_blue) << 16 | cst(green_to_red));
    let mults_b2 = _mm256_set1_epi32(cst(red_to_blue) << 16);
    let mask_ag = _mm256_set1_epi32(0xff00ff00u32 as i32);
    #[rustfmt::skip]
    let perm1 = _mm256_setr_epi8(
        -1, 1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13,
        -1, 1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13,
    );
    #[rustfmt::skip]
    let perm2 = _mm256_setr_epi8(
        -1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12,
        -1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1, -1, 12,
    );
    let mut i = 0;
    while i + 8 <= n {
        let a = load8(p.add(4 * i));
        let d = _mm256_add_epi8(a, _mm256_mulhi_epi16(_mm256_shuffle_epi8(a, perm1), mults_rb));
        let g = _mm256_add_epi8(d, _mm256_mulhi_epi16(_mm256_shuffle_epi8(d, perm2), mults_b2));
        store8(p.add(4 * i), _mm256_blendv_epi8(g, a, mask_ag));
        i += 8;
    }
    i
}

/// `sse2::color_transform_row` with the kernel above.
///
/// # Safety
/// The CPU supports AVX2, and `mults` has a triple for each block of `row`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    for (b, m) in row.chunks_mut(block * 4).zip(mults.as_chunks::<4>().0) {
        let (p, n) = (b.as_mut_ptr(), b.len() / 4);
        let done = color_transform8(p, n, m[0], m[1], m[2]);
        sse2::color_transform(p.add(4 * done), n - done, m[0], m[1], m[2]);
    }
}

/// libwebp's `AddGreenToBlueAndRed_AVX2` over the first whole eights of the `n` pixels at `p`.
///
/// # Safety
/// The CPU supports AVX2, and `n` pixels at `p`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_green(p: *mut u8, n: usize) -> usize {
    #[rustfmt::skip]
    let green = _mm256_setr_epi8(
        1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1,
        1, -1, 1, -1, 5, -1, 5, -1, 9, -1, 9, -1, 13, -1, 13, -1,
    );
    let mut i = 0;
    while i + 8 <= n {
        let v = load8(p.add(4 * i));
        store8(p.add(4 * i), _mm256_add_epi8(v, _mm256_shuffle_epi8(v, green)));
        i += 8;
    }
    i
}

/// The eight-pixel kernel alone, for the tests.
#[cfg(test)]
#[target_feature(enable = "avx2")]
unsafe fn color_transform8_alone(p: *mut u8, n: usize, m: [u8; 4]) -> usize {
    color_transform8(p, n, m[0], m[1], m[2])
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

    use super::super::reverse_transform as scalar;

    fn random(rng: &mut impl Rng, len: usize) -> Vec<u8> {
        let mut data = vec![0u8; len];
        rng.fill(&mut data[..]);
        // Small values and equal neighbours bring the ties and wraps out.
        if rng.gen() {
            data.iter_mut().for_each(|b| *b = (*b % 4).wrapping_sub(2));
        }
        data
    }

    #[test]
    fn color_transform_rows_match_the_portable_ones() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(14);
        for width in [1u16, 5, 9, 17, 64, 67, 130] {
            for size_bits in 2..=6 {
                let data = random(&mut rng, usize::from(width) * 4);
                let mut mults = vec![0u8; usize::from(super::super::subsample_size(width, size_bits)) * 4];
                rng.fill(&mut mults[..]);
                let (mut a, mut b) = (data.clone(), data);
                scalar::apply_color_transform_portable(&mut a, width, 0, size_bits, &mults);
                unsafe { super::color_transform_row(&mut b, 1 << size_bits, &mults) };
                assert_eq!(a, b, "width {width} size_bits {size_bits}");
            }
        }
    }

    #[test]
    fn color_transform_and_add_green_match_the_portable_ones() {
        if !crate::wide::has_avx2() {
            return;
        }
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);
        for width in [1u16, 7, 8, 9, 16, 23, 64, 67, 255] {
            for _ in 0..20 {
                let data = random(&mut rng, usize::from(width) * 4);
                let mults: [u8; 4] = rng.gen();
                let (mut a, mut b) = (data.clone(), data.clone());
                scalar::apply_color_transform_portable(&mut a, width, 0, 9, &mults);
                let n = usize::from(width);
                let done = unsafe { super::color_transform8_alone(b.as_mut_ptr(), n, mults) };
                assert_eq!(done, n / 8 * 8);
                assert_eq!(a[..done * 4], b[..done * 4], "width {width} {mults:?}");
                assert_eq!(data[done * 4..], b[done * 4..]);
                let (mut a, mut b) = (data.clone(), data.clone());
                scalar::apply_subtract_green_portable(&mut a);
                let done = unsafe { super::add_green(b.as_mut_ptr(), n) };
                assert_eq!(done, n / 8 * 8);
                assert_eq!(a[..done * 4], b[..done * 4]);
                assert_eq!(data[done * 4..], b[done * 4..]);
            }
        }
    }
}
