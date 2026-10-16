//! Utilities for doing the YUV -> RGB conversion
//! The images are encoded in the Y'CbCr format as detailed here: <https://en.wikipedia.org/wiki/YCbCr>
//! so need to be converted to RGB to be displayed
//! To do the YUV -> RGB conversion we need to first decide how to map the yuv values to the pixels
//! The y buffer is the same size as the pixel buffer so that maps 1-1 but the
//! u and v buffers are half the size of the pixel buffer so we need to scale it up
//! The simple way to upscale is just to take each u/v value and associate it with the 4
//! pixels around it e.g. for a 4x4 image:
//!
//! ||||||
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! |yyyy|
//! ||||||
//!
//! |||||||
//! |uu|vv|
//! |uu|vv|
//! |||||||
//!
//! Then each of the 2x2 pixels would match the u/v from the same quadrant
//!
//! However fancy upsampling is the default for libwebp which does a little more work to make the values smoother
//! It interpolates u and v so that for e.g. the pixel 1 down and 1 from the left the u value
//! would be (9*u0 + 3*u1 + 3*u2 + u3 + 8) / 16 and similar for the other pixels
//! The edges are mirrored, so for the pixel 1 down and 0 from the left it uses (9*u0 + 3*u2 + 3*u0 + u2 + 8) / 16

/// `_mm_mulhi_epu16` emulation
#[inline(always)]
fn mulhi(v: u8, coeff: u16) -> i32 {
    ((u32::from(v) * u32::from(coeff)) >> 8) as i32
}

/// This function has been rewritten to encourage auto-vectorization.
///
/// Based on [src/dsp/yuv.h](https://github.com/webmproject/libwebp/blob/8534f53960befac04c9631e6e50d21dcb42dfeaf/src/dsp/yuv.h#L79)
/// from the libwebp source.
/// ```text
/// const YUV_FIX2: i32 = 6;
/// const YUV_MASK2: i32 = (256 << YUV_FIX2) - 1;
/// fn clip(v: i32) -> u8 {
///     if (v & !YUV_MASK2) == 0 {
///         (v >> YUV_FIX2) as u8
///     } else if v < 0 {
///         0
///     } else {
///         255
///     }
/// }
/// ```
// Clippy suggests the clamp method, but it seems to optimize worse as of rustc 1.82.0 nightly.
#[allow(clippy::manual_clamp)]
#[inline(always)]
fn clip(v: i32) -> u8 {
    const YUV_FIX2: i32 = 6;
    (v >> YUV_FIX2).max(0).min(255) as u8
}

#[inline(always)]
fn yuv_to_r(y: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(v, 26149) - 14234)
}

#[inline(always)]
fn yuv_to_g(y: u8, u: u8, v: u8) -> u8 {
    clip(mulhi(y, 19077) - mulhi(u, 6419) - mulhi(v, 13320) + 8708)
}

#[inline(always)]
fn yuv_to_b(y: u8, u: u8) -> u8 {
    clip(mulhi(y, 19077) + mulhi(u, 33050) - 17685)
}


/// Upsamples a chroma row for one luma row: `near` is the chroma row closest to it, `far`
/// the other one (or `near` again at the image edges), with the same weights as
/// `get_fancy_chroma_value`.
pub(crate) fn upsample_chroma_row(near: &[u8], far: &[u8], out: &mut [u8]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let start = super::simd::upsample_pairs(near, far, out);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let start = 0;
    upsample_chroma_row_from(near, far, out, start);
}

/// `upsample_chroma_row` with the chroma pairs before `start` already done.
fn upsample_chroma_row_from(near: &[u8], far: &[u8], out: &mut [u8], start: usize) {
    let n = near.len();
    let far = &far[..n];
    let mix = |j: usize| 3 * u16::from(near[j]) + u16::from(far[j]);
    out[0] = ((4 * mix(0) + 8) >> 4) as u8;
    let width = out.len();
    let pairs = ((width - 1) / 2).min(n - 1);
    for j in start..pairs {
        let (a, b) = (mix(j), mix(j + 1));
        out[2 * j + 1] = ((3 * a + b + 8) >> 4) as u8;
        out[2 * j + 2] = ((3 * b + a + 8) >> 4) as u8;
    }
    if width % 2 == 0 && width > 1 {
        out[width - 1] = ((4 * mix(n - 1) + 8) >> 4) as u8;
    }
}

/// RGB(A) of a row from its luma and full-width chroma; RGBA alpha may be set to 255.
pub(crate) fn fill_row_yuv<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row: &[u8],
    v_row: &[u8],
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let done = super::simd::yuv_to_rgb_row::<BPP>(y_row, u_row, v_row, row_buffer);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let done = 0;
    fill_row_yuv_portable::<BPP>(
        &mut row_buffer[done * BPP..],
        &y_row[done..],
        &u_row[done..],
        &v_row[done..],
    );
}

fn fill_row_yuv_portable<const BPP: usize>(
    row_buffer: &mut [u8],
    y_row: &[u8],
    u_row: &[u8],
    v_row: &[u8],
) {
    for (((rgb, &y), &u), &v) in row_buffer
        .chunks_exact_mut(BPP)
        .zip(y_row)
        .zip(u_row)
        .zip(v_row)
    {
        set_pixel(rgb, y, u, v);
    }
}




#[inline(always)]
fn set_pixel(rgb: &mut [u8], y: u8, u: u8, v: u8) {
    rgb[0] = yuv_to_r(y, v);
    rgb[1] = yuv_to_g(y, u, v);
    rgb[2] = yuv_to_b(y, u);
}

/// Simple conversion, not currently used but could add a config to allow for using the simple
#[allow(unused)]
pub(crate) fn fill_rgb_buffer_simple<const BPP: usize>(
    buffer: &mut [u8],
    y_buffer: &[u8],
    u_buffer: &[u8],
    v_buffer: &[u8],
    width: usize,
    chroma_width: usize,
    buffer_width: usize,
) {
    let u_row_twice_iter = u_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| std::iter::repeat(n).take(2));
    let v_row_twice_iter = v_buffer
        .chunks_exact(buffer_width / 2)
        .flat_map(|n| std::iter::repeat(n).take(2));

    for (((row, y_row), u_row), v_row) in buffer
        .chunks_exact_mut(width * BPP)
        .zip(y_buffer.chunks_exact(buffer_width))
        .zip(u_row_twice_iter)
        .zip(v_row_twice_iter)
    {
        fill_rgba_row_simple::<BPP>(
            &y_row[..width],
            &u_row[..chroma_width],
            &v_row[..chroma_width],
            row,
        );
    }
}

pub(crate) fn fill_rgba_row_simple<const BPP: usize>(
    y_vec: &[u8],
    u_vec: &[u8],
    v_vec: &[u8],
    rgba: &mut [u8],
) {
    // Fill 2 pixels per iteration: these pixels share `u` and `v` components
    let mut rgb_chunks = rgba.chunks_exact_mut(BPP * 2);
    let mut y_chunks = y_vec.chunks_exact(2);
    let mut u_iter = u_vec.iter();
    let mut v_iter = v_vec.iter();

    for (((rgb, y), &u), &v) in (&mut rgb_chunks)
        .zip(&mut y_chunks)
        .zip(&mut u_iter)
        .zip(&mut v_iter)
    {
        let coeffs = [
            mulhi(v, 26149),
            mulhi(u, 6419),
            mulhi(v, 13320),
            mulhi(u, 33050),
        ];

        let get_r = |y: u8| clip(mulhi(y, 19077) + coeffs[0] - 14234);
        let get_g = |y: u8| clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
        let get_b = |y: u8| clip(mulhi(y, 19077) + coeffs[3] - 17685);

        let rgb1 = &mut rgb[0..3];
        rgb1[0] = get_r(y[0]);
        rgb1[1] = get_g(y[0]);
        rgb1[2] = get_b(y[0]);

        let rgb2 = &mut rgb[BPP..];
        rgb2[0] = get_r(y[1]);
        rgb2[1] = get_g(y[1]);
        rgb2[2] = get_b(y[1]);
    }

    let remainder = rgb_chunks.into_remainder();
    if remainder.len() >= 3 {
        if let (Some(&y), Some(&u), Some(&v)) = (
            y_chunks.remainder().iter().next(),
            u_iter.next(),
            v_iter.next(),
        ) {
            let coeffs = [
                mulhi(v, 26149),
                mulhi(u, 6419),
                mulhi(v, 13320),
                mulhi(u, 33050),
            ];

            remainder[0] = clip(mulhi(y, 19077) + coeffs[0] - 14234);
            remainder[1] = clip(mulhi(y, 19077) - coeffs[1] - coeffs[2] + 8708);
            remainder[2] = clip(mulhi(y, 19077) + coeffs[3] - 17685);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yuv_conversions() {
        let (y, u, v) = (203, 40, 42);

        assert_eq!(yuv_to_r(y, v), 80);
        assert_eq!(yuv_to_g(y, u, v), 255);
        assert_eq!(yuv_to_b(y, u), 40);
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn simd_rows_match_portable() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        for len in [0usize, 1, 7, 8, 9, 31, 32, 33, 64, 77, 200] {
            let y: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let u: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let v: Vec<u8> = (0..len).map(|_| rng.gen()).collect();
            let mut a = vec![255u8; len * 3];
            let mut b = vec![255u8; len * 3];
            fill_row_yuv::<3>(&mut a, &y, &u, &v);
            fill_row_yuv_portable::<3>(&mut b, &y, &u, &v);
            assert_eq!(a, b, "rgb {len}");
            let mut a = vec![255u8; len * 4];
            let mut b = vec![255u8; len * 4];
            fill_row_yuv::<4>(&mut a, &y, &u, &v);
            fill_row_yuv_portable::<4>(&mut b, &y, &u, &v);
            assert_eq!(a, b, "rgba {len}");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn simd_upsampling_matches_portable() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(6);
        for round in 0usize..1200 {
            let width = 1 + round % 200;
            let n = width.div_ceil(2);
            let mut sample = || {
                if round % 3 == 0 {
                    rng.gen_range(0..2) * 255
                } else {
                    rng.gen()
                }
            };
            let near: Vec<u8> = (0..n).map(|_| sample()).collect();
            let far: Vec<u8> = (0..n).map(|_| sample()).collect();
            let mut a = vec![0u8; width];
            let mut b = vec![0u8; width];
            upsample_chroma_row(&near, &far, &mut a);
            upsample_chroma_row_from(&near, &far, &mut b, 0);
            assert_eq!(a, b, "{width}");
        }
    }
}
