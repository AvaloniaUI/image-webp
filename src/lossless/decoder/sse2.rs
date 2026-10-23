//! SSE2 kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_sse2.c`, bit-exact
//! with the portable ones in `reverse_transform`. SSE2 is part of x86_64, so no runtime detection
//! is needed. Pixels are RGBA bytes; every kernel works per channel, so the order is libwebp's.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::x86_64::*;
use std::ops::Range;

#[inline(always)]
unsafe fn load4(p: *const u8) -> __m128i {
    _mm_loadu_si128(p.cast())
}

#[inline(always)]
unsafe fn store4(p: *mut u8, v: __m128i) {
    _mm_storeu_si128(p.cast(), v)
}

#[inline(always)]
unsafe fn load1(p: *const u8) -> __m128i {
    _mm_cvtsi32_si128(p.cast::<i32>().read_unaligned())
}

#[inline(always)]
unsafe fn store1(p: *mut u8, v: __m128i) {
    p.cast::<i32>().write_unaligned(_mm_cvtsi128_si32(v))
}

/// `(a + b) >> 1` per byte.
#[inline(always)]
unsafe fn average2(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1)))
}

/// Adds a prediction from the row above only, four pixels at a time: `pred(top, top_left, top_right)`.
#[inline(always)]
unsafe fn add_upper(out: *mut u8, top: *const u8, n: usize, pred: impl Fn(__m128i, __m128i, __m128i) -> __m128i) {
    let mut i = 0;
    while i + 4 <= n {
        let t = top.add(4 * i);
        let p = pred(load4(t), load4(t.sub(4)), load4(t.add(4)));
        store4(out.add(4 * i), _mm_add_epi8(load4(out.add(4 * i)), p));
        i += 4;
    }
    while i < n {
        let t = top.add(4 * i);
        let p = pred(load1(t), load1(t.sub(4)), load1(t.add(4)));
        store1(out.add(4 * i), _mm_add_epi8(load1(out.add(4 * i)), p));
        i += 1;
    }
}

/// Adds a prediction that takes the pixel decoded just before, a pixel at a time:
/// `pred(left, top, top_left, top_right)`.
#[inline(always)]
unsafe fn add_serial(out: *mut u8, top: *const u8, n: usize, pred: impl Fn(__m128i, __m128i, __m128i, __m128i) -> __m128i) {
    let mut left = load1(out.sub(4));
    for i in 0..n {
        let t = top.add(4 * i);
        let p = pred(left, load1(t), load1(t.sub(4)), load1(t.add(4)));
        left = _mm_add_epi8(load1(out.add(4 * i)), p);
        store1(out.add(4 * i), left);
    }
}

/// Predictor 1, the left pixel: a prefix sum over each four pixels.
#[inline(always)]
unsafe fn add_left(out: *mut u8, n: usize) {
    let mut prev = _mm_set1_epi32(out.sub(4).cast::<i32>().read_unaligned());
    let mut i = 0;
    while i + 4 <= n {
        let src = load4(out.add(4 * i));
        let sum0 = _mm_add_epi8(src, _mm_slli_si128::<4>(src));
        let sum1 = _mm_add_epi8(sum0, _mm_slli_si128::<8>(sum0));
        let res = _mm_add_epi8(sum1, prev);
        store4(out.add(4 * i), res);
        prev = _mm_shuffle_epi32::<0xff>(res);
        i += 4;
    }
    while i < n {
        prev = _mm_add_epi8(load1(out.add(4 * i)), prev);
        store1(out.add(4 * i), prev);
        i += 1;
    }
}

/// Predictor 1 on `image_data[range]`, which has a pixel before it.
pub(crate) fn predict_left(image_data: &mut [u8], range: Range<usize>) {
    assert!(range.start >= 4 && range.start <= range.end && range.end <= image_data.len());
    unsafe { add_left(image_data.as_mut_ptr().add(range.start), (range.end - range.start) / 4) }
}

/// Undoes predictor `mode` (0 to 15) over `image_data[range]`, the pixels of one row from its
/// second on, below the row above at `width` pixels before them.
pub(crate) fn predict(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    let stride = width * 4;
    // The top-left pixel is read 4 bytes before the row above, the top-right of the last pixel
    // at most at `range.end - stride + 4`.
    assert!(range.start >= stride + 4 && range.start <= range.end && range.end <= image_data.len());
    let n = (range.end - range.start) / 4;
    unsafe {
        let out = image_data.as_mut_ptr().add(range.start);
        let top = out.sub(stride).cast_const();
        match mode {
            1 => add_left(out, n),
            2 => add_upper(out, top, n, |t, _, _| t),
            3 => add_upper(out, top, n, |_, _, tr| tr),
            4 => add_upper(out, top, n, |_, tl, _| tl),
            5 => add_serial(out, top, n, |l, t, _, tr| average2(average2(l, tr), t)),
            6 => add_serial(out, top, n, |l, _, tl, _| average2(l, tl)),
            7 => add_serial(out, top, n, |l, t, _, _| average2(l, t)),
            8 => add_upper(out, top, n, |t, tl, _| average2(tl, t)),
            9 => add_upper(out, top, n, |t, _, tr| average2(t, tr)),
            10 => add_serial(out, top, n, |l, t, tl, tr| average2(average2(l, tl), average2(t, tr))),
            11 => add_serial(out, top, n, |l, t, tl, _| {
                // The left pixel where it is further from the top-left than the top pixel is.
                let pa = _mm_sad_epu8(t, tl);
                let pb = _mm_sad_epu8(l, tl);
                let mask = _mm_cmpgt_epi32(pb, pa);
                _mm_or_si128(_mm_and_si128(mask, l), _mm_andnot_si128(mask, t))
            }),
            12 => add_serial(out, top, n, |l, t, tl, _| {
                let zero = _mm_setzero_si128();
                let d = _mm_sub_epi16(_mm_unpacklo_epi8(t, zero), _mm_unpacklo_epi8(tl, zero));
                let s = _mm_add_epi16(_mm_unpacklo_epi8(l, zero), d);
                _mm_packus_epi16(s, s)
            }),
            13 => add_serial(out, top, n, |l, t, tl, _| {
                let zero = _mm_setzero_si128();
                let tl = _mm_unpacklo_epi8(tl, zero);
                let avg = _mm_srli_epi16::<1>(_mm_add_epi16(_mm_unpacklo_epi8(t, zero), _mm_unpacklo_epi8(l, zero)));
                // (avg - tl) / 2, rounded toward zero as C divides.
                let d = _mm_sub_epi16(_mm_sub_epi16(avg, tl), _mm_cmpgt_epi16(tl, avg));
                let s = _mm_add_epi16(avg, _mm_srai_epi16::<1>(d));
                _mm_packus_epi16(s, s)
            }),
            _ => add_upper(out, top, n, |_, _, _| _mm_set1_epi32(0xff00_0000u32 as i32)),
        }
    }
}

/// The colour transform over whole pixels of `row`, with one multiplier triple (red to blue,
/// green to blue, green to red) per block of `block` pixels from `mults`.
pub(crate) fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    let n = row.len() / 4;
    assert!(mults.len() >= n.div_ceil(block) * 4);
    for (b, m) in row.chunks_mut(block * 4).zip(mults.chunks_exact(4)) {
        unsafe { color_transform(b.as_mut_ptr(), b.len() / 4, m[0], m[1], m[2]) }
    }
}

/// libwebp's `TransformColorInverse_SSE2`, with red and blue where RGBA keeps them: the deltas
/// `(t * c) >> 5` of signed bytes are the high halves of the 16-bit products of `c << 8` and `t << 3`.
#[inline(always)]
unsafe fn color_transform(p: *mut u8, n: usize, red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
    let cst = |m: u8| i32::from(((m as i8 as i16) << 3) as u16);
    let mults_rb = _mm_set1_epi32(cst(green_to_blue) << 16 | cst(green_to_red));
    let mults_b2 = _mm_set1_epi32(cst(red_to_blue));
    let mask_ag = _mm_set1_epi32(0xff00ff00u32 as i32);
    let step = |input: __m128i| {
        let ag = _mm_and_si128(input, mask_ag);
        let g = _mm_shufflehi_epi16::<0xa0>(_mm_shufflelo_epi16::<0xa0>(ag));
        let rb = _mm_add_epi8(input, _mm_mulhi_epi16(g, mults_rb));
        // Red and blue in the high bytes of their halves; the red delta of blue goes to byte 3.
        let rb = _mm_slli_epi16::<8>(rb);
        let delta = _mm_slli_epi32::<24>(_mm_mulhi_epi16(rb, mults_b2));
        _mm_or_si128(_mm_srli_epi16::<8>(_mm_add_epi8(rb, delta)), ag)
    };
    let mut i = 0;
    while i + 4 <= n {
        store4(p.add(4 * i), step(load4(p.add(4 * i))));
        i += 4;
    }
    while i < n {
        store1(p.add(4 * i), step(load1(p.add(4 * i))));
        i += 1;
    }
}

/// Adds green to red and blue.
pub(crate) fn add_green(image_data: &mut [u8]) {
    let n = image_data.len() / 4;
    let p = image_data.as_mut_ptr();
    let step = |v: __m128i| unsafe {
        let g = _mm_and_si128(_mm_srli_epi32::<8>(v), _mm_set1_epi32(0xff));
        _mm_add_epi8(v, _mm_or_si128(g, _mm_slli_epi32::<16>(g)))
    };
    unsafe {
        let mut i = 0;
        while i + 4 <= n {
            store4(p.add(4 * i), step(load4(p.add(4 * i))));
            i += 4;
        }
        while i < n {
            store1(p.add(4 * i), step(load1(p.add(4 * i))));
            i += 1;
        }
    }
}
