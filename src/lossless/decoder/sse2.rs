//! SSE2 kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_sse2.c`, bit-exact
//! with the portable ones in `reverse_transform`. They build where SSE2 is in the target's
//! baseline (x86_64, and x86 targets that enable it), so no runtime detection is needed and the
//! register operations are safe to call; on wasm32 with simd128 they build on
//! `crate::sse2_wasm`, which implements the intrinsics. Pixels are RGBA bytes; every kernel
//! works per channel, so the order is libwebp's. Memory goes through slices, through
//! `safe_unaligned_simd` for four pixels.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![forbid(unsafe_code)]

// Every entry enters its kernel with the instruction-set token of the target's baseline, which
// this keeps present: SSE2 on x86, simd128 on wasm32.
const _: () = assert!(cfg!(any(
    all(target_feature = "sse2", target_feature = "fxsr"),
    all(target_arch = "wasm32", target_feature = "simd128")
)));

/// The SSE2 token of the target's baseline.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
fn baseline() -> fearless_simd::x86::Sse2 {
    fearless_simd::Level::baseline().as_sse2().unwrap()
}

/// The simd128 token of the target's baseline.
#[cfg(target_arch = "wasm32")]
#[inline(always)]
fn baseline() -> fearless_simd::wasm32::WasmSimd128 {
    fearless_simd::Level::baseline().as_wasm_simd128().unwrap()
}

/// A `fearless_simd::kernel!` entered with the token `baseline` returns.
macro_rules! entry {
    (fn $name:ident($token:ident: Sse2 $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty)? { $($body:tt)* }) => {
        #[cfg(not(target_arch = "wasm32"))]
        fearless_simd::kernel! { fn $name($token: Sse2 $(, $arg: $ty)*) $(-> $ret)? { $($body)* } }
        #[cfg(target_arch = "wasm32")]
        fearless_simd::kernel! { fn $name($token: WasmSimd128 $(, $arg: $ty)*) $(-> $ret)? { $($body)* } }
    };
}

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use safe_unaligned_simd::x86 as mem;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64 as mem;
#[cfg(target_arch = "wasm32")]
use crate::sse2_wasm::*;
use std::ops::Range;

/// Four pixels of `b` from byte `at`.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn load4(b: &[u8], at: usize) -> __m128i {
    mem::_mm_loadu_si128(<&[u8; 16]>::try_from(&b[at..at + 16]).unwrap())
}

#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn store4(b: &mut [u8], at: usize, v: __m128i) {
    mem::_mm_storeu_si128(<&mut [u8; 16]>::try_from(&mut b[at..at + 16]).unwrap(), v)
}

/// The pixel of `b` at byte `at`.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn load1(b: &[u8], at: usize) -> __m128i {
    _mm_cvtsi32_si128(i32::from_ne_bytes(b[at..at + 4].try_into().unwrap()))
}

#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn store1(b: &mut [u8], at: usize, v: __m128i) {
    b[at..at + 4].copy_from_slice(&_mm_cvtsi128_si32(v).to_ne_bytes());
}

/// `(a + b) >> 1` per byte.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn average2(a: __m128i, b: __m128i) -> __m128i {
    _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), _mm_set1_epi8(1)))
}

/// Adds a prediction from the row above only, four pixels at a time: `pred(top, top_left,
/// top_right)`. The `n` pixels are at byte `out` of `b`, the row above at byte `top`.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn add_upper(b: &mut [u8], out: usize, top: usize, n: usize, pred: impl Fn(__m128i, __m128i, __m128i) -> __m128i) {
    let mut i = 0;
    while i + 4 <= n {
        let t = top + 4 * i;
        let p = pred(load4(b, t), load4(b, t - 4), load4(b, t + 4));
        let o = out + 4 * i;
        store4(b, o, _mm_add_epi8(load4(b, o), p));
        i += 4;
    }
    while i < n {
        let t = top + 4 * i;
        let p = pred(load1(b, t), load1(b, t - 4), load1(b, t + 4));
        let o = out + 4 * i;
        store1(b, o, _mm_add_epi8(load1(b, o), p));
        i += 1;
    }
}

/// Adds a prediction that takes the pixel decoded just before, a pixel at a time:
/// `pred(left, top, top_left, top_right)`.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn add_serial(b: &mut [u8], out: usize, top: usize, n: usize, pred: impl Fn(__m128i, __m128i, __m128i, __m128i) -> __m128i) {
    let mut left = load1(b, out - 4);
    for i in 0..n {
        let t = top + 4 * i;
        let p = pred(left, load1(b, t), load1(b, t - 4), load1(b, t + 4));
        let o = out + 4 * i;
        left = _mm_add_epi8(load1(b, o), p);
        store1(b, o, left);
    }
}

/// Predictor 11, select: the left pixel where it is further from the top-left than the top
/// pixel is. Both sums with the residual are taken before the select, so the next left pixel
/// waits only on the left one's distance.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn add_select(b: &mut [u8], out: usize, top: usize, n: usize) {
    let mut left = load1(b, out - 4);
    for i in 0..n {
        let t = top + 4 * i;
        let o = out + 4 * i;
        let (up, up_left, src) = (load1(b, t), load1(b, t - 4), load1(b, o));
        let mask = _mm_cmpgt_epi32(_mm_sad_epu8(left, up_left), _mm_sad_epu8(up, up_left));
        let (from_left, from_up) = (_mm_add_epi8(src, left), _mm_add_epi8(src, up));
        left = _mm_or_si128(_mm_and_si128(mask, from_left), _mm_andnot_si128(mask, from_up));
        store1(b, o, left);
    }
}

/// Predictor 1, the left pixel: a prefix sum over each four pixels.
#[inline]
#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn add_left(b: &mut [u8], out: usize, n: usize) {
    let mut prev = _mm_set1_epi32(i32::from_ne_bytes(b[out - 4..out].try_into().unwrap()));
    let mut i = 0;
    while i + 4 <= n {
        let o = out + 4 * i;
        let src = load4(b, o);
        let sum0 = _mm_add_epi8(src, _mm_slli_si128::<4>(src));
        let sum1 = _mm_add_epi8(sum0, _mm_slli_si128::<8>(sum0));
        store4(b, o, _mm_add_epi8(sum1, prev));
        // The next left pixel waits on one add, the four pixels' sum taken beside it.
        prev = _mm_add_epi8(prev, _mm_shuffle_epi32::<0xff>(sum1));
        i += 4;
    }
    while i < n {
        let o = out + 4 * i;
        prev = _mm_add_epi8(load1(b, o), prev);
        store1(b, o, prev);
        i += 1;
    }
}

/// Predictor 1 on `image_data[range]`, which has a pixel before it.
#[inline]
pub(crate) fn predict_left(image_data: &mut [u8], range: Range<usize>) {
    entry! {
        fn enter(_token: Sse2, image_data: &mut [u8], range: Range<usize>) {
            predict_left_sse2(image_data, range)
        }
    }
    enter(baseline(), image_data, range)
}

#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn predict_left_sse2(image_data: &mut [u8], range: Range<usize>) {
    assert!(range.start >= 4 && range.start <= range.end && range.end <= image_data.len());
    add_left(image_data, range.start, (range.end - range.start) / 4)
}

/// Undoes predictor `mode` (0 to 15) over `image_data[range]`, the pixels of one row from its
/// second on, below the row above at `width` pixels before them.
#[inline]
pub(crate) fn predict(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    entry! {
        fn enter(_token: Sse2, mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
            predict_sse2(mode, image_data, range, width)
        }
    }
    enter(baseline(), mode, image_data, range, width)
}

#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn predict_sse2(mode: u8, image_data: &mut [u8], range: Range<usize>, width: usize) {
    let stride = width * 4;
    // The top-left pixel is read 4 bytes before the row above, the top-right of the last pixel
    // at most at `range.end - stride + 4`.
    assert!(range.start >= stride + 4 && range.start <= range.end && range.end <= image_data.len());
    let n = (range.end - range.start) / 4;
    let (b, out) = (image_data, range.start);
    let top = out - stride;
    match mode {
        1 => add_left(b, out, n),
        2 => add_upper(b, out, top, n, |t, _, _| t),
        3 => add_upper(b, out, top, n, |_, _, tr| tr),
        4 => add_upper(b, out, top, n, |_, tl, _| tl),
        5 => add_serial(b, out, top, n, |l, t, _, tr| average2(average2(l, tr), t)),
        6 => add_serial(b, out, top, n, |l, _, tl, _| average2(l, tl)),
        7 => add_serial(b, out, top, n, |l, t, _, _| average2(l, t)),
        8 => add_upper(b, out, top, n, |t, tl, _| average2(tl, t)),
        9 => add_upper(b, out, top, n, |t, _, tr| average2(t, tr)),
        10 => add_serial(b, out, top, n, |l, t, tl, tr| average2(average2(l, tl), average2(t, tr))),
        11 => add_select(b, out, top, n),
        // `clamp(l + t - tl)` as `l` saturated up by `t - tl` and down by `tl - t`, one of
        // them zero: the left pixel waits on two byte ops, not an unpack and a pack.
        12 => add_serial(b, out, top, n, |l, t, tl, _| {
            _mm_subs_epu8(_mm_adds_epu8(l, _mm_subs_epu8(t, tl)), _mm_subs_epu8(tl, t))
        }),
        13 => add_serial(b, out, top, n, |l, t, tl, _| {
            let zero = _mm_setzero_si128();
            let tl = _mm_unpacklo_epi8(tl, zero);
            let avg = _mm_srli_epi16::<1>(_mm_add_epi16(_mm_unpacklo_epi8(t, zero), _mm_unpacklo_epi8(l, zero)));
            // (avg - tl) / 2, rounded toward zero as C divides.
            let d = _mm_sub_epi16(_mm_sub_epi16(avg, tl), _mm_cmpgt_epi16(tl, avg));
            let s = _mm_add_epi16(avg, _mm_srai_epi16::<1>(d));
            _mm_packus_epi16(s, s)
        }),
        _ => add_upper(b, out, top, n, |_, _, _| _mm_set1_epi32(0xff00_0000u32 as i32)),
    }
}

/// The colour transform over whole pixels of `row`, with one multiplier triple (red to blue,
/// green to blue, green to red) per block of `block` pixels from `mults`.
#[inline]
pub(crate) fn color_transform_row(row: &mut [u8], block: usize, mults: &[u8]) {
    entry! {
        fn enter(_token: Sse2, row: &mut [u8], block: usize, mults: &[u8]) {
            color_transform_row_sse2(row, block, mults)
        }
    }
    enter(baseline(), row, block, mults)
}

#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn color_transform_row_sse2(row: &mut [u8], block: usize, mults: &[u8]) {
    let n = row.len() / 4;
    assert!(mults.len() >= n.div_ceil(block) * 4);
    for (b, m) in row[..n * 4].chunks_mut(block * 4).zip(mults.chunks_exact(4)) {
        color_transform(b, m[0], m[1], m[2])
    }
}

/// libwebp's `TransformColorInverse_SSE2` over the whole pixels of `pixels`, with red and blue
/// where RGBA keeps them: the deltas `(t * c) >> 5` of signed bytes are the high halves of the
/// 16-bit products of `c << 8` and `t << 3`.
#[inline]
pub(crate) fn color_transform(pixels: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
    entry! {
        fn enter(_token: Sse2, pixels: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
            color_transform_sse2(pixels, red_to_blue, green_to_blue, green_to_red)
        }
    }
    enter(baseline(), pixels, red_to_blue, green_to_blue, green_to_red)
}

#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn color_transform_sse2(pixels: &mut [u8], red_to_blue: u8, green_to_blue: u8, green_to_red: u8) {
    let n = pixels.len() / 4;
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
        store4(pixels, 4 * i, step(load4(pixels, 4 * i)));
        i += 4;
    }
    while i < n {
        store1(pixels, 4 * i, step(load1(pixels, 4 * i)));
        i += 1;
    }
}

/// Adds green to red and blue.
#[inline]
pub(crate) fn add_green(image_data: &mut [u8]) {
    entry! {
        fn enter(_token: Sse2, image_data: &mut [u8]) {
            add_green_sse2(image_data)
        }
    }
    enter(baseline(), image_data)
}

#[cfg_attr(not(target_arch = "wasm32"), target_feature(enable = "sse2"))]
fn add_green_sse2(image_data: &mut [u8]) {
    let n = image_data.len() / 4;
    let step = |v: __m128i| {
        let g = _mm_and_si128(_mm_srli_epi32::<8>(v), _mm_set1_epi32(0xff));
        _mm_add_epi8(v, _mm_or_si128(g, _mm_slli_epi32::<16>(g)))
    };
    let mut i = 0;
    while i + 4 <= n {
        store4(image_data, 4 * i, step(load4(image_data, 4 * i)));
        i += 4;
    }
    while i < n {
        store1(image_data, 4 * i, step(load1(image_data, 4 * i)));
        i += 1;
    }
}
