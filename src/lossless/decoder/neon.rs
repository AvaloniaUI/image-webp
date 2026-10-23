//! NEON kernels of the VP8L inverse transforms after libwebp's `dsp/lossless_neon.c`, bit-exact
//! with the portable ones in `reverse_transform`. NEON is part of aarch64, so no runtime detection
//! is needed. Pixels are RGBA bytes; every predictor works per channel, so the order is libwebp's.
//!
//! libwebp: Copyright Google Inc., BSD 3-Clause, see LICENSE-LIBWEBP.
#![allow(unsafe_code)]

use core::arch::aarch64::*;
use std::ops::Range;

#[inline(always)]
unsafe fn load4(p: *const u8) -> uint8x16_t {
    vld1q_u8(p)
}

#[inline(always)]
unsafe fn store4(p: *mut u8, v: uint8x16_t) {
    vst1q_u8(p, v)
}

/// One pixel, in every lane.
#[inline(always)]
unsafe fn load1(p: *const u8) -> uint8x16_t {
    vreinterpretq_u8_u32(vdupq_n_u32(p.cast::<u32>().read_unaligned()))
}

/// Lane `J` of `v` to the pixel `J` after `p`.
#[inline(always)]
unsafe fn store_lane<const J: i32>(p: *mut u8, v: uint8x16_t) {
    p.add(4 * J as usize)
        .cast::<u32>()
        .write_unaligned(vgetq_lane_u32::<J>(vreinterpretq_u32_u8(v)))
}

/// Adds a prediction from the row above only, four pixels at a time: `pred(top, top_left, top_right)`.
#[inline(always)]
unsafe fn add_upper(
    out: *mut u8,
    top: *const u8,
    n: usize,
    pred: impl Fn(uint8x16_t, uint8x16_t, uint8x16_t) -> uint8x16_t,
) {
    let mut i = 0;
    while i + 4 <= n {
        let t = top.add(4 * i);
        let p = pred(load4(t), load4(t.sub(4)), load4(t.add(4)));
        store4(out.add(4 * i), vaddq_u8(load4(out.add(4 * i)), p));
        i += 4;
    }
    while i < n {
        let t = top.add(4 * i);
        let p = pred(load1(t), load1(t.sub(4)), load1(t.add(4)));
        store_lane::<0>(out.add(4 * i), vaddq_u8(load1(out.add(4 * i)), p));
        i += 1;
    }
}

/// Adds a prediction that takes the pixel decoded just before, `pred(left, top, top_left,
/// top_right)`. As libwebp's NEON, four pixels share each vector: every lane is predicted and the
/// result's lane `j` kept, then rotated into lane `j + 1` as the next pixel's left.
#[inline(always)]
unsafe fn add_serial(
    out: *mut u8,
    top: *const u8,
    n: usize,
    pred: impl Fn(uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t) -> uint8x16_t,
) {
    let mut left = load1(out.sub(4));
    let mut i = 0;
    while i + 4 <= n {
        let (o, t) = (out.add(4 * i), top.add(4 * i));
        let (src, tt, tl, tr) = (load4(o), load4(t), load4(t.sub(4)), load4(t.add(4)));
        macro_rules! lane {
            ($j:literal) => {
                let res = vaddq_u8(src, pred(left, tt, tl, tr));
                store_lane::<$j>(o, res);
                left = vextq_u8::<12>(res, res);
            };
        }
        lane!(0);
        lane!(1);
        lane!(2);
        lane!(3);
        i += 4;
    }
    while i < n {
        let (o, t) = (out.add(4 * i), top.add(4 * i));
        left = vaddq_u8(
            load1(o),
            pred(left, load1(t), load1(t.sub(4)), load1(t.add(4))),
        );
        store_lane::<0>(o, left);
        i += 1;
    }
}

/// Predictor 1, the left pixel: a prefix sum over each four pixels.
#[inline(always)]
unsafe fn add_left(out: *mut u8, n: usize) {
    let zero = vdupq_n_u8(0);
    let mut prev = load1(out.sub(4));
    let mut i = 0;
    while i + 4 <= n {
        let src = load4(out.add(4 * i));
        let sum0 = vaddq_u8(src, vextq_u8::<12>(zero, src));
        let sum1 = vaddq_u8(sum0, vextq_u8::<8>(zero, sum0));
        let res = vaddq_u8(sum1, prev);
        store4(out.add(4 * i), res);
        prev = vreinterpretq_u8_u32(vdupq_laneq_u32::<3>(vreinterpretq_u32_u8(res)));
        i += 4;
    }
    while i < n {
        prev = vaddq_u8(load1(out.add(4 * i)), prev);
        store_lane::<0>(out.add(4 * i), prev);
        i += 1;
    }
}

/// `clamp(a + b - c)` per channel.
#[inline(always)]
unsafe fn clamped_add_subtract_full(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    let half = |a: uint8x8_t, b: uint8x8_t, c: uint8x8_t| {
        let d = vreinterpretq_s16_u16(vsubl_u8(b, c));
        vqmovun_s16(vaddq_s16(vreinterpretq_s16_u16(vmovl_u8(a)), d))
    };
    let lo = half(vget_low_u8(a), vget_low_u8(b), vget_low_u8(c));
    vcombine_u8(lo, half(vget_high_u8(a), vget_high_u8(b), vget_high_u8(c)))
}

/// `clamp(avg + (avg - c) / 2)` per channel, of `avg` the average of `a` and `b`, the division
/// rounding toward zero as C's: one off `c` where it is above the average.
#[inline(always)]
unsafe fn clamped_add_subtract_half(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    let avg = vhaddq_u8(a, b);
    let c = vaddq_u8(c, vcgtq_u8(c, avg));
    let diff = vreinterpretq_s8_u8(vhsubq_u8(avg, c));
    let half = |avg: uint8x8_t, diff: int8x8_t| {
        vqmovun_s16(vaddw_s8(vreinterpretq_s16_u16(vmovl_u8(avg)), diff))
    };
    let lo = half(vget_low_u8(avg), vget_low_s8(diff));
    vcombine_u8(lo, half(vget_high_u8(avg), vget_high_s8(diff)))
}

/// Predictor 1 on `image_data[range]`, which has a pixel before it.
pub(crate) fn predict_left(image_data: &mut [u8], range: Range<usize>) {
    assert!(range.start >= 4 && range.start <= range.end && range.end <= image_data.len());
    unsafe {
        add_left(
            image_data.as_mut_ptr().add(range.start),
            (range.end - range.start) / 4,
        )
    }
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
            5 => add_serial(out, top, n, |l, t, _, tr| vhaddq_u8(vhaddq_u8(l, tr), t)),
            6 => add_serial(out, top, n, |l, _, tl, _| vhaddq_u8(l, tl)),
            7 => add_serial(out, top, n, |l, t, _, _| vhaddq_u8(l, t)),
            8 => add_upper(out, top, n, |t, tl, _| vhaddq_u8(tl, t)),
            9 => add_upper(out, top, n, |t, _, tr| vhaddq_u8(t, tr)),
            10 => add_serial(out, top, n, |l, t, tl, tr| {
                vhaddq_u8(vhaddq_u8(l, tl), vhaddq_u8(t, tr))
            }),
            11 => add_serial(out, top, n, |l, t, tl, _| {
                // The left pixel where it is further from the top-left than the top pixel is.
                let sad = |a, b| vpaddlq_u16(vpaddlq_u8(vabdq_u8(a, b)));
                let take_top = vcleq_u32(sad(l, tl), sad(t, tl));
                vbslq_u8(vreinterpretq_u8_u32(take_top), t, l)
            }),
            12 => add_serial(out, top, n, |l, t, tl, _| {
                clamped_add_subtract_full(l, t, tl)
            }),
            13 => add_serial(out, top, n, |l, t, tl, _| {
                clamped_add_subtract_half(l, t, tl)
            }),
            _ => add_upper(out, top, n, |_, _, _| {
                vreinterpretq_u8_u32(vdupq_n_u32(0xff00_0000))
            }),
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

/// libwebp's `TransformColorInverse_NEON`, with red and blue where RGBA keeps them: the deltas
/// `(t * c) >> 5` of signed bytes are `vqdmulh` of `t << 8` and `c << 2`.
#[inline(always)]
unsafe fn color_transform(
    p: *mut u8,
    n: usize,
    red_to_blue: u8,
    green_to_blue: u8,
    green_to_red: u8,
) {
    let cst = |m: u8| i32::from(i16::from(m as i8) << 2) as u32 & 0xffff;
    let mults_rb = vreinterpretq_s16_u32(vdupq_n_u32(cst(green_to_blue) << 16 | cst(green_to_red)));
    let mults_b2 = vreinterpretq_s16_u32(vdupq_n_u32(cst(red_to_blue)));
    // Green in the high byte of both 16-bit halves of its pixel.
    const G0G: [u8; 16] = [
        255, 1, 255, 1, 255, 5, 255, 5, 255, 9, 255, 9, 255, 13, 255, 13,
    ];
    let g0g = vld1q_u8(G0G.as_ptr());
    let keep_rb = vreinterpretq_u8_u32(vdupq_n_u32(0x00ff_00ff));
    let step = |input: uint8x16_t| {
        let g = vreinterpretq_s16_u8(vqtbl1q_u8(input, g0g));
        let rb = vaddq_u8(input, vreinterpretq_u8_s16(vqdmulhq_s16(g, mults_rb)));
        // The red delta of blue, from red in the high byte of the first half, to byte 2.
        let r = vshlq_n_s16::<8>(vreinterpretq_s16_u8(rb));
        let delta = vshlq_n_u32::<16>(vreinterpretq_u32_s16(vqdmulhq_s16(r, mults_b2)));
        vbslq_u8(keep_rb, vaddq_u8(rb, vreinterpretq_u8_u32(delta)), input)
    };
    let mut i = 0;
    while i + 4 <= n {
        store4(p.add(4 * i), step(load4(p.add(4 * i))));
        i += 4;
    }
    while i < n {
        store_lane::<0>(p.add(4 * i), step(load1(p.add(4 * i))));
        i += 1;
    }
}

/// Adds green to red and blue.
pub(crate) fn add_green(image_data: &mut [u8]) {
    let n = image_data.len() / 4;
    let p = image_data.as_mut_ptr();
    // Green to bytes 0 and 2 of its pixel, zero to 1 and 3.
    const G_G_: [u8; 16] = [
        1, 255, 1, 255, 5, 255, 5, 255, 9, 255, 9, 255, 13, 255, 13, 255,
    ];
    unsafe {
        let shuffle = vld1q_u8(G_G_.as_ptr());
        let step = |v: uint8x16_t| vaddq_u8(v, vqtbl1q_u8(v, shuffle));
        let mut i = 0;
        while i + 4 <= n {
            store4(p.add(4 * i), step(load4(p.add(4 * i))));
            i += 4;
        }
        while i < n {
            store_lane::<0>(p.add(4 * i), step(load1(p.add(4 * i))));
            i += 1;
        }
    }
}
