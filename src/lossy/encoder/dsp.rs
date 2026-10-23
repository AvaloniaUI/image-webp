//! Transforms, predictors, metrics and quantization of libwebp's `dsp/enc.c`, on its
//! 32-byte-stride (`BPS`) work buffers.

pub(crate) const BPS: usize = 32;

pub(crate) const YUV_SIZE: usize = BPS * 16;
pub(crate) const PRED_SIZE: usize = 32 * BPS + 16 * BPS + 8 * BPS;
pub(crate) const Y_OFF: usize = 0;
pub(crate) const U_OFF: usize = 16;
pub(crate) const V_OFF: usize = 16 + 8;

pub(crate) const I16DC16: usize = 0;
pub(crate) const I16TM16: usize = I16DC16 + 16;
pub(crate) const I16VE16: usize = 16 * BPS;
pub(crate) const I16HE16: usize = I16VE16 + 16;
pub(crate) const C8DC8: usize = 2 * 16 * BPS;
pub(crate) const C8TM8: usize = C8DC8 + 16;
pub(crate) const C8VE8: usize = 2 * 16 * BPS + 8 * BPS;
pub(crate) const C8HE8: usize = C8VE8 + 16;
pub(crate) const I4DC4: usize = 3 * 16 * BPS;
pub(crate) const I4TM4: usize = I4DC4 + 4;
pub(crate) const I4VE4: usize = I4DC4 + 8;
pub(crate) const I4HE4: usize = I4DC4 + 12;
pub(crate) const I4RD4: usize = I4DC4 + 16;
pub(crate) const I4VR4: usize = I4DC4 + 20;
pub(crate) const I4LD4: usize = I4DC4 + 24;
pub(crate) const I4VL4: usize = I4DC4 + 28;
pub(crate) const I4HD4: usize = 3 * 16 * BPS + 4 * BPS;
pub(crate) const I4HU4: usize = I4HD4 + 4;

pub(crate) const I16_MODE_OFFSETS: [usize; 4] = [I16DC16, I16TM16, I16VE16, I16HE16];
pub(crate) const UV_MODE_OFFSETS: [usize; 4] = [C8DC8, C8TM8, C8VE8, C8HE8];
pub(crate) const I4_MODE_OFFSETS: [usize; 10] = [
    I4DC4, I4TM4, I4VE4, I4HE4, I4RD4, I4VR4, I4LD4, I4VL4, I4HD4, I4HU4,
];

pub(crate) const SCAN: [usize; 16] = [
    0,
    4,
    8,
    12,
    4 * BPS,
    4 + 4 * BPS,
    8 + 4 * BPS,
    12 + 4 * BPS,
    8 * BPS,
    4 + 8 * BPS,
    8 + 8 * BPS,
    12 + 8 * BPS,
    12 * BPS,
    4 + 12 * BPS,
    8 + 12 * BPS,
    12 + 12 * BPS,
];

pub(crate) const SCAN_UV: [usize; 8] =
    [0, 4, 4 * BPS, 4 + 4 * BPS, 8, 12, 8 + 4 * BPS, 12 + 4 * BPS];

pub(crate) const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

pub(crate) const QFIX: u32 = 17;
pub(crate) const MAX_LEVEL: i32 = 2047;

#[inline]
pub(crate) const fn bias(b: u32) -> u32 {
    b << (QFIX - 8)
}

#[inline]
pub(crate) fn quantdiv(n: u32, iq: u32, b: u32) -> i32 {
    (n.wrapping_mul(iq).wrapping_add(b) >> QFIX) as i32
}

#[inline]
fn clip_8b(v: i32) -> u8 {
    if v & !0xff == 0 {
        v as u8
    } else if v < 0 {
        0
    } else {
        255
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Matrix {
    pub(crate) q: [u16; 16],
    pub(crate) iq: [u16; 16],
    pub(crate) bias: [u32; 16],
    pub(crate) zthresh: [u32; 16],
    pub(crate) sharpen: [u16; 16],
}

type V4 = [i32; 4];

#[inline(always)]
fn v4(f: impl FnMut(usize) -> i32) -> V4 {
    std::array::from_fn(f)
}

#[inline(always)]
fn transpose(m: [V4; 4]) -> [V4; 4] {
    std::array::from_fn(|i| std::array::from_fn(|j| m[j][i]))
}

/// A 4x4 block at stride `BPS`, one row per element.
#[inline(always)]
fn load4x4(p: &[u8]) -> [V4; 4] {
    let p = &p[..3 * BPS + 4];
    std::array::from_fn(|y| std::array::from_fn(|x| i32::from(p[y * BPS + x])))
}

#[inline(always)]
fn mul1(a: V4) -> V4 {
    v4(|i| ((a[i] * 20091) >> 16) + a[i])
}

#[inline(always)]
fn mul2(a: V4) -> V4 {
    v4(|i| (a[i] * 35468) >> 16)
}

#[inline(always)]
fn itransform_one(r: &[u8], input: &[i16], dst: &mut [u8]) {
    let input = &input[..16];
    let row = |k: usize| v4(|i| i32::from(input[k * 4 + i]));
    let (in0, in1, in2, in3) = (row(0), row(1), row(2), row(3));
    // Vertical pass, lane i = column i.
    let a = v4(|i| in0[i] + in2[i]);
    let b = v4(|i| in0[i] - in2[i]);
    let (m2, m1) = (mul2(in1), mul1(in3));
    let c = v4(|i| m2[i] - m1[i]);
    let (m1b, m2b) = (mul1(in1), mul2(in3));
    let d = v4(|i| m1b[i] + m2b[i]);
    let t = transpose([
        v4(|i| a[i] + d[i]),
        v4(|i| b[i] + c[i]),
        v4(|i| b[i] - c[i]),
        v4(|i| a[i] - d[i]),
    ]);
    // Horizontal pass, lane i = output row i.
    let dc = v4(|i| t[0][i] + 4);
    let a = v4(|i| dc[i] + t[2][i]);
    let b = v4(|i| dc[i] - t[2][i]);
    let (m2, m1) = (mul2(t[1]), mul1(t[3]));
    let c = v4(|i| m2[i] - m1[i]);
    let (m1b, m2b) = (mul1(t[1]), mul2(t[3]));
    let d = v4(|i| m1b[i] + m2b[i]);
    let cols = [
        v4(|i| a[i] + d[i]),
        v4(|i| b[i] + c[i]),
        v4(|i| b[i] - c[i]),
        v4(|i| a[i] - d[i]),
    ];
    let r = load4x4(r);
    let dst = &mut dst[..3 * BPS + 4];
    for y in 0..4 {
        for x in 0..4 {
            dst[y * BPS + x] = clip_8b(r[y][x] + (cols[x][y] >> 3));
        }
    }
}

/// Inverse transform of `input` added to `r`, into `dst`; a second block 4 pixels to the
/// right when `two`.
#[inline]
pub(crate) fn itransform(r: &[u8], input: &[i16], dst: &mut [u8], two: bool) {
    #[cfg(target_arch = "x86_64")]
    {
        if two {
            super::sse2::itransform_two(r, input, dst)
        } else {
            super::sse2::itransform_one(r, input, dst)
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if two {
            super::neon::itransform_one(r, input, dst);
            super::neon::itransform_one(&r[4..], &input[16..], &mut dst[4..]);
        } else {
            super::neon::itransform_one(r, input, dst);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        itransform_portable(r, input, dst, two)
    }
}

/// `itransform` of one block into `dst`, returning `sse4x4(src, dst)` and, with `w`,
/// `tdisto4x4(src, dst, w)` (else 0).
#[inline]
pub(crate) fn itransform_disto(
    r: &[u8],
    input: &[i16],
    dst: &mut [u8],
    src: &[u8],
    w: Option<&[u16; 16]>,
) -> (i32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let (sse, t) = super::sse2::itransform_one_disto(r, input, dst, src, w);
        (sse, t.abs() >> 5)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        itransform(r, input, dst, false);
        (sse4x4(src, dst), w.map_or(0, |w| tdisto4x4(src, dst, w)))
    }
}

/// The forward transform of `src - r` quantized with `mtx` into `levels`, then
/// `itransform_disto` of the result. Returns `non_zero_mask(levels)` and the metrics.
#[inline]
pub(crate) fn reconstruct4x4_disto(
    src: &[u8],
    r: &[u8],
    mtx: &Matrix,
    levels: &mut [i16; 16],
    dst: &mut [u8],
    w: Option<&[u16; 16]>,
) -> (u32, i32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        let (nz, sse, t) = super::sse2::reconstruct4x4_disto(src, r, mtx, levels, dst, w);
        (nz, sse, t.abs() >> 5)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut tmp = [0i16; 16];
        ftransform(src, r, &mut tmp);
        quantize_block(&mut tmp, levels, mtx);
        let (sse, t) = itransform_disto(r, &tmp, dst, src, w);
        (non_zero_mask(levels), sse, t)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn itransform_portable(r: &[u8], input: &[i16], dst: &mut [u8], two: bool) {
    itransform_one(r, &input[..16], dst);
    if two {
        itransform_one(&r[4..], &input[16..32], &mut dst[4..]);
    }
}

#[inline]
pub(crate) fn ftransform(src: &[u8], r: &[u8], out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::ftransform(src, r, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::ftransform(src, r, out)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        ftransform_portable(src, r, out)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn ftransform_portable(src: &[u8], r: &[u8], out: &mut [i16]) {
    let (s, p) = (load4x4(src), load4x4(r));
    // First pass, lane i = row i.
    let d: [V4; 4] = std::array::from_fn(|x| v4(|i| s[i][x] - p[i][x]));
    let a0 = v4(|i| d[0][i] + d[3][i]);
    let a1 = v4(|i| d[1][i] + d[2][i]);
    let a2 = v4(|i| d[1][i] - d[2][i]);
    let a3 = v4(|i| d[0][i] - d[3][i]);
    let t = transpose([
        v4(|i| (a0[i] + a1[i]) * 8),
        v4(|i| (a2[i] * 2217 + a3[i] * 5352 + 1812) >> 9),
        v4(|i| (a0[i] - a1[i]) * 8),
        v4(|i| (a3[i] * 2217 - a2[i] * 5352 + 937) >> 9),
    ]);
    // Second pass, lane i = column i.
    let a0 = v4(|i| t[0][i] + t[3][i]);
    let a1 = v4(|i| t[1][i] + t[2][i]);
    let a2 = v4(|i| t[1][i] - t[2][i]);
    let a3 = v4(|i| t[0][i] - t[3][i]);
    let out = &mut out[..16];
    for i in 0..4 {
        out[i] = ((a0[i] + a1[i] + 7) >> 4) as i16;
        out[4 + i] = (((a2[i] * 2217 + a3[i] * 5352 + 12000) >> 16) + i32::from(a3[i] != 0)) as i16;
        out[8 + i] = ((a0[i] - a1[i] + 7) >> 4) as i16;
        out[12 + i] = ((a3[i] * 2217 - a2[i] * 5352 + 51000) >> 16) as i16;
    }
}

#[inline]
pub(crate) fn ftransform2(src: &[u8], r: &[u8], out: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::ftransform2(src, r, out)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::ftransform(src, r, &mut out[..16]);
        super::neon::ftransform(&src[4..], &r[4..], &mut out[16..32]);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        ftransform2_portable(src, r, out)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn ftransform2_portable(src: &[u8], r: &[u8], out: &mut [i16]) {
    ftransform_portable(src, r, &mut out[..16]);
    ftransform_portable(&src[4..], &r[4..], &mut out[16..32]);
}

/// Walsh-Hadamard transform of the DC coefficients of 16 blocks, `input` holding them 16 apart.
pub(crate) fn ftransform_wht(input: &[i16], out: &mut [i16; 16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let p = &input[i * 64..];
        let a0 = i32::from(p[0]) + i32::from(p[2 * 16]);
        let a1 = i32::from(p[16]) + i32::from(p[3 * 16]);
        let a2 = i32::from(p[16]) - i32::from(p[3 * 16]);
        let a3 = i32::from(p[0]) - i32::from(p[2 * 16]);
        tmp[i * 4] = a0 + a1;
        tmp[1 + i * 4] = a3 + a2;
        tmp[2 + i * 4] = a3 - a2;
        tmp[3 + i * 4] = a0 - a1;
    }
    for i in 0..4 {
        let a0 = tmp[i] + tmp[8 + i];
        let a1 = tmp[4 + i] + tmp[12 + i];
        let a2 = tmp[4 + i] - tmp[12 + i];
        let a3 = tmp[i] - tmp[8 + i];
        out[i] = ((a0 + a1) >> 1) as i16;
        out[4 + i] = ((a3 + a2) >> 1) as i16;
        out[8 + i] = ((a3 - a2) >> 1) as i16;
        out[12 + i] = ((a0 - a1) >> 1) as i16;
    }
}

/// Inverse WHT of the decoder, writing the DC of each of the 16 blocks in `out` (16 apart).
pub(crate) fn transform_wht(input: &[i16; 16], out: &mut [i16]) {
    let mut tmp = [0i32; 16];
    for i in 0..4 {
        let a0 = i32::from(input[i]) + i32::from(input[12 + i]);
        let a1 = i32::from(input[4 + i]) + i32::from(input[8 + i]);
        let a2 = i32::from(input[4 + i]) - i32::from(input[8 + i]);
        let a3 = i32::from(input[i]) - i32::from(input[12 + i]);
        tmp[i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for i in 0..4 {
        let dc = tmp[i * 4] + 3;
        let a0 = dc + tmp[3 + i * 4];
        let a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        let a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        let a3 = dc - tmp[3 + i * 4];
        let o = &mut out[i * 64..];
        o[0] = ((a0 + a1) >> 3) as i16;
        o[16] = ((a3 + a2) >> 3) as i16;
        o[32] = ((a0 - a1) >> 3) as i16;
        o[48] = ((a3 - a2) >> 3) as i16;
    }
}

#[inline(always)]
fn rows<const N: usize>(dst: &mut [u8], mut row: impl FnMut(usize) -> [u8; N]) {
    let dst = &mut dst[..(N - 1) * BPS + N];
    for j in 0..N {
        let r: &mut [u8; N] = (&mut dst[j * BPS..j * BPS + N]).try_into().unwrap();
        *r = row(j);
    }
}

#[inline(always)]
fn fill<const N: usize>(dst: &mut [u8], value: u8) {
    rows::<N>(dst, |_| [value; N]);
}

#[inline(always)]
fn vertical_pred<const N: usize>(dst: &mut [u8], top: Option<&[u8]>) {
    match top {
        Some(top) => {
            let top: [u8; N] = top[..N].try_into().unwrap();
            rows::<N>(dst, |_| top);
        }
        None => fill::<N>(dst, 127),
    }
}

#[inline(always)]
fn horizontal_pred<const N: usize>(dst: &mut [u8], left: Option<&[u8]>) {
    match left {
        // `left` starts at the top-left sample.
        Some(left) => {
            let left = &left[1..=N];
            rows::<N>(dst, |j| [left[j]; N]);
        }
        None => fill::<N>(dst, 129),
    }
}

#[inline(always)]
fn true_motion<const N: usize>(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    match (left, top) {
        (Some(left), Some(top)) => {
            let top: [u8; N] = top[..N].try_into().unwrap();
            let top = top.map(i16::from);
            let left = &left[..=N];
            let tl = i16::from(left[0]);
            rows::<N>(dst, |y| {
                let l = i16::from(left[1 + y]) - tl;
                std::array::from_fn(|x| (top[x] + l).clamp(0, 255) as u8)
            });
        }
        (Some(left), None) => horizontal_pred::<N>(dst, Some(left)),
        (None, Some(top)) => vertical_pred::<N>(dst, Some(top)),
        (None, None) => fill::<N>(dst, 129),
    }
}

#[inline(always)]
fn dc_mode<const N: usize>(
    dst: &mut [u8],
    left: Option<&[u8]>,
    top: Option<&[u8]>,
    round: u32,
    shift: u32,
) {
    let sum = |s: &[u8]| s[..N].iter().map(|&v| u32::from(v)).sum::<u32>();
    let dc = match (top, left) {
        (Some(top), Some(left)) => (sum(top) + sum(&left[1..]) + round) >> shift,
        (Some(top), None) => (2 * sum(top) + round) >> shift,
        (None, Some(left)) => (2 * sum(&left[1..]) + round) >> shift,
        (None, None) => 0x80,
    };
    fill::<N>(dst, dc as u8);
}

/// The four chroma predictions. `left` points at the U top-left sample, with V's 16 bytes
/// further; `top` holds 8 U then 8 V samples.
pub(crate) fn pred_chroma8(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    for (k, off) in [(0usize, 0usize), (1, 8)] {
        let l = left.map(|l| &l[k * 16..]);
        let t = top.map(|t| &t[k * 8..]);
        dc_mode::<8>(&mut dst[C8DC8 + off..], l, t, 8, 4);
        vertical_pred::<8>(&mut dst[C8VE8 + off..], t);
        horizontal_pred::<8>(&mut dst[C8HE8 + off..], l);
        true_motion::<8>(&mut dst[C8TM8 + off..], l, t);
    }
}

/// The four 16x16 luma predictions. `left` starts at the top-left sample.
pub(crate) fn pred_luma16(dst: &mut [u8], left: Option<&[u8]>, top: Option<&[u8]>) {
    dc_mode::<16>(&mut dst[I16DC16..], left, top, 16, 5);
    vertical_pred::<16>(&mut dst[I16VE16..], top);
    horizontal_pred::<16>(&mut dst[I16HE16..], left);
    true_motion::<16>(&mut dst[I16TM16..], left, top);
}

#[inline]
fn avg3(a: i32, b: i32, c: i32) -> u8 {
    ((a + 2 * b + c + 2) >> 2) as u8
}

#[inline]
fn avg2(a: i32, b: i32) -> u8 {
    ((a + b + 1) >> 1) as u8
}

/// The ten 4x4 predictions. `b[t]` is the first top sample: left samples are `b[t-5..t-2]`
/// (bottom to top), the top-left is `b[t-1]` and top/top-right samples `b[t..t+8]`.
pub(crate) fn pred_luma4(dst: &mut [u8], b: &[u8], t: usize) {
    #[cfg(target_arch = "aarch64")]
    super::neon::pred_luma4(dst, b, t);
    #[cfg(not(target_arch = "aarch64"))]
    pred_luma4_portable(dst, b, t);
}

#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn pred_luma4_portable(dst: &mut [u8], b: &[u8], t: usize) {
    let edge: &[u8; 13] = b[t - 5..t + 8].try_into().unwrap();
    let at = |i: isize| i32::from(edge[(5 + i) as usize]);
    let dst: &mut [u8; 8 * BPS] = (&mut dst[I4DC4..I4DC4 + 8 * BPS]).try_into().unwrap();
    let set = |dst: &mut [u8; 8 * BPS], off: usize, xx: usize, yy: usize, v: u8| {
        dst[off - I4DC4 + xx + yy * BPS] = v
    };
    let x = at(-1);
    let (i_, j, k, l) = (at(-2), at(-3), at(-4), at(-5));
    let (a, bb, c, d) = (at(0), at(1), at(2), at(3));
    let (e, f, g, h) = (at(4), at(5), at(6), at(7));

    // DC
    let mut dc = 4u32;
    for n in 0..4 {
        dc += (at(n) + at(-5 + n)) as u32;
    }
    fill::<4>(&mut dst[..], (dc >> 3) as u8);

    // TM
    for yy in 0..4 {
        let left = at(-2 - yy as isize);
        for xx in 0..4 {
            set(dst, I4TM4, xx, yy, clip_8b(left + at(xx as isize) - x));
        }
    }

    // VE
    let ve = [
        avg3(x, a, bb),
        avg3(a, bb, c),
        avg3(bb, c, d),
        avg3(c, d, e),
    ];
    for yy in 0..4 {
        let o = I4VE4 - I4DC4 + yy * BPS;
        dst[o..o + 4].copy_from_slice(&ve);
    }

    // HE
    let he = [avg3(x, i_, j), avg3(i_, j, k), avg3(j, k, l), avg3(k, l, l)];
    for yy in 0..4 {
        let o = I4HE4 - I4DC4 + yy * BPS;
        dst[o..o + 4].fill(he[yy]);
    }

    // RD
    let o = I4RD4;
    set(dst, o, 0, 3, avg3(j, k, l));
    let v = avg3(i_, j, k);
    set(dst, o, 0, 2, v);
    set(dst, o, 1, 3, v);
    let v = avg3(x, i_, j);
    set(dst, o, 0, 1, v);
    set(dst, o, 1, 2, v);
    set(dst, o, 2, 3, v);
    let v = avg3(a, x, i_);
    set(dst, o, 0, 0, v);
    set(dst, o, 1, 1, v);
    set(dst, o, 2, 2, v);
    set(dst, o, 3, 3, v);
    let v = avg3(bb, a, x);
    set(dst, o, 1, 0, v);
    set(dst, o, 2, 1, v);
    set(dst, o, 3, 2, v);
    let v = avg3(c, bb, a);
    set(dst, o, 2, 0, v);
    set(dst, o, 3, 1, v);
    set(dst, o, 3, 0, avg3(d, c, bb));

    // VR
    let o = I4VR4;
    let v = avg2(x, a);
    set(dst, o, 0, 0, v);
    set(dst, o, 1, 2, v);
    let v = avg2(a, bb);
    set(dst, o, 1, 0, v);
    set(dst, o, 2, 2, v);
    let v = avg2(bb, c);
    set(dst, o, 2, 0, v);
    set(dst, o, 3, 2, v);
    set(dst, o, 3, 0, avg2(c, d));
    set(dst, o, 0, 3, avg3(k, j, i_));
    set(dst, o, 0, 2, avg3(j, i_, x));
    let v = avg3(i_, x, a);
    set(dst, o, 0, 1, v);
    set(dst, o, 1, 3, v);
    let v = avg3(x, a, bb);
    set(dst, o, 1, 1, v);
    set(dst, o, 2, 3, v);
    let v = avg3(a, bb, c);
    set(dst, o, 2, 1, v);
    set(dst, o, 3, 3, v);
    set(dst, o, 3, 1, avg3(bb, c, d));

    // LD
    let o = I4LD4;
    set(dst, o, 0, 0, avg3(a, bb, c));
    let v = avg3(bb, c, d);
    set(dst, o, 1, 0, v);
    set(dst, o, 0, 1, v);
    let v = avg3(c, d, e);
    set(dst, o, 2, 0, v);
    set(dst, o, 1, 1, v);
    set(dst, o, 0, 2, v);
    let v = avg3(d, e, f);
    set(dst, o, 3, 0, v);
    set(dst, o, 2, 1, v);
    set(dst, o, 1, 2, v);
    set(dst, o, 0, 3, v);
    let v = avg3(e, f, g);
    set(dst, o, 3, 1, v);
    set(dst, o, 2, 2, v);
    set(dst, o, 1, 3, v);
    let v = avg3(f, g, h);
    set(dst, o, 3, 2, v);
    set(dst, o, 2, 3, v);
    set(dst, o, 3, 3, avg3(g, h, h));

    // VL
    let o = I4VL4;
    set(dst, o, 0, 0, avg2(a, bb));
    let v = avg2(bb, c);
    set(dst, o, 1, 0, v);
    set(dst, o, 0, 2, v);
    let v = avg2(c, d);
    set(dst, o, 2, 0, v);
    set(dst, o, 1, 2, v);
    let v = avg2(d, e);
    set(dst, o, 3, 0, v);
    set(dst, o, 2, 2, v);
    set(dst, o, 0, 1, avg3(a, bb, c));
    let v = avg3(bb, c, d);
    set(dst, o, 1, 1, v);
    set(dst, o, 0, 3, v);
    let v = avg3(c, d, e);
    set(dst, o, 2, 1, v);
    set(dst, o, 1, 3, v);
    let v = avg3(d, e, f);
    set(dst, o, 3, 1, v);
    set(dst, o, 2, 3, v);
    set(dst, o, 3, 2, avg3(e, f, g));
    set(dst, o, 3, 3, avg3(f, g, h));

    // HD
    let o = I4HD4;
    let v = avg2(i_, x);
    set(dst, o, 0, 0, v);
    set(dst, o, 2, 1, v);
    let v = avg2(j, i_);
    set(dst, o, 0, 1, v);
    set(dst, o, 2, 2, v);
    let v = avg2(k, j);
    set(dst, o, 0, 2, v);
    set(dst, o, 2, 3, v);
    set(dst, o, 0, 3, avg2(l, k));
    set(dst, o, 3, 0, avg3(a, bb, c));
    set(dst, o, 2, 0, avg3(x, a, bb));
    let v = avg3(i_, x, a);
    set(dst, o, 1, 0, v);
    set(dst, o, 3, 1, v);
    let v = avg3(j, i_, x);
    set(dst, o, 1, 1, v);
    set(dst, o, 3, 2, v);
    let v = avg3(k, j, i_);
    set(dst, o, 1, 2, v);
    set(dst, o, 3, 3, v);
    set(dst, o, 1, 3, avg3(l, k, j));

    // HU
    let o = I4HU4;
    set(dst, o, 0, 0, avg2(i_, j));
    let v = avg2(j, k);
    set(dst, o, 2, 0, v);
    set(dst, o, 0, 1, v);
    let v = avg2(k, l);
    set(dst, o, 2, 1, v);
    set(dst, o, 0, 2, v);
    set(dst, o, 1, 0, avg3(i_, j, k));
    let v = avg3(j, k, l);
    set(dst, o, 3, 0, v);
    set(dst, o, 1, 1, v);
    let v = avg3(k, l, l);
    set(dst, o, 3, 1, v);
    set(dst, o, 1, 2, v);
    let l8 = l as u8;
    for (xx, yy) in [(3, 2), (2, 2), (0, 3), (1, 3), (2, 3), (3, 3)] {
        set(dst, o, xx, yy, l8);
    }
}

#[inline(always)]
fn sse<const W: usize>(a: &[u8], b: &[u8], h: usize) -> i32 {
    let mut count = 0;
    for y in 0..h {
        let ra: &[u8; W] = a[y * BPS..y * BPS + W].try_into().unwrap();
        let rb: &[u8; W] = b[y * BPS..y * BPS + W].try_into().unwrap();
        for x in 0..W {
            let diff = i32::from(ra[x]) - i32::from(rb[x]);
            count += diff * diff;
        }
    }
    count
}

#[inline]
pub(crate) fn sse16x16(a: &[u8], b: &[u8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::sse16xn(a, b, 16)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::sse16xn(a, b, 16)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        sse16x16_portable(a, b)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn sse16x16_portable(a: &[u8], b: &[u8]) -> i32 {
    sse::<16>(a, b, 16)
}

#[inline]
pub(crate) fn sse16x8(a: &[u8], b: &[u8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::sse16xn(a, b, 8)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::sse16xn(a, b, 8)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        sse16x8_portable(a, b)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn sse16x8_portable(a: &[u8], b: &[u8]) -> i32 {
    sse::<16>(a, b, 8)
}

#[inline]
pub(crate) fn sse4x4(a: &[u8], b: &[u8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::sse4x4(a, b)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::sse4x4(a, b)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        sse4x4_portable(a, b)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn sse4x4_portable(a: &[u8], b: &[u8]) -> i32 {
    sse::<4>(a, b, 4)
}

pub(crate) fn mean16x4(r: &[u8], dc: &mut [u32]) {
    for k in 0..4 {
        let mut avg = 0u32;
        for y in 0..4 {
            for x in 0..4 {
                avg += u32::from(r[k * 4 + x + y * BPS]);
            }
        }
        dc[k] = avg;
    }
}

/// Weighted sum of the absolute Hadamard coefficients of `a` and of `b`.
#[inline(always)]
fn ttransform2(a: &[u8], b: &[u8], w: &[u16; 16]) -> (i32, i32) {
    let one = |input: [V4; 4]| {
        // Horizontal pass, lane i = row i.
        let col = |x: usize| v4(|i| input[i][x]);
        let (c0, c1, c2, c3) = (col(0), col(1), col(2), col(3));
        let a0 = v4(|i| c0[i] + c2[i]);
        let a1 = v4(|i| c1[i] + c3[i]);
        let a2 = v4(|i| c1[i] - c3[i]);
        let a3 = v4(|i| c0[i] - c2[i]);
        let t = transpose([
            v4(|i| a0[i] + a1[i]),
            v4(|i| a3[i] + a2[i]),
            v4(|i| a3[i] - a2[i]),
            v4(|i| a0[i] - a1[i]),
        ]);
        // Vertical pass, lane i = column i.
        let a0 = v4(|i| t[0][i] + t[2][i]);
        let a1 = v4(|i| t[1][i] + t[3][i]);
        let a2 = v4(|i| t[1][i] - t[3][i]);
        let a3 = v4(|i| t[0][i] - t[2][i]);
        let mut sum = 0;
        for i in 0..4 {
            sum += i32::from(w[i]) * (a0[i] + a1[i]).abs()
                + i32::from(w[4 + i]) * (a3[i] + a2[i]).abs()
                + i32::from(w[8 + i]) * (a3[i] - a2[i]).abs()
                + i32::from(w[12 + i]) * (a0[i] - a1[i]).abs();
        }
        sum
    };
    (one(load4x4(a)), one(load4x4(b)))
}

#[inline]
pub(crate) fn tdisto4x4(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::ttransform(a, b, w).abs() >> 5
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::tdisto4x4(a, b, w)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        tdisto4x4_portable(a, b, w)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn tdisto4x4_portable(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let (sa, sb) = ttransform2(a, b, w);
    (sb - sa).abs() >> 5
}

pub(crate) fn tdisto16x16(a: &[u8], b: &[u8], w: &[u16; 16]) -> i32 {
    let mut d = 0;
    for y in (0..16 * BPS).step_by(4 * BPS) {
        for x in (0..16).step_by(4) {
            d += tdisto4x4(&a[x + y..], &b[x + y..], w);
        }
    }
    d
}

pub(crate) const MAX_COEFF_THRESH: usize = 31;

/// `max_value` and `last_non_zero` of the DCT-coefficient histogram of blocks
/// `start..end`, numbered as `SCAN` then `SCAN_UV`.
pub(crate) fn collect_histogram(r: &[u8], pred: &[u8], start: usize, end: usize) -> (i32, i32) {
    // Four interleaved histograms, so repeated bins do not serialize on one counter.
    let mut distribution = [[0u16; MAX_COEFF_THRESH + 1]; 4];
    for j in (start..end).step_by(2) {
        let off = if j < 16 { SCAN[j] } else { SCAN_UV[j - 16] };
        let mut out = [0i16; 32];
        ftransform2(&r[off..], &pred[off..], &mut out);
        let bins = out.map(|v| (v.unsigned_abs() >> 3).min(MAX_COEFF_THRESH as u16));
        for b in bins.chunks_exact(4) {
            for (h, &b) in distribution.iter_mut().zip(b) {
                h[usize::from(b)] += 1;
            }
        }
    }
    let distribution: [u16; MAX_COEFF_THRESH + 1] = std::array::from_fn(|k| {
        distribution[0][k] + distribution[1][k] + distribution[2][k] + distribution[3][k]
    });
    let max_value = distribution.iter().copied().max().unwrap_or(0);
    let last_non_zero = distribution.iter().rposition(|&v| v > 0).map_or(1, |k| k);
    (i32::from(max_value), last_non_zero as i32)
}

/// Quantizes `input` in place to its dequantized value, writing levels in zigzag order to
/// `out`. Returns whether any level is non-zero.
#[inline]
pub(crate) fn quantize_block(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        super::sse2::quantize_block(input, out, mtx)
    }
    #[cfg(target_arch = "aarch64")]
    {
        super::neon::quantize_block(input, out, mtx)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        quantize_block_portable(input, out, mtx)
    }
}

#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
fn quantize_block_portable(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
    let input: &mut [i16; 16] = (&mut input[..16]).try_into().unwrap();
    // QUANTDIV is zero exactly when the coefficient is not above `zthresh`, so the
    // comparison libwebp's C code makes is only a shortcut.
    let mut levels = [0i32; 16];
    for j in 0..16 {
        let v = i32::from(input[j]);
        let coeff = v.unsigned_abs() + u32::from(mtx.sharpen[j]);
        let level =
            ((coeff * u32::from(mtx.iq[j]) + mtx.bias[j]) >> QFIX).min(MAX_LEVEL as u32) as i32;
        let level = if v < 0 { -level } else { level };
        levels[j] = level;
        input[j] = (level * i32::from(mtx.q[j])) as i16;
    }
    let out = &mut out[..16];
    for n in 0..16 {
        out[n] = levels[ZIGZAG[n]] as i16;
    }
    levels.iter().any(|&l| l != 0)
}

/// Bit `n` set when `levels[n]` is not zero.
pub(crate) fn non_zero_mask(levels: &[i16; 16]) -> u32 {
    (levels.iter().enumerate()).fold(0, |m, (n, &l)| m | u32::from(l != 0) << n)
}

pub(crate) fn quantize2_blocks(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> u32 {
    let nz0 = quantize_block(&mut input[..16], &mut out[..16], mtx);
    let nz1 = quantize_block(&mut input[16..32], &mut out[16..32], mtx);
    u32::from(nz0) | (u32::from(nz1) << 1)
}

#[inline]
pub(crate) fn is_flat(levels: &[i16], num_blocks: usize, thresh: i32) -> bool {
    let mut score = 0i32;
    for block in levels[..num_blocks * 16].chunks_exact(16) {
        let block: &[i16; 16] = block.try_into().unwrap();
        let nz = block.iter().fold(0i16, |n, &l| n + i16::from(l != 0));
        score += i32::from(nz - i16::from(block[0] != 0));
    }
    score <= thresh
}

pub(crate) fn is_flat_source16(src: &[u8]) -> bool {
    let v = src[0];
    (0..16).all(|y| src[y * BPS..y * BPS + 16].iter().all(|&p| p == v))
}

pub(crate) fn copy(src: &[u8], dst: &mut [u8], w: usize, h: usize) {
    for y in 0..h {
        dst[y * BPS..y * BPS + w].copy_from_slice(&src[y * BPS..y * BPS + w]);
    }
}

#[cfg(test)]
mod reference {
    //! The scalar kernels as libwebp's C code writes them, to check the vectorized ones.
    use super::*;

    fn mul1(a: i32) -> i32 {
        ((a * 20091) >> 16) + a
    }

    fn mul2(a: i32) -> i32 {
        (a * 35468) >> 16
    }

    fn itransform_one(r: &[u8], input: &[i16], dst: &mut [u8]) {
        let mut c = [0i32; 16];
        for i in 0..4 {
            let in0 = i32::from(input[i]);
            let in4 = i32::from(input[4 + i]);
            let in8 = i32::from(input[8 + i]);
            let in12 = i32::from(input[12 + i]);
            let a = in0 + in8;
            let b = in0 - in8;
            let cc = mul2(in4) - mul1(in12);
            let d = mul1(in4) + mul2(in12);
            c[i * 4] = a + d;
            c[i * 4 + 1] = b + cc;
            c[i * 4 + 2] = b - cc;
            c[i * 4 + 3] = a - d;
        }
        for i in 0..4 {
            let dc = c[i] + 4;
            let a = dc + c[8 + i];
            let b = dc - c[8 + i];
            let cc = mul2(c[4 + i]) - mul1(c[12 + i]);
            let d = mul1(c[4 + i]) + mul2(c[12 + i]);
            let row = i * BPS;
            dst[row] = clip_8b(i32::from(r[row]) + ((a + d) >> 3));
            dst[row + 1] = clip_8b(i32::from(r[row + 1]) + ((b + cc) >> 3));
            dst[row + 2] = clip_8b(i32::from(r[row + 2]) + ((b - cc) >> 3));
            dst[row + 3] = clip_8b(i32::from(r[row + 3]) + ((a - d) >> 3));
        }
    }

    pub(crate) fn ftransform(src: &[u8], r: &[u8], out: &mut [i16]) {
        let mut tmp = [0i32; 16];
        for i in 0..4 {
            let s = &src[i * BPS..];
            let p = &r[i * BPS..];
            let d0 = i32::from(s[0]) - i32::from(p[0]);
            let d1 = i32::from(s[1]) - i32::from(p[1]);
            let d2 = i32::from(s[2]) - i32::from(p[2]);
            let d3 = i32::from(s[3]) - i32::from(p[3]);
            let a0 = d0 + d3;
            let a1 = d1 + d2;
            let a2 = d1 - d2;
            let a3 = d0 - d3;
            tmp[i * 4] = (a0 + a1) * 8;
            tmp[1 + i * 4] = (a2 * 2217 + a3 * 5352 + 1812) >> 9;
            tmp[2 + i * 4] = (a0 - a1) * 8;
            tmp[3 + i * 4] = (a3 * 2217 - a2 * 5352 + 937) >> 9;
        }
        for i in 0..4 {
            let a0 = tmp[i] + tmp[12 + i];
            let a1 = tmp[4 + i] + tmp[8 + i];
            let a2 = tmp[4 + i] - tmp[8 + i];
            let a3 = tmp[i] - tmp[12 + i];
            out[i] = ((a0 + a1 + 7) >> 4) as i16;
            out[4 + i] = (((a2 * 2217 + a3 * 5352 + 12000) >> 16) + i32::from(a3 != 0)) as i16;
            out[8 + i] = ((a0 - a1 + 7) >> 4) as i16;
            out[12 + i] = ((a3 * 2217 - a2 * 5352 + 51000) >> 16) as i16;
        }
    }

    fn sse(a: &[u8], b: &[u8], w: usize, h: usize) -> i32 {
        let mut count = 0;
        for y in 0..h {
            for x in 0..w {
                let diff = i32::from(a[y * BPS + x]) - i32::from(b[y * BPS + x]);
                count += diff * diff;
            }
        }
        count
    }

    fn ttransform(input: &[u8], w: &[u16; 16]) -> i32 {
        let mut sum = 0;
        let mut tmp = [0i32; 16];
        for i in 0..4 {
            let p = &input[i * BPS..];
            let a0 = i32::from(p[0]) + i32::from(p[2]);
            let a1 = i32::from(p[1]) + i32::from(p[3]);
            let a2 = i32::from(p[1]) - i32::from(p[3]);
            let a3 = i32::from(p[0]) - i32::from(p[2]);
            tmp[i * 4] = a0 + a1;
            tmp[1 + i * 4] = a3 + a2;
            tmp[2 + i * 4] = a3 - a2;
            tmp[3 + i * 4] = a0 - a1;
        }
        for i in 0..4 {
            let a0 = tmp[i] + tmp[8 + i];
            let a1 = tmp[4 + i] + tmp[12 + i];
            let a2 = tmp[4 + i] - tmp[12 + i];
            let a3 = tmp[i] - tmp[8 + i];
            sum += i32::from(w[i]) * (a0 + a1).abs();
            sum += i32::from(w[4 + i]) * (a3 + a2).abs();
            sum += i32::from(w[8 + i]) * (a3 - a2).abs();
            sum += i32::from(w[12 + i]) * (a0 - a1).abs();
        }
        sum
    }

    /// Quantizes `input` in place to its dequantized value, writing levels in zigzag order to
    /// `out`. Returns whether any level is non-zero.
    pub(crate) fn quantize_block(input: &mut [i16], out: &mut [i16], mtx: &Matrix) -> bool {
        let mut last = -1;
        for n in 0..16 {
            let j = ZIGZAG[n];
            let sign = input[j] < 0;
            let coeff = (i32::from(input[j]).unsigned_abs()) + u32::from(mtx.sharpen[j]);
            if coeff > mtx.zthresh[j] {
                let q = i32::from(mtx.q[j]);
                let mut level = quantdiv(coeff, u32::from(mtx.iq[j]), mtx.bias[j]).min(MAX_LEVEL);
                if sign {
                    level = -level;
                }
                input[j] = (level * q) as i16;
                out[n] = level as i16;
                if level != 0 {
                    last = n as i32;
                }
            } else {
                out[n] = 0;
                input[j] = 0;
            }
        }
        last >= 0
    }

    #[test]
    fn pred_luma4_matches_portable() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(4);
        for round in 0..4000 {
            let mut b = [0u8; 40];
            for v in &mut b {
                *v = if round % 3 == 0 {
                    rng.gen_range(0..2) * 255
                } else {
                    rng.gen()
                };
            }
            for t in [5, 9, 13, 17, 21, 25, 29] {
                let (mut d1, mut d2) = ([0u8; PRED_SIZE], [0u8; PRED_SIZE]);
                super::pred_luma4(&mut d1, &b, t);
                super::pred_luma4_portable(&mut d2, &b, t);
                assert_eq!(d1, d2, "top at {t}");
            }
        }
    }

    #[test]
    fn kernels_match_reference() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for round in 0..20000 {
            let mut a = [0u8; 8 * BPS];
            let mut b = [0u8; 8 * BPS];
            let spread = [255u8, 16, 3][round % 3];
            let base: u8 = rng.gen();
            for i in 0..8 * BPS {
                a[i] = base.saturating_add(rng.gen_range(0..=spread));
                b[i] = base.saturating_sub(rng.gen_range(0..=spread));
            }
            let (mut f1, mut f2) = ([0i16; 16], [0i16; 16]);
            super::ftransform(&a, &b, &mut f1);
            ftransform(&a, &b, &mut f2);
            assert_eq!(f1, f2);
            super::ftransform_portable(&a, &b, &mut f1);
            assert_eq!(f1, f2);
            let (mut g1, mut g2) = ([0i16; 32], [0i16; 32]);
            super::ftransform2(&a, &b, &mut g1);
            ftransform(&a, &b, &mut g2[..16]);
            ftransform(&a[4..], &b[4..], &mut g2[16..]);
            assert_eq!(g1, g2);

            assert_eq!(super::sse4x4(&a, &b), sse(&a, &b, 4, 4));
            assert_eq!(super::sse4x4_portable(&a, &b), sse(&a, &b, 4, 4));
            let w = [38u16, 32, 20, 9, 32, 28, 17, 7, 20, 17, 10, 4, 9, 7, 4, 2];
            let disto = (ttransform(&b, &w) - ttransform(&a, &w)).abs() >> 5;
            assert_eq!(super::tdisto4x4(&a, &b, &w), disto);
            assert_eq!(super::tdisto4x4_portable(&a, &b, &w), disto);

            let mut coeffs = [0i16; 16];
            for c in &mut coeffs {
                *c = rng.gen_range(-2048..=2048) >> [0, 3, 6][round % 3];
            }
            let mut mtx = Matrix::default();
            let q0: u16 = rng.gen_range(4..=314);
            let q1: u16 = rng.gen_range(4..=284);
            mtx.q = [q1; 16];
            mtx.q[0] = q0;
            for j in 0..16 {
                mtx.iq[j] = ((1 << QFIX) / u32::from(mtx.q[j])) as u16;
                mtx.bias[j] = bias([96, 110, 115][round % 3]);
                mtx.zthresh[j] = ((1 << QFIX) - 1 - mtx.bias[j]) / u32::from(mtx.iq[j]);
                mtx.sharpen[j] = if round % 2 == 0 {
                    (j as u16 * 6 * mtx.q[j]) >> 11
                } else {
                    0
                };
            }
            let (mut in1, mut in2, mut in3) = (coeffs, coeffs, coeffs);
            let (mut o1, mut o2, mut o3) = ([0i16; 16], [0i16; 16], [0i16; 16]);
            assert_eq!(
                super::quantize_block(&mut in1, &mut o1, &mtx),
                quantize_block(&mut in2, &mut o2, &mtx)
            );
            assert_eq!((in1, o1), (in2, o2));
            super::quantize_block_portable(&mut in3, &mut o3, &mtx);
            assert_eq!((in3, o3), (in2, o2));

            let (mut d1, mut d2) = ([0u8; 4 * BPS], [0u8; 4 * BPS]);
            super::itransform(&b, &in1, &mut d1, false);
            itransform_one(&b, &in1, &mut d2);
            assert_eq!(d1, d2);
            super::itransform_portable(&b, &in1, &mut d1, false);
            assert_eq!(d1, d2);
            let mut d3 = [0u8; 4 * BPS];
            for w in [None, Some(&w)] {
                let disto = super::itransform_disto(&b, &in1, &mut d3, &a, w);
                assert_eq!(d3, d2);
                let t = w.map_or(0, |w| super::tdisto4x4(&a, &d2, w));
                assert_eq!(disto, (super::sse4x4(&a, &d2), t));
            }
            let two: Vec<i16> = in1.iter().chain(in2.iter()).copied().collect();
            let (mut e1, mut e2) = ([0u8; 4 * BPS], [0u8; 4 * BPS]);
            super::itransform(&b, &two, &mut e1, true);
            itransform_one(&b, &two[..16], &mut e2);
            itransform_one(&b[4..], &two[16..], &mut e2[4..]);
            assert_eq!(e1, e2);
            for w in [None, Some(&w)] {
                let mut levels = [0i16; 16];
                let got = super::reconstruct4x4_disto(&a, &b, &mtx, &mut levels, &mut d1, w);
                let (mut c, mut l) = ([0i16; 16], [0i16; 16]);
                super::ftransform(&a, &b, &mut c);
                super::quantize_block(&mut c, &mut l, &mtx);
                let (sse, t) = super::itransform_disto(&b, &c, &mut d2, &a, w);
                let nz = super::non_zero_mask(&l);
                assert_eq!((got, levels, d1), ((nz, sse, t), l, d2));
            }
            assert_eq!(super::sse16x8(&a, &b), sse(&a, &b, 16, 8));
        }
    }
}
