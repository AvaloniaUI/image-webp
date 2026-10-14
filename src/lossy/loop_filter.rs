//! Does loop filtering on webp lossy images

#[inline]
fn c(val: i32) -> i32 {
    val.clamp(-128, 127)
}

//unsigned to signed
#[inline]
fn u2s(val: u8) -> i32 {
    i32::from(val) - 128
}

//signed to unsigned
#[inline]
fn s2u(val: i32) -> u8 {
    (c(val) + 128) as u8
}

#[inline]
const fn diff(val1: u8, val2: u8) -> u8 {
    u8::abs_diff(val1, val2)
}

/// Used in both the simple and normal filters described in 15.2 and 15.3
///
/// Adjusts the 2 middle pixels in a vertical loop filter
#[cfg(any(test, feature = "_benchmarks"))]
fn common_adjust_vertical(
    use_outer_taps: bool,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) -> i32 {
    let p1 = u2s(pixels[point - 2 * stride]);
    let p0 = u2s(pixels[point - stride]);
    let q0 = u2s(pixels[point]);
    let q1 = u2s(pixels[point + stride]);

    //value for the outer 2 pixels
    let outer = if use_outer_taps { c(p1 - q1) } else { 0 };

    let a = c(outer + 3 * (q0 - p0));

    let b = (c(a + 3)) >> 3;

    let a = (c(a + 4)) >> 3;

    pixels[point] = s2u(q0 - a);
    pixels[point - stride] = s2u(p0 + b);

    a
}

/// Used in both the simple and normal filters described in 15.2 and 15.3
///
/// Adjusts the 2 middle pixels in a horizontal loop filter
fn common_adjust_horizontal(use_outer_taps: bool, pixels: &mut [u8]) -> i32 {
    let p1 = u2s(pixels[2]);
    let p0 = u2s(pixels[3]);
    let q0 = u2s(pixels[4]);
    let q1 = u2s(pixels[5]);

    //value for the outer 2 pixels
    let outer = if use_outer_taps { c(p1 - q1) } else { 0 };

    let a = c(outer + 3 * (q0 - p0));

    let b = (c(a + 3)) >> 3;

    let a = (c(a + 4)) >> 3;

    pixels[4] = s2u(q0 - a);
    pixels[3] = s2u(p0 + b);

    a
}

#[inline]
#[cfg(any(test, feature = "_benchmarks"))]
fn simple_threshold_vertical(
    filter_limit: i32,
    pixels: &[u8],
    point: usize,
    stride: usize,
) -> bool {
    i32::from(diff(pixels[point - stride], pixels[point])) * 2
        + i32::from(diff(pixels[point - 2 * stride], pixels[point + stride])) / 2
        <= filter_limit
}

#[inline]
fn simple_threshold_horizontal(filter_limit: i32, pixels: &[u8]) -> bool {
    assert!(pixels.len() >= 6); // one bounds check up front eliminates all subsequent checks in this function
    i32::from(diff(pixels[3], pixels[4])) * 2 + i32::from(diff(pixels[2], pixels[5])) / 2
        <= filter_limit
}

#[cfg(any(test, feature = "_benchmarks"))]
fn should_filter_vertical(
    interior_limit: u8,
    edge_limit: u8,
    pixels: &[u8],
    point: usize,
    stride: usize,
) -> bool {
    simple_threshold_vertical(i32::from(edge_limit), pixels, point, stride)
        // this looks like an erroneous way to compute differences between 8 points, but isn't:
        // there are actually only 6 diff comparisons required as per the spec:
        // https://www.rfc-editor.org/rfc/rfc6386#section-20.6
        && diff(pixels[point - 4 * stride], pixels[point - 3 * stride]) <= interior_limit
        && diff(pixels[point - 3 * stride], pixels[point - 2 * stride]) <= interior_limit
        && diff(pixels[point - 2 * stride], pixels[point - stride]) <= interior_limit
        && diff(pixels[point + 3 * stride], pixels[point + 2 * stride]) <= interior_limit
        && diff(pixels[point + 2 * stride], pixels[point + stride]) <= interior_limit
        && diff(pixels[point + stride], pixels[point]) <= interior_limit
}

fn should_filter_horizontal(interior_limit: u8, edge_limit: u8, pixels: &[u8]) -> bool {
    assert!(pixels.len() >= 8); // one bounds check up front eliminates all subsequent checks in this function
    simple_threshold_horizontal(i32::from(edge_limit), pixels)
        // this looks like an erroneous way to compute differences between 8 points, but isn't:
        // there are actually only 6 diff comparisons required as per the spec:
        // https://www.rfc-editor.org/rfc/rfc6386#section-20.6
        && diff(pixels[0], pixels[1]) <= interior_limit
        && diff(pixels[1], pixels[2]) <= interior_limit
        && diff(pixels[2], pixels[3]) <= interior_limit
        && diff(pixels[7], pixels[6]) <= interior_limit
        && diff(pixels[6], pixels[5]) <= interior_limit
        && diff(pixels[5], pixels[4]) <= interior_limit
}

#[inline]
#[cfg(any(test, feature = "_benchmarks"))]
fn high_edge_variance_vertical(threshold: u8, pixels: &[u8], point: usize, stride: usize) -> bool {
    diff(pixels[point - 2 * stride], pixels[point - stride]) > threshold
        || diff(pixels[point + stride], pixels[point]) > threshold
}

#[inline]
fn high_edge_variance_horizontal(threshold: u8, pixels: &[u8]) -> bool {
    diff(pixels[2], pixels[3]) > threshold || diff(pixels[5], pixels[4]) > threshold
}

/// Part of the simple filter described in 15.2 in the specification
///
/// Affects 4 pixels on an edge(2 each side)
#[cfg(any(test, feature = "_benchmarks"))]
pub(crate) fn simple_segment_vertical(
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if simple_threshold_vertical(i32::from(edge_limit), pixels, point, stride) {
        common_adjust_vertical(true, pixels, point, stride);
    }
}

/// Part of the simple filter described in 15.2 in the specification
///
/// Affects 4 pixels on an edge(2 each side)
pub(crate) fn simple_segment_horizontal(edge_limit: u8, pixels: &mut [u8]) {
    if simple_threshold_horizontal(i32::from(edge_limit), pixels) {
        common_adjust_horizontal(true, pixels);
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the edges between subblocks inside a macroblock
#[cfg(any(test, feature = "_benchmarks"))]
pub(crate) fn subblock_filter_vertical(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if should_filter_vertical(interior_limit, edge_limit, pixels, point, stride) {
        let hv = high_edge_variance_vertical(hev_threshold, pixels, point, stride);

        let a = (common_adjust_vertical(hv, pixels, point, stride) + 1) >> 1;

        if !hv {
            pixels[point + stride] = s2u(u2s(pixels[point + stride]) - a);
            pixels[point - 2 * stride] = s2u(u2s(pixels[point - 2 * stride]) + a);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the edges between subblocks inside a macroblock
pub(crate) fn subblock_filter_horizontal(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
) {
    if should_filter_horizontal(interior_limit, edge_limit, pixels) {
        let hv = high_edge_variance_horizontal(hev_threshold, pixels);

        let a = (common_adjust_horizontal(hv, pixels) + 1) >> 1;

        if !hv {
            pixels[5] = s2u(u2s(pixels[5]) - a);
            pixels[2] = s2u(u2s(pixels[2]) + a);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the vertical edges between macroblocks\
/// The point passed in must be the first vertical pixel on the bottom macroblock
#[cfg(any(test, feature = "_benchmarks"))]
pub(crate) fn macroblock_filter_vertical(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    if should_filter_vertical(interior_limit, edge_limit, pixels, point, stride) {
        if !high_edge_variance_vertical(hev_threshold, pixels, point, stride) {
            // p0-3 are the pixels on the left macroblock from right to left
            let p2 = u2s(pixels[point - 3 * stride]);
            let p1 = u2s(pixels[point - 2 * stride]);
            let p0 = u2s(pixels[point - stride]);
            // q0-3 are the pixels on the right macroblock from left to right
            let q0 = u2s(pixels[point]);
            let q1 = u2s(pixels[point + stride]);
            let q2 = u2s(pixels[point + 2 * stride]);

            let w = c(c(p1 - q1) + 3 * (q0 - p0));

            let a = c((27 * w + 63) >> 7);

            pixels[point] = s2u(q0 - a);
            pixels[point - stride] = s2u(p0 + a);

            let a = c((18 * w + 63) >> 7);

            pixels[point + stride] = s2u(q1 - a);
            pixels[point - 2 * stride] = s2u(p1 + a);

            let a = c((9 * w + 63) >> 7);

            pixels[point + 2 * stride] = s2u(q2 - a);
            pixels[point - 3 * stride] = s2u(p2 + a);
        } else {
            common_adjust_vertical(true, pixels, point, stride);
        }
    }
}

/// Part of the normal filter described in 15.3 in the specification
///
/// Filters on the 8 pixels on the horizontal edges between macroblocks\
/// The pixels passed in must be a slice containing the 4 pixels on each macroblock
pub(crate) fn macroblock_filter_horizontal(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
) {
    assert!(pixels.len() >= 8);
    if should_filter_horizontal(interior_limit, edge_limit, pixels) {
        if !high_edge_variance_horizontal(hev_threshold, pixels) {
            // p0-3 are the pixels on the left macroblock from right to left
            let p2 = u2s(pixels[1]);
            let p1 = u2s(pixels[2]);
            let p0 = u2s(pixels[3]);
            // q0-3 are the pixels on the right macroblock from left to right
            let q0 = u2s(pixels[4]);
            let q1 = u2s(pixels[5]);
            let q2 = u2s(pixels[6]);

            let w = c(c(p1 - q1) + 3 * (q0 - p0));

            let a = c((27 * w + 63) >> 7);

            pixels[4] = s2u(q0 - a);
            pixels[3] = s2u(p0 + a);

            let a = c((18 * w + 63) >> 7);

            pixels[5] = s2u(q1 - a);
            pixels[2] = s2u(p1 + a);

            let a = c((9 * w + 63) >> 7);

            pixels[6] = s2u(q2 - a);
            pixels[1] = s2u(p2 + a);
        } else {
            common_adjust_horizontal(true, pixels);
        }
    }
}

/// Rows `point + k * stride` for k in -4..4 of `N` neighbouring columns, as signed values.
#[inline(always)]
fn load_rows<const N: usize>(pixels: &[u8], point: usize, stride: usize) -> [[i32; N]; 8] {
    let mut rows = [[0i32; N]; 8];
    for (k, row) in rows.iter_mut().enumerate() {
        let start = point + k * stride - 4 * stride;
        for (r, &p) in row.iter_mut().zip(&pixels[start..start + N]) {
            *r = i32::from(p) - 128;
        }
    }
    rows
}

#[inline(always)]
fn store_row<const N: usize>(pixels: &mut [u8], at: usize, row: &[i32; N]) {
    for (p, &v) in pixels[at..at + N].iter_mut().zip(row) {
        *p = (c(v) + 128) as u8;
    }
}

#[inline(always)]
fn adiff(a: i32, b: i32) -> i32 {
    (a - b).abs()
}

/// `simple_segment_vertical` on `N` neighbouring columns.
pub(crate) fn simple_segment_vertical_n<const N: usize>(
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    let r = load_rows::<N>(pixels, point, stride);
    let (mut p0n, mut q0n) = ([0i32; N], [0i32; N]);
    for i in 0..N {
        let (p1, p0, q0, q1) = (r[2][i], r[3][i], r[4][i], r[5][i]);
        let filter = adiff(p0, q0) * 2 + adiff(p1, q1) / 2 <= i32::from(edge_limit);
        let a = c(c(p1 - q1) + 3 * (q0 - p0));
        let b = c(a + 3) >> 3;
        let a = c(a + 4) >> 3;
        q0n[i] = if filter { q0 - a } else { q0 };
        p0n[i] = if filter { p0 + b } else { p0 };
    }
    store_row(pixels, point - stride, &p0n);
    store_row(pixels, point, &q0n);
}

#[inline(always)]
fn should_filter_lane<const N: usize>(
    r: &[[i32; N]; 8],
    i: usize,
    interior_limit: u8,
    edge_limit: u8,
) -> bool {
    let il = i32::from(interior_limit);
    adiff(r[3][i], r[4][i]) * 2 + adiff(r[2][i], r[5][i]) / 2 <= i32::from(edge_limit)
        && adiff(r[0][i], r[1][i]) <= il
        && adiff(r[1][i], r[2][i]) <= il
        && adiff(r[2][i], r[3][i]) <= il
        && adiff(r[7][i], r[6][i]) <= il
        && adiff(r[6][i], r[5][i]) <= il
        && adiff(r[5][i], r[4][i]) <= il
}

/// `subblock_filter_vertical` on `N` neighbouring columns.
pub(crate) fn subblock_filter_vertical_n<const N: usize>(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    let r = load_rows::<N>(pixels, point, stride);
    let mut out = [[0i32; N]; 4];
    for i in 0..N {
        let (p1, p0, q0, q1) = (r[2][i], r[3][i], r[4][i], r[5][i]);
        let filter = should_filter_lane(&r, i, interior_limit, edge_limit);
        let hv =
            adiff(p1, p0) > i32::from(hev_threshold) || adiff(q1, q0) > i32::from(hev_threshold);
        let outer = if hv { c(p1 - q1) } else { 0 };
        let a = c(outer + 3 * (q0 - p0));
        let b = c(a + 3) >> 3;
        let a = c(a + 4) >> 3;
        let a3 = (a + 1) >> 1;
        let (np1, nq1) = if hv {
            (p1, q1)
        } else {
            (c(p1 + a3), c(q1 - a3))
        };
        out[0][i] = if filter { np1 } else { p1 };
        out[1][i] = if filter { p0 + b } else { p0 };
        out[2][i] = if filter { q0 - a } else { q0 };
        out[3][i] = if filter { nq1 } else { q1 };
    }
    for (k, row) in out.iter().enumerate() {
        store_row(pixels, point + k * stride - 2 * stride, row);
    }
}

/// `macroblock_filter_vertical` on `N` neighbouring columns.
pub(crate) fn macroblock_filter_vertical_n<const N: usize>(
    hev_threshold: u8,
    interior_limit: u8,
    edge_limit: u8,
    pixels: &mut [u8],
    point: usize,
    stride: usize,
) {
    let r = load_rows::<N>(pixels, point, stride);
    let mut out = [[0i32; N]; 6];
    for i in 0..N {
        let (p2, p1, p0, q0, q1, q2) = (r[1][i], r[2][i], r[3][i], r[4][i], r[5][i], r[6][i]);
        let filter = should_filter_lane(&r, i, interior_limit, edge_limit);
        let hv =
            adiff(p1, p0) > i32::from(hev_threshold) || adiff(q1, q0) > i32::from(hev_threshold);
        let w = c(c(p1 - q1) + 3 * (q0 - p0));
        let a27 = c((27 * w + 63) >> 7);
        let a18 = c((18 * w + 63) >> 7);
        let a9 = c((9 * w + 63) >> 7);
        let b = c(w + 3) >> 3;
        let a = c(w + 4) >> 3;
        let soft = [
            c(p2 + a9),
            c(p1 + a18),
            c(p0 + a27),
            c(q0 - a27),
            c(q1 - a18),
            c(q2 - a9),
        ];
        let hard = [p2, p1, c(p0 + b), c(q0 - a), q1, q2];
        let orig = [p2, p1, p0, q0, q1, q2];
        for k in 0..6 {
            out[k][i] = if !filter {
                orig[k]
            } else if hv {
                hard[k]
            } else {
                soft[k]
            };
        }
    }
    for (k, row) in out.iter().enumerate() {
        store_row(pixels, point + k * stride - 3 * stride, row);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) use super::simd::{
    h16, h16i, h8, h8i, simple_h16, simple_h16i, simple_v16, simple_v16i, v16, v16i, v8, v8i,
};
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) use portable::{
    h16, h16i, h8, h8i, simple_h16, simple_h16i, simple_v16, simple_v16i, v16, v16i, v8, v8i,
};

/// libwebp's edge filters (`VP8VFilter16`, `VP8HFilter8i`, ...): `p` is the first pixel past
/// the edge, or the macroblock origin for the inner edges (`i`).
#[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
pub(crate) mod portable {
    use super::*;

    pub(crate) fn simple_v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
        simple_segment_vertical_n::<16>(thresh, buf, p, stride);
    }

    pub(crate) fn simple_h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
        for y in 0..16 {
            simple_segment_horizontal(thresh, &mut buf[p + y * stride - 4..][..8]);
        }
    }

    pub(crate) fn simple_v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
        for k in 1..4 {
            simple_v16(buf, p + 4 * k * stride, stride, thresh);
        }
    }

    pub(crate) fn simple_h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8) {
        for k in 1..4 {
            simple_h16(buf, p + 4 * k, stride, thresh);
        }
    }

    pub(crate) fn v16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
        macroblock_filter_vertical_n::<16>(hev, ithresh, thresh, buf, p, stride);
    }

    pub(crate) fn h16(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
        for y in 0..16 {
            macroblock_filter_horizontal(hev, ithresh, thresh, &mut buf[p + y * stride - 4..][..8]);
        }
    }

    pub(crate) fn v16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
        for k in 1..4 {
            subblock_filter_vertical_n::<16>(hev, ithresh, thresh, buf, p + 4 * k * stride, stride);
        }
    }

    pub(crate) fn h16i(buf: &mut [u8], p: usize, stride: usize, thresh: u8, ithresh: u8, hev: u8) {
        for k in 1..4 {
            for y in 0..16 {
                subblock_filter_horizontal(
                    hev,
                    ithresh,
                    thresh,
                    &mut buf[p + 4 * k + y * stride - 4..][..8],
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn v8(
        u: &mut [u8],
        v: &mut [u8],
        p: usize,
        stride: usize,
        thresh: u8,
        ithresh: u8,
        hev: u8,
    ) {
        macroblock_filter_vertical_n::<8>(hev, ithresh, thresh, u, p, stride);
        macroblock_filter_vertical_n::<8>(hev, ithresh, thresh, v, p, stride);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h8(
        u: &mut [u8],
        v: &mut [u8],
        p: usize,
        stride: usize,
        thresh: u8,
        ithresh: u8,
        hev: u8,
    ) {
        for y in 0..8 {
            macroblock_filter_horizontal(hev, ithresh, thresh, &mut u[p + y * stride - 4..][..8]);
            macroblock_filter_horizontal(hev, ithresh, thresh, &mut v[p + y * stride - 4..][..8]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn v8i(
        u: &mut [u8],
        v: &mut [u8],
        p: usize,
        stride: usize,
        thresh: u8,
        ithresh: u8,
        hev: u8,
    ) {
        subblock_filter_vertical_n::<8>(hev, ithresh, thresh, u, p + 4 * stride, stride);
        subblock_filter_vertical_n::<8>(hev, ithresh, thresh, v, p + 4 * stride, stride);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h8i(
        u: &mut [u8],
        v: &mut [u8],
        p: usize,
        stride: usize,
        thresh: u8,
        ithresh: u8,
        hev: u8,
    ) {
        for y in 0..8 {
            subblock_filter_horizontal(hev, ithresh, thresh, &mut u[p + y * stride..][..8]);
            subblock_filter_horizontal(hev, ithresh, thresh, &mut v[p + y * stride..][..8]);
        }
    }
}

#[cfg(all(test, feature = "_benchmarks"))]
mod benches {
    use super::*;
    use test::{black_box, Bencher};

    #[rustfmt::skip]
    const TEST_DATA: [u8; 8 * 8] = [
        177, 192, 179, 181, 185, 174, 186, 193,
        185, 180, 175, 179, 175, 190, 189, 190,
        185, 181, 177, 190, 190, 174, 176, 188,
        192, 179, 186, 175, 190, 184, 190, 175,
        175, 183, 183, 190, 187, 186, 176, 181,
        183, 177, 182, 185, 183, 179, 178, 181,
        191, 183, 188, 181, 180, 193, 185, 180,
        177, 182, 177, 178, 179, 178, 191, 178,
    ];

    #[bench]
    fn measure_horizontal_macroblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for y in 0..8 {
                black_box(macroblock_filter_horizontal(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data[y * stride..][..8],
                ));
            }
        });
    }

    #[bench]
    fn measure_vertical_macroblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for x in 0..8 {
                black_box(macroblock_filter_vertical(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data,
                    4 * stride + x,
                    stride,
                ));
            }
        });
    }

    #[bench]
    fn measure_horizontal_subblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for y in 0usize..8 {
                black_box(subblock_filter_horizontal(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data[y * stride..][..8],
                ))
            }
        });
    }

    #[bench]
    fn measure_vertical_subblock_filter(b: &mut Bencher) {
        let hev_threshold = 5;
        let interior_limit = 15;
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for x in 0..8 {
                black_box(subblock_filter_vertical(
                    hev_threshold,
                    interior_limit,
                    edge_limit,
                    &mut data,
                    4 * stride + x,
                    stride,
                ))
            }
        });
    }

    #[bench]
    fn measure_simple_segment_horizontal_filter(b: &mut Bencher) {
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for y in 0usize..8 {
                black_box(simple_segment_horizontal(
                    edge_limit,
                    &mut data[y * stride..][..8],
                ))
            }
        });
    }

    #[bench]
    fn measure_simple_segment_vertical_filter(b: &mut Bencher) {
        let edge_limit = 15;

        let mut data = TEST_DATA.clone();
        let stride = 8;

        b.iter(|| {
            for x in 0usize..16 {
                black_box(simple_segment_vertical(
                    edge_limit,
                    &mut data,
                    4 * stride + x,
                    stride,
                ))
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_batches_match_scalar() {
        let mut seed = 0x2545_f491u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let stride = 16;
        for round in 0..2000 {
            // Smooth columns with small steps, so that every branch is taken.
            let spread = [2u32, 8, 40, 255][round % 4];
            let mut pixels = vec![0u8; stride * 8];
            for x in 0..stride {
                let mut v = (rnd() & 0xff) as i32;
                for y in 0..8 {
                    v = (v + (rnd() % (2 * spread + 1)) as i32 - spread as i32).clamp(0, 255);
                    pixels[y * stride + x] = v as u8;
                }
            }
            let (hev, interior, edge) = (
                (rnd() % 8) as u8,
                (rnd() % 20 + 1) as u8,
                (rnd() % 80) as u8,
            );
            let point = 4 * stride;
            let mut a = pixels.clone();
            let mut b = pixels.clone();
            for x in 0..stride {
                macroblock_filter_vertical(hev, interior, edge, &mut a, point + x, stride);
            }
            macroblock_filter_vertical_n::<16>(hev, interior, edge, &mut b, point, stride);
            assert_eq!(a, b, "macroblock");
            let mut a = pixels.clone();
            let mut b = pixels.clone();
            for x in 0..stride {
                subblock_filter_vertical(hev, interior, edge, &mut a, point + x, stride);
            }
            subblock_filter_vertical_n::<16>(hev, interior, edge, &mut b, point, stride);
            assert_eq!(a, b, "subblock");
            let mut a = pixels.clone();
            let mut b = pixels;
            for x in 0..stride {
                simple_segment_vertical(edge, &mut a, point + x, stride);
            }
            simple_segment_vertical_n::<16>(edge, &mut b, point, stride);
            assert_eq!(a, b, "simple");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn simd_edges_match_portable() {
        use super::portable as pf;
        use crate::lossy::simd as sf;
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let stride = 40;
        for round in 0..3000 {
            let spread = [1u32, 3, 8, 40, 255][round % 5];
            let mut fill = |len: usize| {
                let mut v = (rnd() & 0xff) as i32;
                let mut buf = vec![0u8; len];
                for px in buf.iter_mut() {
                    v = (v + (rnd() % (2 * spread + 1)) as i32 - spread as i32).clamp(0, 255);
                    *px = v as u8;
                }
                buf
            };
            let y = fill(stride * 24);
            let u = fill(stride * 16);
            let v = fill(stride * 16);
            let thresh = (rnd() % 200) as u8;
            let ithresh = (rnd() % 64) as u8;
            let hev = (rnd() % 4) as u8;
            let p = 4 * stride + 4 + (rnd() % 8) as usize;
            macro_rules! same16 {
                ($f:ident $(, $arg:expr)*) => {{
                    let (mut a, mut b) = (y.clone(), y.clone());
                    pf::$f(&mut a, p, stride $(, $arg)*);
                    sf::$f(&mut b, p, stride $(, $arg)*);
                    assert!(a == b, "{} round {round}", stringify!($f));
                }};
            }
            macro_rules! same8 {
                ($f:ident) => {{
                    let (mut a, mut b, mut c, mut d) = (u.clone(), v.clone(), u.clone(), v.clone());
                    pf::$f(&mut a, &mut b, p, stride, thresh, ithresh, hev);
                    sf::$f(&mut c, &mut d, p, stride, thresh, ithresh, hev);
                    assert!(a == c && b == d, "{} round {round}", stringify!($f));
                }};
            }
            same16!(simple_v16, thresh);
            same16!(simple_h16, thresh);
            same16!(simple_v16i, thresh);
            same16!(simple_h16i, thresh);
            same16!(v16, thresh, ithresh, hev);
            same16!(h16, thresh, ithresh, hev);
            same16!(v16i, thresh, ithresh, hev);
            same16!(h16i, thresh, ithresh, hev);
            same8!(v8);
            same8!(h8);
            same8!(v8i);
            same8!(h8i);
        }
    }
}
