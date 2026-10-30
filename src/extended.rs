use super::lossless::{AlphaBands, LosslessDecoder};
use crate::decoder::DecodingError;
use byteorder_lite::ReadBytesExt;
use std::io::{BufRead, Read};

use crate::alpha_blending::do_alpha_blending;

/// Composites a frame onto a canvas.
///
/// Starts by filling the rectangle occupied by the previous frame with the background
/// color, if provided. Then copies or blends the frame onto the canvas.
#[allow(clippy::too_many_arguments)]
pub(crate) fn composite_frame(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    clear_color: Option<[u8; 4]>,
    frame: &[u8],
    frame_offset_x: u32,
    frame_offset_y: u32,
    frame_width: u32,
    frame_height: u32,
    frame_has_alpha: bool,
    frame_use_alpha_blending: bool,
    previous_frame_width: u32,
    previous_frame_height: u32,
    previous_frame_offset_x: u32,
    previous_frame_offset_y: u32,
) {
    let frame_is_full_size = frame_offset_x == 0
        && frame_offset_y == 0
        && frame_width == canvas_width
        && frame_height == canvas_height;

    if frame_is_full_size && !frame_use_alpha_blending {
        if frame_has_alpha {
            canvas.copy_from_slice(frame);
        } else {
            for (input, output) in frame.chunks_exact(3).zip(canvas.chunks_exact_mut(4)) {
                output[..3].copy_from_slice(input);
                output[3] = 255;
            }
        }
        return;
    }

    // Clear rectangle occupied by previous frame.
    // The canvas is always RGBA (4 bytes/pixel) regardless of whether the
    // current frame carries alpha, so we always clear with 4-byte pixels.
    if let Some(clear_color) = clear_color {
        if frame_is_full_size {
            for pixel in canvas.chunks_exact_mut(4) {
                pixel.copy_from_slice(&clear_color);
            }
        } else {
            for y in 0..previous_frame_height as usize {
                for x in 0..previous_frame_width as usize {
                    let canvas_index = ((x + previous_frame_offset_x as usize)
                        + (y + previous_frame_offset_y as usize) * canvas_width as usize)
                        * 4;

                    let output = &mut canvas[canvas_index..][..4];
                    output.copy_from_slice(&clear_color);
                }
            }
        }
    }

    let width = frame_width.min(canvas_width.saturating_sub(frame_offset_x)) as usize;
    let height = frame_height.min(canvas_height.saturating_sub(frame_offset_y)) as usize;

    if frame_has_alpha && frame_use_alpha_blending {
        for y in 0..height {
            for x in 0..width {
                let frame_index = (x + y * frame_width as usize) * 4;
                let canvas_index = ((x + frame_offset_x as usize)
                    + (y + frame_offset_y as usize) * canvas_width as usize)
                    * 4;

                let input = &frame[frame_index..][..4];
                let output = &mut canvas[canvas_index..][..4];

                let blended =
                    do_alpha_blending(input.try_into().unwrap(), output.try_into().unwrap());
                output.copy_from_slice(&blended);
            }
        }
    } else if frame_has_alpha {
        for y in 0..height {
            let frame_index = (y * frame_width as usize) * 4;
            let canvas_index = (frame_offset_x as usize
                + (y + frame_offset_y as usize) * canvas_width as usize)
                * 4;

            canvas[canvas_index..][..width * 4].copy_from_slice(&frame[frame_index..][..width * 4]);
        }
    } else {
        for y in 0..height {
            let index = (y * frame_width as usize) * 3;
            let canvas_index = (frame_offset_x as usize
                + (y + frame_offset_y as usize) * canvas_width as usize)
                * 4;
            let input = &frame[index..][..width * 3];
            let output = &mut canvas[canvas_index..][..width * 4];

            for (input, output) in input.chunks_exact(3).zip(output.chunks_exact_mut(4)) {
                output[..3].copy_from_slice(input);
                output[3] = 255;
            }
        }
    }
}

pub(crate) fn unfilter_alpha_row(method: FilteringMethod, prev: Option<&[u8]>, row: &mut [u8]) {
    if row.is_empty() {
        return;
    }
    // The left neighbour stays in a register rather than being read back from `row`.
    match (method, prev) {
        (FilteringMethod::None, _) => {}
        (FilteringMethod::Horizontal, _) | (_, None) => {
            if let (FilteringMethod::Horizontal, Some(prev)) = (method, prev) {
                row[0] = row[0].wrapping_add(prev[0]);
            }
            #[cfg(decoder_simd)]
            let x = crate::lossy::unfilter_horizontal(row, 1);
            #[cfg(not(decoder_simd))]
            let x = 1;
            let mut left = row[x - 1];
            for a in &mut row[x..] {
                left = a.wrapping_add(left);
                *a = left;
            }
        }
        (FilteringMethod::Vertical, Some(prev)) => {
            for (a, &p) in row.iter_mut().zip(prev) {
                *a = a.wrapping_add(p);
            }
        }
        (FilteringMethod::Gradient, Some(prev)) => {
            row[0] = row[0].wrapping_add(prev[0]);
            let mut x = 1;
            while x < row.len() {
                #[cfg(decoder_simd)]
                {
                    x = crate::lossy::unfilter_gradient(prev, row, x);
                }
                // After a block the running sum stopped at, the next 16 pixels; without it, all.
                #[cfg(decoder_simd)]
                let end = row.len().min(x + 16);
                #[cfg(not(decoder_simd))]
                let end = row.len();
                let mut left = row[x - 1];
                for ((a, &top), &top_left) in
                    row[x..end].iter_mut().zip(&prev[x..]).zip(&prev[x - 1..])
                {
                    let mut g = i32::from(left) + (i32::from(top) - i32::from(top_left));
                    // Rarely taken, so the clamp stays off the chain from one pixel to the next.
                    if g as u32 > 255 {
                        g = clamp_gradient(g);
                    }
                    left = a.wrapping_add(g as u8);
                    *a = left;
                }
                x = end;
            }
        }
    }
}

#[cold]
#[inline(never)]
fn clamp_gradient(g: i32) -> i32 {
    g.clamp(0, 255)
}

/// An ALPH chunk's plane handed out a row at a time, top to bottom, decoded and unfiltered as
/// the rows are asked for: libwebp's `VP8DecompressAlphaRows`.
pub(crate) struct AlphaRows<'a> {
    filtering_method: FilteringMethod,
    source: AlphaSource<'a>,
    prev: Vec<u8>,
    row: Vec<u8>,
    y: usize,
}

enum AlphaSource<'a> {
    Raw(Box<dyn BufRead + Send + 'a>),
    Lossless(AlphaBands<Box<dyn BufRead + Send + 'a>>),
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum FilteringMethod {
    None,
    Horizontal,
    Vertical,
    Gradient,
}

impl AlphaRows<'_> {
    /// The next row of the plane.
    pub(crate) fn next_row(&mut self) -> Result<&[u8], DecodingError> {
        std::mem::swap(&mut self.prev, &mut self.row);
        match &mut self.source {
            AlphaSource::Raw(r) => r.read_exact(&mut self.row)?,
            AlphaSource::Lossless(bands) => bands.read_row(self.y, &mut self.row)?,
        }
        let prev = (self.y > 0).then_some(&self.prev[..]);
        unfilter_alpha_row(self.filtering_method, prev, &mut self.row);
        self.y += 1;
        Ok(&self.row)
    }
}

pub(crate) fn read_alpha_chunk<'a>(
    mut reader: Box<dyn BufRead + Send + 'a>,
    width: u16,
    height: u16,
) -> Result<AlphaRows<'a>, DecodingError> {
    let info_byte = reader.read_u8()?;

    let preprocessing = (info_byte & 0b00110000) >> 4;
    let filtering = (info_byte & 0b00001100) >> 2;
    let compression = info_byte & 0b00000011;

    if preprocessing > 1 {
        return Err(DecodingError::InvalidAlphaPreprocessing);
    }

    let filtering_method = match filtering {
        0 => FilteringMethod::None,
        1 => FilteringMethod::Horizontal,
        2 => FilteringMethod::Vertical,
        3 => FilteringMethod::Gradient,
        _ => unreachable!(),
    };

    let source = match compression {
        0 => AlphaSource::Raw(reader),
        1 => AlphaSource::Lossless(LosslessDecoder::new(reader).alpha(width, height)?),
        _ => return Err(DecodingError::InvalidCompressionMethod),
    };

    Ok(AlphaRows {
        filtering_method,
        source,
        prev: vec![0; usize::from(width)],
        row: vec![0; usize::from(width)],
        y: 0,
    })
}

#[cfg(all(test, feature = "encoder"))]
mod tests {
    use super::*;
    use crate::lossless::encoder::encode_alpha_filtered;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// The plane as it was decoded before it came a row at a time: whole, then unfiltered.
    fn whole(mut payload: &[u8], w: u16, h: u16) -> Result<Vec<u8>, DecodingError> {
        let info = payload.read_u8()?;
        let filter = [FilteringMethod::None, FilteringMethod::Horizontal, FilteringMethod::Vertical, FilteringMethod::Gradient]
            [usize::from(info >> 2 & 3)];
        let mut data = match info & 3 {
            0 => {
                let mut d = vec![0; usize::from(w) * usize::from(h)];
                payload.read_exact(&mut d)?;
                d
            }
            _ => LosslessDecoder::new(payload).decode_alpha_whole(w, h)?,
        };
        let w = usize::from(w);
        for y in 0..usize::from(h) {
            let (above, rest) = data.split_at_mut(y * w);
            unfilter_alpha_row(filter, (y > 0).then(|| &above[(y - 1) * w..]), &mut rest[..w]);
        }
        Ok(data)
    }

    fn banded(payload: &[u8], w: u16, h: u16) -> Result<Vec<u8>, DecodingError> {
        let mut rows = read_alpha_chunk(Box::new(payload), w, h)?;
        let mut out = Vec::new();
        for _ in 0..h {
            out.extend_from_slice(rows.next_row()?);
        }
        Ok(out)
    }

    fn planes(w: usize, h: usize, seed: u64) -> Vec<Vec<u8>> {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = w * h;
        vec![
            (0..n).map(|i| if (i % w) * 3 > w + (i / w) { 255 } else { 0 }).collect(),
            (0..n).map(|i| [0, 64, 128, 255][(i / 7 + i % w / 9) % 4]).collect(),
            (0..n).map(|i| ((i % w) * 255 / w.max(2)) as u8).collect(),
            (0..n).map(|_| rng.gen_range(0..12u8) * 20).collect(),
            (0..n).map(|_| rng.gen()).collect(),
            (0..n).map(|i| ((i % w) * 2 + (i / w) * 3 + rng.gen_range(0..3)) as u8).collect(),
        ]
    }

    /// Rows decoded and unfiltered in step with the colour come out as the whole plane did, for
    /// every filter, stored or compressed, and fail where it failed on cut or damaged data.
    #[test]
    fn alpha_rows_match_the_whole_plane() {
        for (w, h) in [(67usize, 41usize), (1, 300), (300, 1), (5, 5), (130, 70)] {
            for (k, plane) in planes(w, h, (w * h) as u64).iter().enumerate() {
                for filter in 0..4 {
                    for method in [None, Some(0), Some(3), Some(6)] {
                        let payload = encode_alpha_filtered(plane, w as u32, h as u32, filter, method);
                        let label = format!("{w}x{h} plane {k} filter {filter} method {method:?}");
                        let got = banded(&payload, w as u16, h as u16).unwrap();
                        assert!(got == *plane, "{label}");
                        assert!(whole(&payload, w as u16, h as u16).unwrap() == got, "{label}");
                        let mut damaged: Vec<Vec<u8>> = [1, 2, 5, payload.len() / 2, payload.len() - 3]
                            .iter()
                            .filter(|&&cut| cut < payload.len())
                            .map(|&cut| payload[..payload.len() - cut].to_vec())
                            .collect();
                        for at in [1, 2, 3, payload.len() / 3, payload.len() - 1] {
                            let mut p = payload.clone();
                            p[at] ^= 0x5a;
                            damaged.push(p);
                        }
                        for (d, p) in damaged.iter().enumerate() {
                            let (a, b) = (whole(p, w as u16, h as u16), banded(p, w as u16, h as u16));
                            match (a, b) {
                                (Ok(a), Ok(b)) => assert!(a == b, "{label} damage {d}"),
                                (Err(_), Err(_)) => {}
                                (a, b) => panic!("{label} damage {d}: whole {:?} banded {:?}", a.is_ok(), b.is_ok()),
                            }
                        }
                    }
                }
            }
        }
    }

    /// Palette indices a byte each, over more pixels than the decoder's window holds.
    #[test]
    fn alpha_rows_past_the_window() {
        let (w, h) = (2100usize, 1100usize);
        let mut rng = StdRng::seed_from_u64(5);
        let mut plane: Vec<u8> = (0..w * 480).map(|_| rng.gen_range(0..40u8) * 6).collect();
        while plane.len() < w * h {
            let at = plane.len();
            plane.push(plane[at - w * 480]);
        }
        for method in [Some(0), Some(4)] {
            let payload = encode_alpha_filtered(&plane, w as u32, h as u32, 0, method);
            let bands = LosslessDecoder::new(&payload[1..]).alpha(w as u16, h as u16).unwrap();
            let indices = matches!(bands, AlphaBands::Indices(_));
            let got = banded(&payload, w as u16, h as u16).unwrap();
            assert!(got == plane, "method {method:?} indices {indices}");
            assert!(whole(&payload, w as u16, h as u16).unwrap() == got);
            assert!(indices || method != Some(0), "a byte per index at method 0");
        }
    }

    /// Regression test: clearing the canvas for a non-alpha frame used 3-byte
    /// stride on the always-RGBA canvas, corrupting pixel alignment.
    #[test]
    fn dispose_clear_fullsize_rgb_frame() {
        let w = 4u32;
        let h = 4u32;
        let mut canvas = vec![0xAA_u8; (w * h * 4) as usize];
        let frame = vec![0xFF_u8; (w * h * 3) as usize];

        composite_frame(
            &mut canvas,
            w,
            h,
            Some([0, 0, 0, 0]),
            &frame,
            0,
            0,
            w,
            h,
            false, // frame_has_alpha
            true,  // frame_use_alpha_blending (forces slow path)
            w,
            h,
            0,
            0,
        );

        for (i, pixel) in canvas.chunks_exact(4).enumerate() {
            assert_eq!(
                pixel,
                [0xFF, 0xFF, 0xFF, 0xFF],
                "pixel {i} corrupted: {pixel:?}"
            );
        }
    }

    /// Regression test: sub-frame clear used 3-byte indexing on RGBA canvas.
    #[test]
    fn dispose_clear_subframe_rgb_frame() {
        let canvas_w = 8u32;
        let canvas_h = 8u32;
        let mut canvas = vec![0xAA_u8; (canvas_w * canvas_h * 4) as usize];

        let prev_x = 2u32;
        let prev_y = 2u32;
        let prev_w = 4u32;
        let prev_h = 4u32;

        let frame_w = 4u32;
        let frame_h = 4u32;
        let frame = vec![0xFF_u8; (frame_w * frame_h * 3) as usize];

        composite_frame(
            &mut canvas,
            canvas_w,
            canvas_h,
            Some([0, 0, 0, 0]),
            &frame,
            0,
            0,
            frame_w,
            frame_h,
            false,
            true,
            prev_w,
            prev_h,
            prev_x,
            prev_y,
        );

        let stride = canvas_w as usize * 4;

        // Previous-frame rectangle should be cleared to [0,0,0,0].
        // Only check the region cleared but NOT overwritten by the new frame.
        for y in prev_y as usize..(prev_y + prev_h) as usize {
            for x in prev_x as usize..(prev_x + prev_w) as usize {
                let idx = y * stride + x * 4;
                let pixel = &canvas[idx..idx + 4];
                if x >= frame_w as usize || y >= frame_h as usize {
                    assert_eq!(
                        pixel,
                        [0, 0, 0, 0],
                        "prev-frame pixel ({x},{y}) not cleared: {pixel:?}"
                    );
                }
            }
        }

        // New frame region should be opaque white from RGB→RGBA.
        for y in 0..frame_h as usize {
            for x in 0..frame_w as usize {
                let idx = y * stride + x * 4;
                let pixel = &canvas[idx..idx + 4];
                assert_eq!(
                    pixel,
                    [0xFF, 0xFF, 0xFF, 0xFF],
                    "new-frame pixel ({x},{y}) wrong: {pixel:?}"
                );
            }
        }

        // Pixels outside both rectangles should be untouched.
        let pixel = &canvas[7 * 4..7 * 4 + 4];
        assert_eq!(pixel, [0xAA, 0xAA, 0xAA, 0xAA], "untouched pixel modified");
    }

    #[test]
    fn horizontal_unfilter_is_a_running_sum() {
        for len in 0..80usize {
            let prev: Vec<u8> = (0..len).map(|i| (i * 37 % 251) as u8).collect();
            let row: Vec<u8> = (0..len).map(|i| (i * 91 % 253) as u8).collect();
            for p in [None, Some(&prev[..])] {
                let mut expected = row.clone();
                if let (Some(p), true) = (p, len > 0) {
                    expected[0] = expected[0].wrapping_add(p[0]);
                }
                for x in 1..len {
                    expected[x] = expected[x].wrapping_add(expected[x - 1]);
                }
                let mut got = row.clone();
                unfilter_alpha_row(FilteringMethod::Horizontal, p, &mut got);
                assert_eq!(got, expected, "{len}");
                if p.is_none() {
                    let mut got = row.clone();
                    unfilter_alpha_row(FilteringMethod::Gradient, None, &mut got);
                    assert_eq!(got, expected, "{len}");
                }
            }
        }
    }

    #[test]
    fn gradient_unfilter_clamps() {
        let mut seed = 1u32;
        let mut next = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 24) as u8
        };
        for round in 0..400 {
            let len = 1 + round % 300;
            // Smooth rows with rare jumps take both the running sum and the clamping path.
            let smooth = round % 2 == 0;
            let mut prev = vec![next(); len];
            for x in 1..len {
                prev[x] = if smooth && next() > 8 {
                    prev[x - 1].wrapping_add(next() % 3)
                } else {
                    next()
                };
            }
            let mut row: Vec<u8> = (0..len)
                .map(|_| {
                    if smooth && next() > 8 {
                        next() % 2
                    } else {
                        next()
                    }
                })
                .collect();
            let mut expected = row.clone();
            expected[0] = expected[0].wrapping_add(prev[0]);
            for x in 1..expected.len() {
                let g = (i16::from(expected[x - 1]) + i16::from(prev[x]) - i16::from(prev[x - 1]))
                    .clamp(0, 255) as u8;
                expected[x] = expected[x].wrapping_add(g);
            }
            unfilter_alpha_row(FilteringMethod::Gradient, Some(&prev), &mut row);
            assert_eq!(row, expected, "round {round}");
        }
    }
}
