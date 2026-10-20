//! `AnimationEncoder` round trips through `WebPDecoder`: lossless frames come back pixel for
//! pixel, lossy ones close, frames that do not change merge into the previous one's duration,
//! key frames come every `kmax` frames, a one-frame animation is a still, and the files pass
//! libwebp's animation decoder with the same frames.

use std::io::Cursor;

use image_webp::{AnimationEncoder, AnimationParams, ColorType, EncoderParams, WebPDecoder};
use libwebp_sys as sys;

const W: u32 = 40;
const H: u32 = 30;

fn frame(shift: u32, alpha: bool, sprite: Option<(u32, u32)>) -> Vec<u8> {
    let mut out = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let a = if alpha {
                ((x * 5 + y * 3 + shift) % 256) as u8
            } else {
                255
            };
            let mut px = [
                ((x + shift) * 255 / W) as u8,
                (y * 255 / H) as u8,
                ((x ^ y) * 7 % 256) as u8,
                a,
            ];
            if let Some((sx, sy)) = sprite {
                px = if x >= sx && x < sx + 8 && y >= sy && y < sy + 8 {
                    [200, 40, 40, 255]
                } else {
                    [0, 0, 0, 0]
                };
            }
            out.extend_from_slice(&px);
        }
    }
    out
}

fn encode(frames: &[(Vec<u8>, u32)], lossless: bool, kmin: u32, kmax: u32) -> Vec<u8> {
    let mut enc = AnimationEncoder::new(Cursor::new(Vec::new()), W, H).unwrap();
    let mut params = EncoderParams::default();
    params.use_lossy = !lossless;
    params.lossy_quality = 80;
    enc.set_params(params);
    let mut anim = AnimationParams::default();
    anim.kmin = kmin;
    anim.kmax = kmax;
    anim.loop_count = 3;
    enc.set_animation_params(anim);
    enc.set_exif_metadata(b"Exif\0\0II*\0".to_vec());
    for (f, d) in frames {
        enc.add_frame(f, ColorType::Rgba8, *d).unwrap();
    }
    enc.finish().unwrap().into_inner()
}

/// The composited RGBA canvases and durations as this crate decodes them.
fn decode(data: &[u8]) -> (WebPDecoder<Cursor<Vec<u8>>>, Vec<(Vec<u8>, u32)>) {
    let mut d = WebPDecoder::new(Cursor::new(data.to_vec())).unwrap();
    let mut buf = vec![0u8; d.output_buffer_size().unwrap()];
    let has_alpha = d.has_alpha();
    let rgba = |b: &[u8]| -> Vec<u8> {
        match has_alpha {
            true => b.to_vec(),
            false => b
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        }
    };
    let mut frames = Vec::new();
    if d.is_animated() {
        while let Ok(duration) = d.read_frame(&mut buf) {
            frames.push((rgba(&buf), duration));
        }
    } else {
        d.read_image(&mut buf).unwrap();
        frames.push((rgba(&buf), 0));
    }
    (d, frames)
}

/// The composited RGBA canvases as libwebp's animation decoder shows them.
fn decode_libwebp(data: &[u8]) -> Vec<(Vec<u8>, u32)> {
    unsafe {
        let mut options = std::mem::zeroed::<sys::WebPAnimDecoderOptions>();
        assert_ne!(
            sys::WebPAnimDecoderOptionsInitInternal(
                &mut options,
                sys::WEBP_DEMUX_ABI_VERSION as i32
            ),
            0
        );
        options.color_mode = sys::WEBP_CSP_MODE::MODE_RGBA;
        let webp = sys::WebPData {
            bytes: data.as_ptr(),
            size: data.len(),
        };
        let dec =
            sys::WebPAnimDecoderNewInternal(&webp, &options, sys::WEBP_DEMUX_ABI_VERSION as i32);
        assert!(!dec.is_null());
        let mut info = std::mem::zeroed::<sys::WebPAnimInfo>();
        assert_ne!(sys::WebPAnimDecoderGetInfo(dec, &mut info), 0);
        let size = (info.canvas_width * info.canvas_height * 4) as usize;
        let (mut frames, mut last) = (Vec::new(), 0);
        while sys::WebPAnimDecoderHasMoreFrames(dec) != 0 {
            let (mut buf, mut timestamp): (*mut u8, i32) = (std::ptr::null_mut(), 0);
            assert_ne!(
                sys::WebPAnimDecoderGetNext(dec, &mut buf, &mut timestamp),
                0
            );
            frames.push((
                std::slice::from_raw_parts(buf, size).to_vec(),
                (timestamp - last) as u32,
            ));
            last = timestamp;
        }
        sys::WebPAnimDecoderDelete(dec);
        frames
    }
}

fn masked(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4)
        .flat_map(|p| {
            if p[3] == 0 {
                [0; 4]
            } else {
                [p[0], p[1], p[2], p[3]]
            }
        })
        .collect()
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (i64::from(x) - i64::from(y)).pow(2) as u64)
        .sum();
    10. * (255f64 * 255. * a.len() as f64 / sum.max(1) as f64).log10()
}

fn check(
    frames: &[(Vec<u8>, u32)],
    lossless: bool,
    kmin: u32,
    kmax: u32,
) -> (WebPDecoder<Cursor<Vec<u8>>>, Vec<(Vec<u8>, u32)>) {
    let data = encode(frames, lossless, kmin, kmax);
    let (d, ours) = decode(&data);
    let theirs = decode_libwebp(&data);
    assert_eq!(ours.len(), theirs.len());
    for (o, t) in ours.iter().zip(&theirs) {
        assert!(masked(&o.0) == masked(&t.0), "the decoders disagree");
    }
    let (mut at, mut i) = (0u32, 0usize);
    for (f, duration) in frames {
        while i + 1 < ours.len() && ours[..=i].iter().map(|f| f.1).sum::<u32>() <= at {
            i += 1;
        }
        match lossless {
            true => assert!(
                masked(&ours[i].0) == masked(f),
                "frame {i} is not the source"
            ),
            false => assert!(
                psnr(&masked(&ours[i].0), &masked(f)) > 22.,
                "frame {i} is far from the source"
            ),
        }
        at += duration;
    }
    (d, ours)
}

#[test]
fn frames_come_back() {
    for lossless in [true, false] {
        for alpha in [false, true] {
            let frames: Vec<_> = (0..12)
                .map(|i| (frame(i * 3, alpha, None), 40 + i))
                .collect();
            let (mut d, ours) = check(&frames, lossless, 9, 10);
            assert_eq!(ours.len(), 12);
            assert_eq!(d.animation_params().1, 3);
            assert_eq!(
                d.exif_metadata().unwrap().as_deref(),
                Some(&b"Exif\0\0II*\0"[..])
            );
            let full = |f: &image_webp::FrameInfo| {
                (f.x_offset, f.y_offset, f.width, f.height) == (0, 0, W as i32, H as i32)
            };
            assert!(
                full(&d.frames()[0]) && full(&d.frames()[10]) && d.frames()[10].no_blend,
                "key frames at 0 and kmax"
            );
            assert!(d
                .frames()
                .iter()
                .map(|f| f.duration as u32)
                .eq((0..12).map(|i| 40 + i)));
        }
    }
}

#[test]
fn unchanged_frames_merge_into_the_previous_duration() {
    let still = frame(0, false, None);
    let frames = vec![
        (still.clone(), 100),
        (still.clone(), 50),
        (frame(9, false, None), 70),
        (still.clone(), 30),
        (still, 30),
    ];
    for lossless in [true, false] {
        let (d, ours) = check(&frames, lossless, 9, 10);
        assert_eq!(ours.iter().map(|f| f.1).collect::<Vec<_>>(), [150, 70, 60]);
        assert_eq!(d.frames().len(), 3);
    }
}

#[test]
fn one_frame_is_a_still() {
    let data = encode(&[(frame(0, true, None), 100)], true, 9, 10);
    let (d, ours) = decode(&data);
    assert!(!d.is_animated());
    assert_eq!(masked(&ours[0].0), masked(&frame(0, true, None)));
    let data = encode(
        &[(frame(0, false, None), 100), (frame(0, false, None), 100)],
        false,
        9,
        10,
    );
    assert!(
        !decode(&data).0.is_animated(),
        "frames that never change are a still"
    );
}

#[test]
fn a_sprite_over_transparency_disposes_and_blends() {
    let frames: Vec<_> = (0..8)
        .map(|i| (frame(0, true, Some((i * 4, 6 + i))), 60))
        .collect();
    for lossless in [true, false] {
        let (d, _) = check(&frames, lossless, 9, 10);
        let f = &d.frames()[1];
        assert!(
            f.width < W as i32 / 2 && f.height < H as i32 / 2,
            "the sprite's rectangle, not the canvas"
        );
        assert!(
            d.frames().iter().any(|f| f.dispose_to_background),
            "a frame disposes the sprite"
        );
    }
}

#[test]
fn key_frames_follow_kmin_and_kmax() {
    let frames: Vec<_> = (0..9)
        .map(|i| (frame(i * 2, false, Some((i * 3, 2))), 60))
        .collect();
    let (d, ours) = check(&frames, true, 2, 4);
    assert_eq!(ours.len(), 9);
    let keys: Vec<usize> = d
        .frames()
        .iter()
        .enumerate()
        .filter(|(_, f)| f.width == W as i32 && f.height == H as i32 && f.no_blend)
        .map(|(i, _)| i)
        .collect();
    // The first frame is cut to the sprite over the transparent canvas it starts from.
    assert_eq!(keys, [4, 8]);
}

#[test]
fn a_tiny_canvas_and_rgb_input() {
    let mut enc = AnimationEncoder::new(Cursor::new(Vec::new()), 2, 2).unwrap();
    for i in 0..3u8 {
        enc.add_frame(
            &[i * 40, 0, 0, 10, 20, 30, 0, i * 50, 0, 40, 50, 60],
            ColorType::Rgb8,
            100,
        )
        .unwrap();
    }
    let data = enc.finish().unwrap().into_inner();
    let (d, ours) = decode(&data);
    assert_eq!((d.dimensions(), ours.len()), ((2, 2), 3));
    assert!(!d.has_alpha());
    assert_eq!(&ours[2].0[..4], &[80, 0, 0, 255]);
    assert_eq!(decode_libwebp(&data).len(), 3);
}

#[test]
fn rows_pushed_in_bands_code_as_whole_frames() {
    let frames: Vec<_> = (0..6)
        .map(|i| (frame(i * 3, i % 2 == 1, (i > 3).then_some((i * 4, 5))), 50))
        .collect();
    for lossless in [true, false] {
        let mut enc = AnimationEncoder::new(Cursor::new(Vec::new()), W, H).unwrap();
        let mut params = EncoderParams::default();
        params.use_lossy = !lossless;
        params.lossy_quality = 80;
        enc.set_params(params);
        let mut anim = AnimationParams::default();
        anim.loop_count = 3;
        enc.set_animation_params(anim);
        enc.set_exif_metadata(b"Exif\0\0II*\0".to_vec());
        for (i, (f, d)) in frames.iter().enumerate() {
            let band = (W * 4) as usize * (1 + i % 7);
            for rows in f.chunks(band) {
                enc.push_rows(rows, ColorType::Rgba8).unwrap();
            }
            enc.end_frame(*d).unwrap();
        }
        let banded = enc.finish().unwrap().into_inner();
        assert!(banded == encode(&frames, lossless, 9, 10), "lossless {lossless}");
    }
}

#[test]
fn a_frame_of_the_wrong_rows_fails() {
    let mut enc = AnimationEncoder::new(Cursor::new(Vec::new()), W, H).unwrap();
    let f = frame(0, false, None);
    assert!(enc.push_rows(&f[..(W * 4) as usize + 4], ColorType::Rgba8).is_err());
    enc.push_rows(&f[..(W * 4 * 3) as usize], ColorType::Rgba8).unwrap();
    assert!(enc.end_frame(100).is_err());
    enc.push_rows(&f[(W * 4 * 3) as usize..], ColorType::Rgba8).unwrap();
    assert!(enc.push_rows(&f[..(W * 4) as usize], ColorType::Rgba8).is_err());
    enc.end_frame(100).unwrap();
}
