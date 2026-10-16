//! Near-lossless encodes against libwebp's: the same bytes for every level and method.

use image_webp::{ColorType, EncoderParams, WebPEncoder};
use libwebp_sys as webp;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn libwebp(w: usize, h: usize, rgba: &[u8], alpha: bool, level: i32, method: i32) -> Vec<u8> {
    let mut config =
        webp::WebPConfig::new_with_preset(webp::WebPPreset::WEBP_PRESET_DEFAULT, 75.0).unwrap();
    config.lossless = 1;
    config.exact = 0;
    config.method = method;
    config.near_lossless = level;
    let mut picture = webp::WebPPicture::new().unwrap();
    picture.use_argb = 1;
    picture.width = w as i32;
    picture.height = h as i32;
    let mut writer = unsafe { std::mem::zeroed::<webp::WebPMemoryWriter>() };
    unsafe {
        webp::WebPMemoryWriterInit(&mut writer);
        let ok = if alpha {
            webp::WebPPictureImportRGBA(&mut picture, rgba.as_ptr(), (w * 4) as i32)
        } else {
            webp::WebPPictureImportRGB(&mut picture, rgba.as_ptr(), (w * 3) as i32)
        };
        assert!(ok != 0);
        picture.writer = Some(webp::WebPMemoryWrite);
        picture.custom_ptr = &mut writer as *mut _ as *mut std::ffi::c_void;
        assert!(webp::WebPEncode(&config, &mut picture) != 0);
        let v = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
        webp::WebPMemoryWriterClear(&mut writer);
        webp::WebPPictureFree(&mut picture);
        v
    }
}

fn ours(w: usize, h: usize, data: &[u8], alpha: bool, level: i32, method: i32) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = WebPEncoder::new(&mut out);
    let mut params = EncoderParams::default();
    params.lossless_method = method as u8;
    params.lossless_quality = 75;
    params.near_lossless = level as u8;
    encoder.set_params(params);
    let color = if alpha {
        ColorType::Rgba8
    } else {
        ColorType::Rgb8
    };
    encoder.encode(data, w as u32, h as u32, color).unwrap();
    out
}

fn decode(data: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h * 4];
    let p = unsafe {
        webp::WebPDecodeRGBAInto(
            data.as_ptr(),
            data.len(),
            out.as_mut_ptr(),
            out.len(),
            (w * 4) as i32,
        )
    };
    assert!(!p.is_null());
    out
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mse: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum::<f64>()
        / a.len() as f64;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 / mse).log10()
    }
}

fn picture(rng: &mut StdRng, w: usize, h: usize, kind: u8, alpha: bool) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let n: u8 = rng.gen_range(0..6);
            let smooth = [
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) * 128 / (w + h)) as u8,
            ];
            let [r, g, b] = match kind {
                0 => smooth,
                1 => smooth.map(|c| c.wrapping_add(n)),
                2 if (x / 9 + y / 7) % 2 == 0 => [240, 10, 30],
                2 => smooth,
                _ => [rng.gen(), rng.gen(), rng.gen()],
            };
            let a = match (alpha, kind) {
                (false, _) => 255,
                (true, 2) if (x / 11 + y / 13) % 5 == 0 => 0,
                (true, 2) => 255,
                (true, _) => ((x * 255 / w) as u8).wrapping_add(n),
            };
            if alpha {
                vec![r, g, b, a]
            } else {
                vec![r, g, b]
            }
        })
        .collect()
}

#[test]
fn near_lossless_as_libwebp() {
    let mut rng = StdRng::seed_from_u64(7);
    let mut differences = 0;
    for &(w, h) in &[(96usize, 80usize), (63, 70), (40, 30), (130, 65)] {
        for kind in 0..4u8 {
            for alpha in [false, true] {
                let data = picture(&mut rng, w, h, kind, alpha);
                let source = if alpha {
                    data.clone()
                } else {
                    data.chunks_exact(3)
                        .flat_map(|p| [p[0], p[1], p[2], 255])
                        .collect()
                };
                for level in [0, 20, 40, 60, 80, 100] {
                    for method in [0, 4, 6] {
                        let theirs = libwebp(w, h, &data, alpha, level, method);
                        let mine = ours(w, h, &data, alpha, level, method);
                        if theirs != mine {
                            differences += 1;
                            let (a, b) = (decode(&theirs, w, h), decode(&mine, w, h));
                            println!(
                                "{w}x{h} kind {kind} alpha {alpha} level {level} method {method}: libwebp {} B {:.2} dB, ours {} B {:.2} dB",
                                theirs.len(),
                                psnr(&source, &a),
                                mine.len(),
                                psnr(&source, &b)
                            );
                        }
                    }
                }
            }
        }
    }
    assert_eq!(differences, 0);
}
