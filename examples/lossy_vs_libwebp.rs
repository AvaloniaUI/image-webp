//! Encodes PNGs lossily with image-webp and with libwebp at the same settings and compares
//! the files byte for byte, plus size, PSNR and time.
//!
//! cargo run --release --example lossy_vs_libwebp -- [-q 75,80] [-m 4] a.png b.png ...

use std::time::Instant;

use image_webp::{ColorType, EncoderParams, WebPEncoder};
use libwebp_sys as sys;

fn libwebp_encode(
    data: &[u8],
    width: u32,
    height: u32,
    rgba: bool,
    quality: f32,
    method: i32,
) -> Vec<u8> {
    unsafe {
        let mut config = std::mem::zeroed::<sys::WebPConfig>();
        assert_ne!(
            sys::WebPConfigInitInternal(
                &mut config,
                sys::WebPPreset::WEBP_PRESET_DEFAULT,
                quality,
                sys::WEBP_ENCODER_ABI_VERSION as i32
            ),
            0
        );
        config.method = method;
        assert_ne!(sys::WebPValidateConfig(&config), 0);
        let mut pic = std::mem::zeroed::<sys::WebPPicture>();
        assert_ne!(
            sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32),
            0
        );
        pic.width = width as i32;
        pic.height = height as i32;
        pic.use_argb = 0;
        let ok = if rgba {
            sys::WebPPictureImportRGBA(&mut pic, data.as_ptr(), (width * 4) as i32)
        } else {
            sys::WebPPictureImportRGB(&mut pic, data.as_ptr(), (width * 3) as i32)
        };
        assert_ne!(ok, 0);
        let mut writer = std::mem::zeroed::<sys::WebPMemoryWriter>();
        sys::WebPMemoryWriterInit(&mut writer);
        pic.writer = Some(sys::WebPMemoryWrite);
        pic.custom_ptr = (&mut writer as *mut sys::WebPMemoryWriter).cast();
        assert_ne!(
            sys::WebPEncode(&config, &mut pic),
            0,
            "libwebp: {:?}",
            pic.error_code
        );
        let out = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
        sys::WebPMemoryWriterClear(&mut writer);
        sys::WebPPictureFree(&mut pic);
        out
    }
}

fn libwebp_decode_rgba(data: &[u8]) -> Vec<u8> {
    unsafe {
        let (mut w, mut h) = (0, 0);
        let ptr = sys::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut w, &mut h);
        assert!(!ptr.is_null());
        let out = std::slice::from_raw_parts(ptr, (w * h * 4) as usize).to_vec();
        sys::WebPFree(ptr.cast());
        out
    }
}

/// Payloads of the `VP8 ` and `ALPH` chunks.
fn chunks(data: &[u8]) -> (&[u8], &[u8]) {
    let (mut vp8, mut alph): (&[u8], &[u8]) = (&[], &[]);
    let mut i = 12;
    while i + 8 <= data.len() {
        let size = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &data[i + 8..(i + 8 + size).min(data.len())];
        match &data[i..i + 4] {
            b"VP8 " => vp8 = body,
            b"ALPH" => alph = body,
            _ => {}
        }
        i += 8 + size + (size & 1);
    }
    (vp8, alph)
}

fn psnr(a: &[u8], b: &[u8], step: usize) -> f64 {
    let mut sse = 0u64;
    let mut n = 0u64;
    for (pa, pb) in a.chunks_exact(step).zip(b.chunks_exact(4)) {
        for c in 0..3 {
            let d = i64::from(pa[c]) - i64::from(pb[c]);
            sse += (d * d) as u64;
            n += 1;
        }
    }
    if sse == 0 {
        99.
    } else {
        10. * (255. * 255. * n as f64 / sse as f64).log10()
    }
}

/// Edge cases: tiny and odd sizes, flat areas, gradients, noise, partial transparency.
fn synthetic() -> Vec<(String, u32, u32, Vec<u8>, bool)> {
    let mut seed = 0x1234_5678u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut out = Vec::new();
    for &(w, h) in &[
        (1u32, 1u32),
        (2, 3),
        (17, 9),
        (33, 47),
        (255, 129),
        (640, 31),
    ] {
        let mut noise = Vec::new();
        let mut grad = Vec::new();
        let mut flat = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let r = rnd();
                noise.extend_from_slice(&[r as u8, (r >> 8) as u8, (r >> 16) as u8]);
                grad.extend_from_slice(&[
                    (x * 255 / w) as u8,
                    (y * 255 / h) as u8,
                    ((x + y) * 127 / (w + h)) as u8,
                ]);
                flat.extend_from_slice(&[200, 30, 90]);
            }
        }
        out.push((format!("noise-{w}x{h}"), w, h, noise, false));
        out.push((format!("gradient-{w}x{h}"), w, h, grad, false));
        out.push((format!("flat-{w}x{h}"), w, h, flat, false));
        let mut alpha = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let r = rnd();
                let a = if x < w / 3 {
                    0
                } else if y < h / 2 {
                    255
                } else {
                    (x * 7 + y * 3) as u8
                };
                alpha.extend_from_slice(&[r as u8, (x * 255 / w) as u8, (y * 255 / h) as u8, a]);
            }
        }
        out.push((format!("alpha-{w}x{h}"), w, h, alpha, true));
        let opaque_rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 3) as u8, (i * 5) as u8, (i * 7) as u8, 255])
            .collect();
        out.push((format!("opaque-rgba-{w}x{h}"), w, h, opaque_rgba, true));
    }
    out
}

fn main() {
    let mut qualities = vec![80u8];
    let mut method = 4u8;
    let mut files = Vec::new();
    let mut with_synthetic = false;
    let mut dump: Option<String> = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-q" => {
                qualities = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|q| q.parse().unwrap())
                    .collect()
            }
            "-m" => method = args.next().unwrap().parse().unwrap(),
            "--synthetic" => with_synthetic = true,
            "--dump" => dump = args.next(),
            _ => files.push(arg),
        }
    }

    let mut identical = 0;
    let mut total = 0;
    let (mut ours_bytes, mut lib_bytes) = (0usize, 0usize);
    let (mut ours_time, mut lib_time) = (0f64, 0f64);
    let mut images = if with_synthetic {
        synthetic()
    } else {
        Vec::new()
    };
    for file in &files {
        let decoder = png::Decoder::new(std::fs::File::open(file).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let (w, h) = (info.width, info.height);
        let (data, rgba) = match info.color_type {
            png::ColorType::Rgb => (buf[..info.buffer_size()].to_vec(), false),
            png::ColorType::Rgba => (buf[..info.buffer_size()].to_vec(), true),
            png::ColorType::Grayscale => (
                buf[..info.buffer_size()]
                    .iter()
                    .flat_map(|&l| [l, l, l])
                    .collect(),
                false,
            ),
            png::ColorType::GrayscaleAlpha => (
                buf[..info.buffer_size()]
                    .chunks_exact(2)
                    .flat_map(|p| [p[0], p[0], p[0], p[1]])
                    .collect(),
                true,
            ),
            other => panic!("{file}: unsupported {other:?}"),
        };
        if info.bit_depth != png::BitDepth::Eight {
            eprintln!("{file}: skipping {:?}", info.bit_depth);
            continue;
        }
        images.push((file.clone(), w, h, data, rgba));
    }
    for (file, w, h, data, rgba) in &images {
        let (w, h, rgba) = (*w, *h, *rgba);
        for &q in &qualities {
            let t = Instant::now();
            let lib = libwebp_encode(data, w, h, rgba, f32::from(q), i32::from(method));
            let lib_t = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let mut ours = Vec::new();
            let mut encoder = WebPEncoder::new(&mut ours);
            let mut params = EncoderParams::default();
            params.use_lossy = true;
            params.lossy_quality = q;
            params.lossy_method = method;
            encoder.set_params(params);
            encoder
                .encode(
                    data,
                    w,
                    h,
                    if rgba {
                        ColorType::Rgba8
                    } else {
                        ColorType::Rgb8
                    },
                )
                .unwrap();
            let ours_t = t.elapsed().as_secs_f64();

            let step = if rgba { 4 } else { 3 };
            let p_lib = psnr(data, &libwebp_decode_rgba(&lib), step);
            let p_ours = psnr(data, &libwebp_decode_rgba(&ours), step);
            let same = lib == ours;
            let (lib_vp8, lib_alph) = chunks(&lib);
            let (ours_vp8, ours_alph) = chunks(&ours);
            let same_vp8 = lib_vp8 == ours_vp8;
            let first_diff = lib.iter().zip(&ours).position(|(a, b)| a != b);
            if let (Some(dir), false) = (&dump, same) {
                let name = std::path::Path::new(file)
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned();
                std::fs::write(format!("{dir}/{name}-q{q}-m{method}-libwebp.webp"), &lib).unwrap();
                std::fs::write(format!("{dir}/{name}-q{q}-m{method}-ours.webp"), &ours).unwrap();
            }
            total += 1;
            identical +=
                usize::from(same || (same_vp8 && !lib_alph.is_empty() && !ours_alph.is_empty()));
            ours_bytes += ours.len();
            lib_bytes += lib.len();
            ours_time += ours_t;
            lib_time += lib_t;
            println!(
                "{file} {w}x{h} q{q} m{method}: {} libwebp {} B {:.3} dB {:.1} ms | ours {} B {:.3} dB {:.1} ms{}",
                if same {
                    "IDENTICAL".to_string()
                } else if same_vp8 {
                    format!("VP8-IDENTICAL (ALPH libwebp {} B, ours {} B)", lib_alph.len(), ours_alph.len())
                } else {
                    "DIFFERENT".to_string()
                },
                lib.len(),
                p_lib,
                lib_t * 1e3,
                ours.len(),
                p_ours,
                ours_t * 1e3,
                first_diff.filter(|_| !same).map(|d| format!(" first diff @{d}")).unwrap_or_default()
            );
        }
    }
    println!(
        "identical {identical}/{total}; bytes ours/libwebp {:.4}; time ours/libwebp {:.2}",
        ours_bytes as f64 / lib_bytes.max(1) as f64,
        ours_time / lib_time.max(1e-9)
    );
    if identical != total {
        std::process::exit(1);
    }
}
