//! Encodes images losslessly, and the alpha plane of lossy images, with image-webp and with
//! libwebp at the same settings and compares the VP8L and ALPH chunks byte for byte, plus
//! size and time.
//!
//! cargo run --release --example lossless_vs_libwebp -- [-q 75] [-m 4] [--alpha] [--synthetic] a.png b.pam ...

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
    lossless: bool,
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
        config.lossless = i32::from(lossless);
        assert_ne!(sys::WebPValidateConfig(&config), 0);
        let mut pic = std::mem::zeroed::<sys::WebPPicture>();
        assert_ne!(
            sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32),
            0
        );
        pic.width = width as i32;
        pic.height = height as i32;
        pic.use_argb = i32::from(lossless);
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

fn chunk<'a>(data: &'a [u8], name: &[u8]) -> &'a [u8] {
    let mut i = 12;
    while i + 8 <= data.len() {
        let size = u32::from_le_bytes(data[i + 4..i + 8].try_into().unwrap()) as usize;
        if &data[i..i + 4] == name {
            return &data[i + 8..(i + 8 + size).min(data.len())];
        }
        i += 8 + size + (size & 1);
    }
    &[]
}

/// Edge cases: tiny and odd sizes, flat areas, gradients, noise, few colours, transparency.
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
        let (mut noise, mut grad, mut flat, mut few, mut alpha, mut mask) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
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
                let c = [
                    [0u8, 0, 0],
                    [255, 255, 255],
                    [200, 30, 90],
                    [10, 100, 250],
                    [90, 90, 90],
                ][((x / 3 + y / 5 + (r & 1)) % 5) as usize];
                few.extend_from_slice(&c);
                let a = if x < w / 3 {
                    0
                } else if y < h / 2 {
                    255
                } else {
                    (x * 7 + y * 3) as u8
                };
                alpha.extend_from_slice(&[r as u8, (x * 255 / w) as u8, (y * 255 / h) as u8, a]);
                let d = ((x as f64 - w as f64 / 2.).hypot(y as f64 - h as f64 / 2.)
                    / (w.min(h) as f64 / 2.))
                    .min(2.);
                let m = ((1.1 - d) * 510.).clamp(0., 255.) as u8;
                mask.extend_from_slice(&[(x * 3) as u8, (y * 5) as u8, 128, m]);
            }
        }
        out.push((format!("noise-{w}x{h}"), w, h, noise, false));
        out.push((format!("gradient-{w}x{h}"), w, h, grad, false));
        out.push((format!("flat-{w}x{h}"), w, h, flat, false));
        out.push((format!("few-colors-{w}x{h}"), w, h, few, false));
        out.push((format!("alpha-{w}x{h}"), w, h, alpha, true));
        out.push((format!("round-mask-{w}x{h}"), w, h, mask, true));
    }
    out
}

fn read_image(file: &str) -> Option<(u32, u32, Vec<u8>, bool)> {
    if file.ends_with(".pam") || file.ends_with(".ppm") {
        let data = std::fs::read(file).unwrap();
        let end = data.windows(7).position(|w| w == b"ENDHDR\n")? + 7;
        let text = std::str::from_utf8(&data[..end]).unwrap();
        let get = |k: &str| {
            text.lines().find(|l| l.starts_with(k)).unwrap()[k.len()..]
                .trim()
                .parse::<u32>()
                .unwrap()
        };
        let (w, h, depth) = (get("WIDTH"), get("HEIGHT"), get("DEPTH"));
        let n = (w * h * depth) as usize;
        return Some((w, h, data[end..end + n].to_vec(), depth == 4));
    }
    let decoder = png::Decoder::new(std::fs::File::open(file).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    if info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    let buf = &buf[..info.buffer_size()];
    Some(match info.color_type {
        png::ColorType::Rgb => (info.width, info.height, buf.to_vec(), false),
        png::ColorType::Rgba => (info.width, info.height, buf.to_vec(), true),
        png::ColorType::Grayscale => (
            info.width,
            info.height,
            buf.iter().flat_map(|&l| [l, l, l]).collect(),
            false,
        ),
        png::ColorType::GrayscaleAlpha => (
            info.width,
            info.height,
            buf.chunks_exact(2)
                .flat_map(|p| [p[0], p[0], p[0], p[1]])
                .collect(),
            true,
        ),
        _ => return None,
    })
}

fn main() {
    let mut qualities = vec![75u8];
    let mut methods = vec![4u8];
    let mut files = Vec::new();
    let mut with_synthetic = false;
    let mut alpha_mode = false;
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
            "-m" => {
                methods = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|q| q.parse().unwrap())
                    .collect()
            }
            "--synthetic" => with_synthetic = true,
            "--alpha" => alpha_mode = true,
            _ => files.push(arg),
        }
    }
    let mut images = if with_synthetic {
        synthetic()
    } else {
        Vec::new()
    };
    for file in &files {
        match read_image(file) {
            Some((w, h, data, rgba)) => images.push((file.clone(), w, h, data, rgba)),
            None => eprintln!("{file}: skipped"),
        }
    }
    let (mut identical, mut total) = (0, 0);
    let (mut ours_bytes, mut lib_bytes) = (0usize, 0usize);
    let (mut ours_time, mut lib_time) = (0f64, 0f64);
    for (file, w, h, data, rgba) in &images {
        let (w, h, rgba) = (*w, *h, *rgba);
        if alpha_mode && !rgba {
            continue;
        }
        for &m in &methods {
            for &q in &qualities {
                let t = Instant::now();
                let lib = libwebp_encode(data, w, h, rgba, f32::from(q), i32::from(m), !alpha_mode);
                let lib_t = t.elapsed().as_secs_f64();
                let t = Instant::now();
                let mut ours = Vec::new();
                let mut encoder = WebPEncoder::new(&mut ours);
                let mut params = EncoderParams::default();
                params.use_lossy = alpha_mode;
                params.lossy_quality = q;
                params.lossy_method = m;
                params.lossless_quality = q;
                params.lossless_method = m;
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
                let name: &[u8] = if alpha_mode { b"ALPH" } else { b"VP8L" };
                let (lc, oc) = (chunk(&lib, name), chunk(&ours, name));
                let same = lc == oc;
                if !alpha_mode {
                    let src: Vec<u8> = if rgba {
                        data.clone()
                    } else {
                        data.chunks_exact(3)
                            .flat_map(|p| [p[0], p[1], p[2], 255])
                            .collect()
                    };
                    let dec = libwebp_decode_rgba(&ours);
                    let lossless_ok = src
                        .chunks_exact(4)
                        .zip(dec.chunks_exact(4))
                        .all(|(s, d)| s == d || (s[3] == 0 && d[3] == 0));
                    assert!(lossless_ok, "{file}: ours does not decode to the input");
                }
                total += 1;
                identical += usize::from(same);
                ours_bytes += oc.len();
                lib_bytes += lc.len();
                ours_time += ours_t;
                lib_time += lib_t;
                let first_diff = lc.iter().zip(oc).position(|(a, b)| a != b);
                println!(
                    "{file} {w}x{h} q{q} m{m} {}: {} libwebp {} B {:.1} ms | ours {} B {:.1} ms{}",
                    if alpha_mode { "ALPH" } else { "VP8L" },
                    if same { "IDENTICAL" } else { "DIFFERENT" },
                    lc.len(),
                    lib_t * 1e3,
                    oc.len(),
                    ours_t * 1e3,
                    first_diff
                        .filter(|_| !same)
                        .map(|d| format!(" first diff @{d}"))
                        .unwrap_or_default()
                );
            }
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
