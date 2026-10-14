//! Times image-webp's lossless encoder, or its alpha plane encoder, on a PAM image.
//!
//! cargo run --release --example lossless_bench -- [--alpha | --plane] [--libwebp] [-q 75] [-m 4] [-n 3] image.pam
//!
//! `--plane` times the unfiltered alpha plane stream; with `--libwebp`, libwebp's lossless
//! encoder on the same green-only image at the alpha encoder's settings.

use std::time::Instant;

use image_webp::{ColorType, EncoderParams, WebPEncoder};
use libwebp_sys as sys;

fn libwebp_lossless(
    argb: &[u32],
    width: u32,
    height: u32,
    quality: f32,
    method: i32,
    exact: bool,
) -> usize {
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
        config.lossless = 1;
        config.method = method;
        config.exact = i32::from(exact);
        let mut pic = std::mem::zeroed::<sys::WebPPicture>();
        assert_ne!(
            sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32),
            0
        );
        pic.width = width as i32;
        pic.height = height as i32;
        pic.use_argb = 1;
        assert_ne!(sys::WebPPictureAlloc(&mut pic), 0);
        std::ptr::copy_nonoverlapping(argb.as_ptr(), pic.argb, argb.len());
        let mut writer = std::mem::zeroed::<sys::WebPMemoryWriter>();
        sys::WebPMemoryWriterInit(&mut writer);
        pic.writer = Some(sys::WebPMemoryWrite);
        pic.custom_ptr = (&mut writer as *mut sys::WebPMemoryWriter).cast();
        assert_ne!(sys::WebPEncode(&config, &mut pic), 0);
        let size = writer.size;
        sys::WebPMemoryWriterClear(&mut writer);
        sys::WebPPictureFree(&mut pic);
        size
    }
}

fn main() {
    let (mut q, mut m, mut n, mut alpha, mut file) = (75u8, 4u8, 1, false, String::new());
    let (mut plane, mut lib) = (false, false);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-q" => q = args.next().unwrap().parse().unwrap(),
            "-m" => m = args.next().unwrap().parse().unwrap(),
            "-n" => n = args.next().unwrap().parse().unwrap(),
            "--alpha" => alpha = true,
            "--plane" => (alpha, plane) = (true, true),
            "--libwebp" => lib = true,
            _ => file = arg,
        }
    }
    let data = std::fs::read(&file).unwrap();
    let end = data.windows(7).position(|w| w == b"ENDHDR\n").unwrap() + 7;
    let text = std::str::from_utf8(&data[..end]).unwrap();
    let get = |k: &str| {
        text.lines().find(|l| l.starts_with(k)).unwrap()[k.len()..]
            .trim()
            .parse::<u32>()
            .unwrap()
    };
    let (w, h, depth) = (get("WIDTH"), get("HEIGHT"), get("DEPTH"));
    let px = &data[end..end + (w * h * depth) as usize];
    let (px, color): (Vec<u8>, ColorType) = if alpha {
        (
            px.chunks_exact(depth as usize)
                .map(|p| p[depth as usize - 1])
                .collect(),
            ColorType::L8,
        )
    } else {
        (
            px.to_vec(),
            if depth == 4 {
                ColorType::Rgba8
            } else {
                ColorType::Rgb8
            },
        )
    };
    for _ in 0..n {
        let t = Instant::now();
        let size = if plane && lib {
            let argb: Vec<u32> = px.iter().map(|&a| u32::from(a) << 8).collect();
            let quality = if m == 6 { 100. } else { 8. * f32::from(m) };
            libwebp_lossless(&argb, w, h, quality, i32::from(m), true)
        } else if plane {
            image_webp::bench_encode_alpha_plane(&px, w, h, m)
        } else if lib {
            let argb: Vec<u32> = px
                .chunks_exact(depth as usize)
                .map(|p| {
                    let a = if depth == 4 { u32::from(p[3]) } else { 255 };
                    a << 24 | u32::from(p[0]) << 16 | u32::from(p[1]) << 8 | u32::from(p[2])
                })
                .collect();
            libwebp_lossless(&argb, w, h, f32::from(q), i32::from(m), false)
        } else if alpha {
            image_webp::bench_encode_alpha(&px, w, h, m)
        } else {
            let mut out = Vec::new();
            let mut enc = WebPEncoder::new(&mut out);
            let mut p = EncoderParams::default();
            p.lossless_quality = q;
            p.lossless_method = m;
            enc.set_params(p);
            enc.encode(&px, w, h, color).unwrap();
            out.len()
        };
        println!(
            "{} {w}x{h} m{m}: {size} B {:.1} ms",
            if lib { "libwebp" } else { "ours" },
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
