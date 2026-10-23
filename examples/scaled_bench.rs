//! Times scaled decodes of a lossy WebP at 1/2, 1/4 and 1/8 (whole and one pixel off), and of a
//! lossless WebP made from it, printing the fastest run, libwebp's fastest single-threaded run of
//! the same decode and a checksum of the pixels (flagged where they differ from libwebp's).
//!
//! cargo run --release --example scaled_bench -- image.webp [runs] [case]
//!
//! With `SCALED_BENCH_OURS` set it skips libwebp's decodes, so a profile holds only ours.

use std::io::Cursor;
use std::time::Instant;

use image_webp::WebPDecoder;
use libwebp_sys as sys;

fn decode(data: &[u8], size: Option<(u32, u32)>, alpha: bool) -> (Vec<u8>, u32, u32) {
    let mut d = WebPDecoder::new(Cursor::new(data)).unwrap();
    let (w, h) = size.unwrap_or(d.dimensions());
    let mut out = Vec::with_capacity((w * h * if alpha { 4 } else { 3 }) as usize);
    let mut frame = d.frame(0, size, alpha).unwrap();
    while let Some((_, rows)) = frame.next_band().unwrap() {
        out.extend_from_slice(rows);
    }
    (out, w, h)
}

fn decode_libwebp(data: &[u8], size: (u32, u32), alpha: bool) -> Vec<u8> {
    unsafe {
        let mut config = std::mem::zeroed::<sys::WebPDecoderConfig>();
        assert!(sys::WebPInitDecoderConfig(&mut config));
        let bpp = if alpha { 4 } else { 3 };
        let mut out = vec![0u8; (size.0 * size.1) as usize * bpp];
        config.output.colorspace = if alpha {
            sys::WEBP_CSP_MODE::MODE_RGBA
        } else {
            sys::WEBP_CSP_MODE::MODE_RGB
        };
        config.output.is_external_memory = 1;
        config.output.u.RGBA = sys::WebPRGBABuffer {
            rgba: out.as_mut_ptr(),
            stride: (size.0 as usize * bpp) as i32,
            size: out.len(),
        };
        config.options.use_scaling = 1;
        config.options.scaled_width = size.0 as i32;
        config.options.scaled_height = size.1 as i32;
        let status = sys::WebPDecode(data.as_ptr(), data.len(), &mut config);
        assert_eq!(status, sys::VP8StatusCode::VP8_STATUS_OK);
        out
    }
}

fn lossless(rgba: &[u8], w: u32, h: u32) -> Vec<u8> {
    unsafe {
        let mut out = std::ptr::null_mut();
        let size = sys::WebPEncodeLosslessRGBA(
            rgba.as_ptr(),
            w as i32,
            h as i32,
            (w * 4) as i32,
            &mut out,
        );
        assert_ne!(size, 0);
        let webp = std::slice::from_raw_parts(out, size).to_vec();
        sys::WebPFree(out.cast());
        webp
    }
}

fn fnv(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let runs: usize = args.get(2).map_or(5, |r| r.parse().unwrap());
    let (w, h) = WebPDecoder::new(Cursor::new(&data)).unwrap().dimensions();
    let ours_only = std::env::var_os("SCALED_BENCH_OURS").is_some();
    let (small, sw, sh) = match args.get(3) {
        Some(only) if !only.starts_with("lossless") => (vec![0; 4], 1, 1),
        _ => decode(&data, Some((w / 3, h / 3)), true),
    };
    let lossless = lossless(&small, sw, sh);
    let mut cases = vec![];
    for k in [2, 4, 8] {
        cases.push((format!("lossy 1/{k}"), &data, (w / k, h / k), false));
        cases.push((
            format!("lossy 1/{k} +1px"),
            &data,
            (w / k + 1, h / k + 1),
            false,
        ));
    }
    cases.push(("lossy 1/2 rgba".into(), &data, (w / 2, h / 2), true));
    for k in [2, 4] {
        cases.push((format!("lossless 1/{k}"), &lossless, (sw / k, sh / k), true));
        cases.push((
            format!("lossless 1/{k} +1px"),
            &lossless,
            (sw / k + 1, sh / k + 1),
            true,
        ));
    }
    for (name, file, size, alpha) in cases {
        if args.get(3).is_some_and(|only| name != *only) {
            continue;
        }
        let (mut best, mut theirs) = (f64::MAX, f64::MAX);
        let mut sum = 0;
        let mut same = true;
        for run in 0..runs {
            let t = Instant::now();
            let (out, _, _) = decode(file, Some(size), alpha);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
            if run + 1 == runs {
                sum = fnv(&out);
            }
            if ours_only {
                continue;
            }
            let t = Instant::now();
            let reference = decode_libwebp(file, size, alpha);
            theirs = theirs.min(t.elapsed().as_secs_f64() * 1e3);
            same &= out == reference;
        }
        println!(
            "{name:<20} {:>5}x{:<5} {best:8.2} ms  libwebp {theirs:8.2} ms  {sum:016x}{}",
            size.0,
            size.1,
            if same { "" } else { "  DIFFERS" }
        );
    }
}
