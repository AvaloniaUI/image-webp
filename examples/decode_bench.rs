//! Times decoding a WebP file with read_image, with read_image_rows or with libwebp.
//!
//! cargo run --release --example decode_bench -- [-n 3] [--rows | --libwebp | --libwebp-threads] image.webp
//! cargo run --release --example decode_bench -- --encode out.webp image.pam
//!
//! `--encode` writes a PAM (RGB or RGBA) as libwebp's lossy Q80 method 4 WebP.

use std::time::Instant;

use libwebp_sys as sys;

fn read_pam(file: &str) -> (u32, u32, Vec<u8>, bool) {
    let data = std::fs::read(file).unwrap();
    let end = data.windows(7).position(|w| w == b"ENDHDR\n").unwrap() + 7;
    let text = std::str::from_utf8(&data[..end]).unwrap();
    let get = |k: &str| {
        text.lines().find(|l| l.starts_with(k)).unwrap()[k.len()..]
            .trim()
            .parse::<u32>()
            .unwrap()
    };
    let (w, h, depth) = (get("WIDTH"), get("HEIGHT"), get("DEPTH"));
    let n = (w * h * depth) as usize;
    (w, h, data[end..end + n].to_vec(), depth == 4)
}

fn libwebp_encode(data: &[u8], width: u32, height: u32, rgba: bool) -> Vec<u8> {
    unsafe {
        let mut out = std::ptr::null_mut();
        let size = if rgba {
            sys::WebPEncodeRGBA(
                data.as_ptr(),
                width as i32,
                height as i32,
                (width * 4) as i32,
                80.0,
                &mut out,
            )
        } else {
            sys::WebPEncodeRGB(
                data.as_ptr(),
                width as i32,
                height as i32,
                (width * 3) as i32,
                80.0,
                &mut out,
            )
        };
        assert_ne!(size, 0);
        let webp = std::slice::from_raw_parts(out, size).to_vec();
        sys::WebPFree(out.cast());
        webp
    }
}

fn libwebp_decode(data: &[u8], threads: bool) -> u64 {
    unsafe {
        let mut config = std::mem::zeroed::<sys::WebPDecoderConfig>();
        assert_ne!(
            sys::WebPInitDecoderConfigInternal(&mut config, sys::WEBP_DECODER_ABI_VERSION as i32),
            0
        );
        assert_eq!(
            sys::WebPGetFeaturesInternal(
                data.as_ptr(),
                data.len(),
                &mut config.input,
                sys::WEBP_DECODER_ABI_VERSION as i32
            ),
            sys::VP8StatusCode::VP8_STATUS_OK
        );
        config.output.colorspace = if config.input.has_alpha != 0 {
            sys::WEBP_CSP_MODE::MODE_RGBA
        } else {
            sys::WEBP_CSP_MODE::MODE_RGB
        };
        config.options.use_threads = i32::from(threads);
        assert_eq!(
            sys::WebPDecode(data.as_ptr(), data.len(), &mut config),
            sys::VP8StatusCode::VP8_STATUS_OK
        );
        let rgba = &config.output.u.RGBA;
        let sum = u64::from(*rgba.rgba.add(rgba.size / 2));
        sys::WebPFreeDecBuffer(&mut config.output);
        sum
    }
}

fn main() {
    let (mut n, mut mode, mut file, mut encode) = (1, "whole", String::new(), None);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-n" => n = args.next().unwrap().parse().unwrap(),
            "--rows" => mode = "rows",
            "--libwebp" => mode = "libwebp",
            "--libwebp-threads" => mode = "libwebp-threads",
            "--check" => mode = "check",
            "--encode" => encode = args.next(),
            _ => file = arg,
        }
    }
    if let Some(out) = encode {
        let (w, h, pixels, rgba) = read_pam(&file);
        std::fs::write(out, libwebp_encode(&pixels, w, h, rgba)).unwrap();
        return;
    }
    let data = std::fs::read(&file).unwrap();
    if mode == "check" {
        let (mut w, mut h) = (0, 0);
        let lib = unsafe {
            let mut features = std::mem::zeroed::<sys::WebPBitstreamFeatures>();
            sys::WebPGetFeatures(data.as_ptr(), data.len(), &mut features);
            let ptr = if features.has_alpha != 0 {
                sys::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut w, &mut h)
            } else {
                sys::WebPDecodeRGB(data.as_ptr(), data.len(), &mut w, &mut h)
            };
            let len = (w * h) as usize * if features.has_alpha != 0 { 4 } else { 3 };
            let out = std::slice::from_raw_parts(ptr, len).to_vec();
            sys::WebPFree(ptr.cast());
            out
        };
        let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
        let mut rows = Vec::new();
        decoder
            .read_image_rows(|_, band| rows.extend_from_slice(band))
            .unwrap();
        let mut decoder = image_webp::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
        let mut whole = vec![0; decoder.output_buffer_size().unwrap()];
        decoder.read_image(&mut whole).unwrap();
        let differ = |px: &[u8]| px.iter().zip(&lib).filter(|(a, b)| a != b).count();
        let (diff, diff_whole) = (differ(&rows), differ(&whole));
        println!(
            "check {file}: {} bytes, {diff} differ from libwebp ({diff_whole} with read_image)",
            lib.len()
        );
        assert!(diff == 0 && rows.len() == lib.len());
        assert!(diff_whole == 0 && whole.len() == lib.len());
        return;
    }
    for _ in 0..n {
        let t = Instant::now();
        let sum = match mode {
            "libwebp" => libwebp_decode(&data, false),
            "libwebp-threads" => libwebp_decode(&data, true),
            _ => {
                let mut decoder =
                    image_webp::WebPDecoder::new(std::io::Cursor::new(&data)).unwrap();
                let mut sum = 0u64;
                if mode == "rows" {
                    decoder
                        .read_image_rows(|_, band| sum += u64::from(band[band.len() / 2]))
                        .unwrap();
                } else {
                    let mut buf = vec![0; decoder.output_buffer_size().unwrap()];
                    decoder.read_image(&mut buf).unwrap();
                    sum = u64::from(buf[buf.len() / 2]);
                }
                sum
            }
        };
        println!("{mode} {:.1} ms ({sum})", t.elapsed().as_secs_f64() * 1e3);
    }
}
