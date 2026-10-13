//! Encodes a large image built by tiling an RGB or RGBA PNG, with image-webp or libwebp, for timing and
//! peak-memory measurement (`/usr/bin/time -v` / `-l`).
//!
//! cargo run --release --example lossy_bench -- ours|ours-rows|libwebp in.png [WxH] [quality] [method]
//!
//! `ours-rows` generates the rows as the encoder asks for them, without the whole image.

use std::time::Instant;

use image_webp::{ColorType, EncoderParams, WebPEncoder};
use libwebp_sys as sys;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let which = args[1].as_str();
    let decoder = png::Decoder::new(std::fs::File::open(&args[2]).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bpp = match info.color_type {
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        c => panic!("{c:?}"),
    };
    let color = if bpp == 4 {
        ColorType::Rgba8
    } else {
        ColorType::Rgb8
    };
    let (sw, sh) = (info.width as usize, info.height as usize);
    let (w, h) = args
        .get(3)
        .map(|s| {
            s.split_once('x')
                .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
                .unwrap()
        })
        .unwrap_or((6000usize, 4000usize));
    let quality: u8 = args.get(4).map_or(80, |q| q.parse().unwrap());
    let method: u8 = args.get(5).map_or(4, |m| m.parse().unwrap());

    // Mirror-tile so that tile edges stay continuous.
    let tile_row = |y: usize, row: &mut [u8]| {
        let ty = y / sh;
        let sy = if ty % 2 == 0 { y % sh } else { sh - 1 - y % sh };
        for x in 0..w {
            let tx = x / sw;
            let sx = if tx % 2 == 0 { x % sw } else { sw - 1 - x % sw };
            row[x * bpp..][..bpp].copy_from_slice(&buf[(sy * sw + sx) * bpp..][..bpp]);
        }
    };
    let rgb = if which == "ours-rows" {
        Vec::new()
    } else {
        let mut rgb = vec![0u8; w * h * bpp];
        for (y, row) in rgb.chunks_exact_mut(w * bpp).enumerate() {
            tile_row(y, row);
        }
        rgb
    };

    let t = Instant::now();
    let size = match which {
        "ours" => {
            let mut out = Vec::new();
            let mut encoder = WebPEncoder::new(&mut out);
            let mut params = EncoderParams::default();
            params.use_lossy = true;
            params.lossy_quality = quality;
            params.lossy_method = method;
            encoder.set_params(params);
            encoder.encode(&rgb, w as u32, h as u32, color).unwrap();
            out.len()
        }
        "ours-rows" => {
            let mut out = Vec::new();
            let mut encoder = WebPEncoder::new(&mut out);
            let mut params = EncoderParams::default();
            params.use_lossy = true;
            params.lossy_quality = quality;
            params.lossy_method = method;
            encoder.set_params(params);
            encoder
                .encode_rows(w as u32, h as u32, color, |y, row| {
                    tile_row(y as usize, row)
                })
                .unwrap();
            out.len()
        }
        "libwebp" => unsafe {
            let mut config = std::mem::zeroed::<sys::WebPConfig>();
            sys::WebPConfigInitInternal(
                &mut config,
                sys::WebPPreset::WEBP_PRESET_DEFAULT,
                f32::from(quality),
                sys::WEBP_ENCODER_ABI_VERSION as i32,
            );
            config.method = i32::from(method);
            let mut pic = std::mem::zeroed::<sys::WebPPicture>();
            sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32);
            pic.width = w as i32;
            pic.height = h as i32;
            if bpp == 4 {
                sys::WebPPictureImportRGBA(&mut pic, rgb.as_ptr(), (w * 4) as i32);
            } else {
                sys::WebPPictureImportRGB(&mut pic, rgb.as_ptr(), (w * 3) as i32);
            }
            let mut writer = std::mem::zeroed::<sys::WebPMemoryWriter>();
            sys::WebPMemoryWriterInit(&mut writer);
            pic.writer = Some(sys::WebPMemoryWrite);
            pic.custom_ptr = (&mut writer as *mut sys::WebPMemoryWriter).cast();
            assert_ne!(sys::WebPEncode(&config, &mut pic), 0);
            let size = writer.size;
            sys::WebPMemoryWriterClear(&mut writer);
            sys::WebPPictureFree(&mut pic);
            size
        },
        _ => panic!("ours, ours-rows or libwebp"),
    };
    println!(
        "{which} {w}x{h} q{quality} m{method}: {size} B in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
}
