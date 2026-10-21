//! `FrameDecoder` against libwebp's `WebPDecode` into RGB and RGBA buffers, at full size and
//! scaled, over the fixtures, libwebp-encoded pictures (one and eight token partitions, larger
//! than a partition window) and frames, and their cut and byte-changed copies: the same pixels,
//! and an error exactly where libwebp fails, decoding from a slice and from a reader.

use image_webp::{headers, FrameDecoder, WebPDecoder, ANIMATION_FLAG};
use libwebp_sys as webp;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::io::{BufReader, Cursor};

fn libwebp(data: &[u8], size: Option<(u32, u32)>, alpha: bool) -> Option<Vec<u8>> {
    let mut config = unsafe { std::mem::zeroed::<webp::WebPDecoderConfig>() };
    assert!(unsafe { webp::WebPInitDecoderConfig(&mut config) });
    if unsafe { webp::WebPGetFeatures(data.as_ptr(), data.len(), &mut config.input) } != webp::VP8StatusCode::VP8_STATUS_OK {
        return None;
    }
    let (w, h) = size.unwrap_or((config.input.width as u32, config.input.height as u32));
    let bpp = if alpha { 4 } else { 3 };
    let mut out = vec![0u8; w as usize * h as usize * bpp];
    config.output.colorspace = if alpha { webp::WEBP_CSP_MODE::MODE_RGBA } else { webp::WEBP_CSP_MODE::MODE_RGB };
    config.output.is_external_memory = 1;
    config.output.u.RGBA = webp::WebPRGBABuffer { rgba: out.as_mut_ptr(), stride: (w as usize * bpp) as i32, size: out.len() };
    if let Some((w, h)) = size {
        config.options.use_scaling = 1;
        config.options.scaled_width = w as i32;
        config.options.scaled_height = h as i32;
    }
    let status = unsafe { webp::WebPDecode(data.as_ptr(), data.len(), &mut config) };
    (status == webp::VP8StatusCode::VP8_STATUS_OK).then_some(out)
}

fn ours(data: &[u8], size: Option<(u32, u32)>, alpha: bool) -> Option<Vec<u8>> {
    let h = headers::features(data).ok()?;
    if h.has_animation {
        return None;
    }
    let h = headers::decoding(data).ok()?;
    let from_slice = bands(FrameDecoder::new(data, &h, size, alpha));
    let whole = whole(FrameDecoder::new(data, &h, size, alpha));
    assert_eq!(from_slice, whole, "band and whole-frame decodes differ");
    // The same bytes behind others, read in small pieces.
    let prefixed = [&[0x55; 13][..], data].concat();
    let mut r = BufReader::with_capacity(61, Cursor::new(prefixed));
    let h = headers::read_decoding(&mut r, 13, data.len() as u64).ok()?;
    let from_reader = bands(FrameDecoder::from_reader(r, 13, data.len() as u64, &h, size, alpha));
    assert_eq!(from_slice, from_reader, "slice and reader decodes differ");
    from_slice
}

fn bands(d: Result<FrameDecoder, image_webp::DecodingError>) -> Option<Vec<u8>> {
    let mut d = d.ok()?;
    let mut out = Vec::new();
    loop {
        match d.next_band() {
            Ok(Some((_, rows))) => out.extend_from_slice(rows),
            Ok(None) => return Some(out),
            Err(_) => return None,
        }
    }
}

/// The frame decoded with `read_into`, straight into one buffer.
fn whole(d: Result<FrameDecoder, image_webp::DecodingError>) -> Option<Vec<u8>> {
    let mut d = d.ok()?;
    let (w, h) = d.dimensions();
    let mut out = vec![0; w as usize * h as usize * d.bytes_per_pixel()];
    d.read_into(&mut out).ok()?;
    Some(out)
}

fn sizes(data: &[u8]) -> Vec<Option<(u32, u32)>> {
    let (mut w, mut h) = (0, 0);
    if unsafe { webp::WebPGetInfo(data.as_ptr(), data.len(), &mut w, &mut h) } == 0 {
        return vec![None];
    }
    let (w, h) = (w as u32, h as u32);
    let mut s = vec![None, Some((w, h)), Some(((w / 2).max(1), (h / 2).max(1))), Some(((w / 3).max(1), (h / 5).max(1)))];
    s.push(Some(((w * 7 / 8).max(1), (h * 7 / 8).max(1))));
    s.push(Some((w + 3, h * 2)));
    s.push(Some((1, 1)));
    s
}

/// Returns the number of decodes compared and of differences, printing the first few.
fn compare(name: &str, data: &[u8], all_sizes: bool, shown: &mut usize) -> (usize, usize) {
    let mut n = (0, 0);
    let sizes = if all_sizes { sizes(data) } else { vec![None, sizes(data).get(2).copied().flatten()] };
    for size in sizes {
        for alpha in [false, true] {
            let (a, b) = (libwebp(data, size, alpha), ours(data, size, alpha));
            n.0 += 1;
            if a != b {
                n.1 += 1;
                if *shown < 400 && size.is_none() && alpha {
                    *shown += 1;
                    let what = |o: &Option<Vec<u8>>| o.as_ref().map_or("error".to_string(), |p| format!("{} bytes", p.len()));
                    let first = match (&a, &b) {
                        (Some(a), Some(b)) => a.iter().zip(b).position(|(x, y)| x != y),
                        _ => None,
                    };
                    if let Ok(dir) = std::env::var("FRAME_DIFFS") {
                        let _ = std::fs::create_dir_all(&dir);
                        let _ = std::fs::write(format!("{dir}/{}.webp", name.replace(' ', "_")), data);
                    }
                    eprintln!("{name} size {size:?} alpha {alpha}: libwebp {} ours {} first diff {first:?}", what(&a), what(&b));
                }
            }
        }
    }
    n
}

fn encode(w: usize, h: usize, rgba: &[u8], q: Option<f32>) -> Vec<u8> {
    let mut out = std::ptr::null_mut();
    let len = unsafe {
        match q {
            Some(q) => webp::WebPEncodeRGBA(rgba.as_ptr(), w as i32, h as i32, (w * 4) as i32, q, &mut out),
            None => webp::WebPEncodeLosslessRGBA(rgba.as_ptr(), w as i32, h as i32, (w * 4) as i32, &mut out),
        }
    };
    let v = unsafe { std::slice::from_raw_parts(out, len) }.to_vec();
    unsafe { webp::WebPFree(out as *mut _) };
    v
}

/// Lossy with `partitions` (libwebp's log2 count of token partitions).
fn encode_partitions(w: usize, h: usize, rgba: &[u8], q: f32, partitions: i32) -> Vec<u8> {
    let mut config = webp::WebPConfig::new_with_preset(webp::WebPPreset::WEBP_PRESET_DEFAULT, q).unwrap();
    config.partitions = partitions;
    encode_config(&config, w, h, rgba)
}

/// Lossless at `quality` and `method`: past quality 75 libwebp's references reach a million pixels back.
fn encode_lossless(w: usize, h: usize, rgba: &[u8], quality: f32, method: i32) -> Vec<u8> {
    let mut config = webp::WebPConfig::new_with_preset(webp::WebPPreset::WEBP_PRESET_DEFAULT, quality).unwrap();
    config.lossless = 1;
    config.method = method;
    encode_config(&config, w, h, rgba)
}

fn encode_config(config: &webp::WebPConfig, w: usize, h: usize, rgba: &[u8]) -> Vec<u8> {
    let mut picture = webp::WebPPicture::new().unwrap();
    picture.use_argb = config.lossless;
    picture.width = w as i32;
    picture.height = h as i32;
    let mut writer = unsafe { std::mem::zeroed::<webp::WebPMemoryWriter>() };
    unsafe {
        webp::WebPMemoryWriterInit(&mut writer);
        assert!(webp::WebPPictureImportRGBA(&mut picture, rgba.as_ptr(), (w * 4) as i32) != 0);
        picture.writer = Some(webp::WebPMemoryWrite);
        picture.custom_ptr = &mut writer as *mut _ as *mut std::ffi::c_void;
        assert!(webp::WebPEncode(config, &mut picture) != 0);
        let v = std::slice::from_raw_parts(writer.mem, writer.size).to_vec();
        webp::WebPMemoryWriterClear(&mut writer);
        webp::WebPPictureFree(&mut picture);
        v
    }
}

fn picture(rng: &mut StdRng, w: usize, h: usize, alpha: u8) -> Vec<u8> {
    let (fx, fy) = (rng.gen_range(1..9), rng.gen_range(1..9));
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let n: u8 = rng.gen_range(0..24);
            let a = match alpha {
                0 => 255,
                1 => ((x * 255 / w.max(1)) as u8).wrapping_add(n),
                _ => if (x / 7 + y / 5) % 3 == 0 { 0 } else { 200 },
            };
            match alpha {
                // Rows repeating a million pixels apart, past the window a reference reaches.
                5 => [(x * 3) as u8, (y % 450) as u8, (x ^ (y % 450)) as u8, 255],
                3 | 4 => {
                    let k = ((x / 3 + y / 2 + n as usize % 2) % if alpha == 3 { 4 } else { 40 }) as u8;
                    [k * 60, k * 13, 255 - k * 5, if k % 3 == 0 { 128 } else { 255 }]
                }
                _ => [(x * fx) as u8 ^ n, (y * fy) as u8, ((x + y) * 3) as u8 ^ (n / 2), a],
            }
        })
        .collect()
}

/// A smooth picture with bursts of four noise pixels, the second half's copied from bursts half
/// a million to a million pixels back: the burst's length code and the distance code are rare
/// and long, and the distance's 18 extra bits follow them, past one fill of the reader's window.
fn noise_bursts(rng: &mut StdRng, w: usize, h: usize, noise: u8) -> Vec<u8> {
    let mut p: Vec<u8> = (0..w * h)
        .flat_map(|i| [(i % w / 8) as u8 + rng.gen_range(0..noise), (i / w / 8) as u8 + rng.gen_range(0..noise), rng.gen_range(0..noise), 255])
        .collect();
    let mut bursts: Vec<usize> = Vec::new();
    let mut at = rng.gen_range(0..300);
    while at + 4 <= w * h {
        let from = bursts.iter().filter(|&&s| at - s >= (1 << 19) + 1000 && at - s < (1 << 20) - 1000).count();
        if from > 0 && rng.gen_bool(0.7) {
            let s = *bursts.iter().filter(|&&s| at - s >= (1 << 19) + 1000 && at - s < (1 << 20) - 1000).nth(rng.gen_range(0..from)).unwrap();
            p.copy_within(s * 4..(s + 4) * 4, at * 4);
        } else {
            for b in &mut p[at * 4..(at + 4) * 4] {
                *b = rng.gen::<u8>() | 1;
            }
            bursts.push(at);
        }
        at += rng.gen_range(100..500);
    }
    p
}

fn inputs() -> Vec<(String, Vec<u8>)> {
    let mut v = Vec::new();
    let mut dirs = vec![std::path::PathBuf::from("tests/images")];
    while let Some(d) = dirs.pop() {
        for e in std::fs::read_dir(d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                dirs.push(p);
            } else if p.extension().is_some_and(|x| x == "webp") {
                v.push((p.display().to_string(), std::fs::read(&p).unwrap()));
            }
        }
    }
    let mut rng = StdRng::seed_from_u64(7);
    for (w, h) in [(1, 1), (2, 3), (17, 9), (64, 64), (97, 61), (300, 211), (33, 700)] {
        for alpha in 0..5 {
            let p = picture(&mut rng, w, h, alpha);
            for q in [Some(10.0), Some(75.0), Some(100.0), None] {
                v.push((format!("gen {w}x{h} alpha {alpha} q {q:?}"), encode(w, h, &p, q)));
            }
        }
    }
    let p = picture(&mut rng, 2100, 1100, 5);
    v.push(("gen 2100x1100 tiled lossless".into(), encode(2100, 1100, &p, None)));
    let p = noise_bursts(&mut rng, 1600, 1200, 1);
    v.push(("gen 1600x1200 noise bursts lossless q100 m4".into(), encode_lossless(1600, 1200, &p, 100.0, 4)));
    // Partitions past a 64 KiB window, refilled while decoding.
    for (w, h, alpha) in [(1500, 1100, 0), (700, 900, 1)] {
        let p = picture(&mut rng, w, h, alpha);
        for (q, partitions) in [(95.0, 0), (95.0, 3), (40.0, 2)] {
            v.push((format!("gen {w}x{h} alpha {alpha} q {q} partitions {partitions}"), encode_partitions(w, h, &p, q, partitions)));
        }
    }
    v
}

/// The frames of an animation as libwebp's demuxer hands them to `WebPDecode`.
fn fragments(data: &[u8]) -> Vec<Vec<u8>> {
    let Ok(d) = WebPDecoder::new(std::io::Cursor::new(data)) else { return Vec::new() };
    if d.format_flags() & ANIMATION_FLAG == 0 {
        return Vec::new();
    }
    d.frames()
        .iter()
        .map(|f| {
            let start = if f.alpha.size > 0 { f.alpha.offset } else { f.image.offset } as usize;
            data[start..(f.image.offset + f.image.size) as usize].to_vec()
        })
        .collect()
}

#[test]
fn decodes_as_libwebp() {
    let mut shown = 0;
    let (mut n, mut diff) = (0, 0);
    let mut add = |(a, b): (usize, usize)| {
        n += a;
        diff += b;
    };
    let mut rng = StdRng::seed_from_u64(11);
    for (name, data) in inputs() {
        add(compare(&name, &data, true, &mut shown));
        for (i, frame) in fragments(&data).iter().enumerate().take(8) {
            add(compare(&format!("{name} frame {i}"), frame, true, &mut shown));
        }
        let variants = if data.len() < 100_000 { 90 } else { 8 };
        for k in 0..variants {
            let mut d = data.clone();
            if k % 3 == 2 {
                let at = rng.gen_range(20..d.len().max(21).min(120)).min(d.len() - 1);
                d[at] = rng.gen();
            } else if k % 3 == 0 {
                d.truncate(rng.gen_range(12..d.len().max(13)));
            } else {
                for _ in 0..rng.gen_range(1..4) {
                    let at = rng.gen_range(12..d.len().max(13)).min(d.len() - 1);
                    d[at] = rng.gen();
                }
            }
            add(compare(&format!("{name} variant {k}"), &d, false, &mut shown));
        }
        // A partition starting with 0xff carries bits past the top of the reader's value,
        // where libwebp's load sizes decide what comes out.
        if data.len() > 40 && &data[12..16] == b"VP8 " && data[23..26] == [0x9d, 0x01, 0x2a] {
            let size0 = (u32::from_le_bytes([data[20], data[21], data[22], 0]) >> 5) as usize;
            for at in [30, 30 + size0] {
                for run in [1, 2, 9] {
                    let mut d = data.clone();
                    let end = (at + run).min(d.len());
                    d[at.min(end)..end].fill(0xff);
                    add(compare(&format!("{name} 0xff x{run} at {at}"), &d, false, &mut shown));
                }
            }
        }
    }
    eprintln!("{n} decodes, {diff} differences");
    assert_eq!(diff, 0);
}
