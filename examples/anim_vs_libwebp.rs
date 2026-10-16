//! Encodes animations with image-webp's `AnimationEncoder` and with libwebp's `WebPAnimEncoder`
//! at the same settings (as libvips drives it: kmin 9, kmax 10, exact off, alpha quality 100)
//! and compares them: bytes, per-frame PSNR of the composited frames against the source, time,
//! and peak memory with each side encoding in a process of its own. Both files are decoded with
//! libwebp's animation decoder and with image-webp's, which have to agree.
//!
//! cargo run --release --example anim_vs_libwebp -- [-q 30,75,90] [-m 0,4,6] [--lossless]
//!     [--argb] [--frames N] [--size WxH] [animated.webp ...]
//!
//! Without files the synthetic animations run: a moving square, small changes, full-frame
//! noise, an alpha fade, identical frames, one frame, 2x2, a sprite over transparency.
//! `--argb` feeds libwebp ARGB frames as this crate is fed, instead of the YUV libvips imports
//! lossy frames as (which libwebp turns back into ARGB, losing chroma). The gate check fails the run
//! (exit 1) where a lossy case is more than 0.5 dB under libwebp or more than 5% larger, or a
//! lossless case is not pixel-identical.

use std::io::Cursor;
use std::process::Command;
use std::time::Instant;

use image_webp::{AnimationEncoder, AnimationParams, ColorType, EncoderParams, WebPDecoder};
use libwebp_sys as sys;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct Animation {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// RGBA frames and their durations.
    pub frames: Vec<(Vec<u8>, u32)>,
}

#[derive(Clone, Copy)]
pub struct Settings {
    pub quality: u8,
    pub method: u8,
    pub lossless: bool,
    /// libwebp gets ARGB for lossy frames too: libvips hands it YUV, which it turns back
    /// into ARGB before coding, so its frames lose chroma before the comparison starts.
    pub argb: bool,
}

fn fill(width: u32, height: u32, mut f: impl FnMut(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            out.extend_from_slice(&f(x, y));
        }
    }
    out
}

pub fn synthetic(frames: usize, width: u32, height: u32) -> Vec<Animation> {
    let mut rng = StdRng::seed_from_u64(7);
    let gradient = |x: u32, y: u32| [(x * 255 / width) as u8, (y * 255 / height) as u8, 128, 255];
    let mut out = Vec::new();

    let moving = (0..frames)
        .map(|i| {
            let (sx, sy) = ((i as u32 * 13) % width, (i as u32 * 7) % height);
            let f = fill(width, height, |x, y| {
                if x >= sx && x < sx + width / 5 && y >= sy && y < sy + height / 5 {
                    [220, 30, 30, 255]
                } else {
                    gradient(x, y)
                }
            });
            (f, 70)
        })
        .collect();
    out.push(Animation {
        name: "moving".into(),
        width,
        height,
        frames: moving,
    });

    let base: Vec<u8> = fill(width, height, |x, y| {
        let n = rng.gen_range(0..40u32);
        [
            ((x * 3 + n) % 256) as u8,
            ((y * 2 + n) % 256) as u8,
            (((x + y) / 3 + n) % 256) as u8,
            255,
        ]
    });
    let mut small = Vec::new();
    let mut frame = base.clone();
    for i in 0..frames {
        for _ in 0..20 {
            let at = rng.gen_range(0..(width * height) as usize) * 4;
            frame[at..at + 3].copy_from_slice(&[(i * 40) as u8, 200, 90]);
        }
        small.push((frame.clone(), 50));
    }
    out.push(Animation {
        name: "small-changes".into(),
        width,
        height,
        frames: small,
    });

    let noise = (0..frames)
        .map(|_| {
            (
                fill(width, height, |_, _| [rng.gen(), rng.gen(), rng.gen(), 255]),
                40,
            )
        })
        .collect();
    out.push(Animation {
        name: "noise".into(),
        width,
        height,
        frames: noise,
    });

    let fade = (0..frames)
        .map(|i| {
            let a = (255 * i / frames.max(2).saturating_sub(1)) as u8;
            let f = fill(width, height, |x, y| {
                let (dx, dy) = (x as i64 - width as i64 / 2, y as i64 - height as i64 / 2);
                if dx * dx + dy * dy < (width as i64 / 3).pow(2) {
                    [40, 160, 240, a]
                } else {
                    [0, 0, 0, 0]
                }
            });
            (f, 60)
        })
        .collect();
    out.push(Animation {
        name: "alpha-fade".into(),
        width,
        height,
        frames: fade,
    });

    let still = fill(width, height, gradient);
    let other = fill(width, height, |x, y| gradient(y % width, x % height));
    let identical = vec![
        (still.clone(), 100),
        (still.clone(), 100),
        (still.clone(), 100),
        (other, 100),
        (still.clone(), 100),
    ];
    out.push(Animation {
        name: "identical".into(),
        width,
        height,
        frames: identical,
    });
    out.push(Animation {
        name: "single".into(),
        width,
        height,
        frames: vec![(still, 100)],
    });

    let tiny = (0..3)
        .map(|i| {
            (
                fill(2, 2, |x, y| {
                    [(x * 100 + i * 50) as u8, (y * 100) as u8, 0, 255]
                }),
                100,
            )
        })
        .collect();
    out.push(Animation {
        name: "tiny".into(),
        width: 2,
        height: 2,
        frames: tiny,
    });

    let sprite = (0..frames)
        .map(|i| {
            let sx = (i as u32 * 11) % width;
            let f = fill(width, height, |x, y| {
                if x >= sx && x < sx + width / 6 && y >= height / 3 && y < height / 3 + height / 4 {
                    [250, 200, 20, 255]
                } else {
                    [0, 0, 0, 0]
                }
            });
            (f, 80)
        })
        .collect();
    out.push(Animation {
        name: "sprite".into(),
        width,
        height,
        frames: sprite,
    });
    out
}

fn load(path: &str) -> Animation {
    let data = std::fs::read(path).unwrap();
    let mut d = WebPDecoder::new(Cursor::new(data)).unwrap();
    let (width, height) = d.dimensions();
    let mut frames = Vec::new();
    let mut buf = vec![0u8; d.output_buffer_size().unwrap()];
    while let Ok(duration) = d.read_frame(&mut buf) {
        let rgba = match d.has_alpha() {
            true => buf.clone(),
            false => buf
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        };
        frames.push((rgba, duration));
    }
    Animation {
        name: path.to_string(),
        width,
        height,
        frames,
    }
}

pub fn encode_ours(a: &Animation, s: Settings) -> Vec<u8> {
    let mut enc = AnimationEncoder::new(Cursor::new(Vec::new()), a.width, a.height).unwrap();
    let mut params = EncoderParams::default();
    params.use_lossy = !s.lossless;
    params.lossy_quality = s.quality;
    params.lossy_method = s.method;
    params.lossless_quality = s.quality;
    params.lossless_method = s.method;
    enc.set_params(params);
    enc.set_animation_params(AnimationParams::default());
    for (frame, duration) in &a.frames {
        enc.add_frame(frame, ColorType::Rgba8, *duration).unwrap();
    }
    enc.finish().unwrap().into_inner()
}

fn encode_libwebp(a: &Animation, s: Settings) -> Vec<u8> {
    unsafe {
        let mut config = std::mem::zeroed::<sys::WebPConfig>();
        let quality = if s.lossless {
            75.
        } else {
            f32::from(s.quality)
        };
        assert_ne!(
            sys::WebPConfigInitInternal(
                &mut config,
                sys::WebPPreset::WEBP_PRESET_DEFAULT,
                quality,
                sys::WEBP_ENCODER_ABI_VERSION as i32
            ),
            0
        );
        config.lossless = s.lossless as i32;
        config.exact = 0;
        config.alpha_quality = 100;
        config.method = i32::from(s.method);
        config.pass = 1;
        if s.lossless {
            config.quality = f32::from(s.quality);
        }
        assert_ne!(sys::WebPValidateConfig(&config), 0);
        let mut options = std::mem::zeroed::<sys::WebPAnimEncoderOptions>();
        assert_ne!(
            sys::WebPAnimEncoderOptionsInitInternal(&mut options, sys::WEBP_MUX_ABI_VERSION as i32),
            0
        );
        options.kmin = 9;
        options.kmax = 10;
        let enc = sys::WebPAnimEncoderNewInternal(
            a.width as i32,
            a.height as i32,
            &options,
            sys::WEBP_MUX_ABI_VERSION as i32,
        );
        assert!(!enc.is_null());
        let mut timestamp = 0;
        for (frame, duration) in &a.frames {
            let mut pic = std::mem::zeroed::<sys::WebPPicture>();
            assert_ne!(
                sys::WebPPictureInitInternal(&mut pic, sys::WEBP_ENCODER_ABI_VERSION as i32),
                0
            );
            pic.width = a.width as i32;
            pic.height = a.height as i32;
            pic.use_argb = (s.lossless || s.argb) as i32;
            assert_ne!(
                sys::WebPPictureImportRGBA(&mut pic, frame.as_ptr(), (a.width * 4) as i32),
                0
            );
            assert_ne!(
                sys::WebPAnimEncoderAdd(enc, &mut pic, timestamp, &config),
                0
            );
            sys::WebPPictureFree(&mut pic);
            timestamp += *duration as i32;
        }
        assert_ne!(
            sys::WebPAnimEncoderAdd(enc, std::ptr::null_mut(), timestamp, std::ptr::null()),
            0
        );
        let mut out = sys::WebPData {
            bytes: std::ptr::null(),
            size: 0,
        };
        assert_ne!(sys::WebPAnimEncoderAssemble(enc, &mut out), 0);
        let bytes = std::slice::from_raw_parts(out.bytes, out.size).to_vec();
        sys::WebPDataClear(&mut out);
        sys::WebPAnimEncoderDelete(enc);
        bytes
    }
}

/// The composited RGBA frames and their durations, as libwebp's animation decoder shows them.
pub fn decode_libwebp(data: &[u8]) -> Vec<(Vec<u8>, u32)> {
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
        assert!(!dec.is_null(), "libwebp refuses the file");
        let mut info = std::mem::zeroed::<sys::WebPAnimInfo>();
        assert_ne!(sys::WebPAnimDecoderGetInfo(dec, &mut info), 0);
        let size = (info.canvas_width * info.canvas_height * 4) as usize;
        let mut frames = Vec::new();
        let mut last = 0;
        while sys::WebPAnimDecoderHasMoreFrames(dec) != 0 {
            let mut buf: *mut u8 = std::ptr::null_mut();
            let mut timestamp = 0;
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

/// RGBA of a still as libwebp decodes it.
pub fn decode_libwebp_still(data: &[u8]) -> Vec<u8> {
    unsafe {
        let mut config = std::mem::zeroed::<sys::WebPDecoderConfig>();
        assert!(sys::WebPInitDecoderConfig(&mut config));
        assert_eq!(
            sys::WebPGetFeatures(data.as_ptr(), data.len(), &mut config.input),
            sys::VP8StatusCode::VP8_STATUS_OK
        );
        let (w, h) = (config.input.width as usize, config.input.height as usize);
        let mut out = vec![0u8; w * h * 4];
        config.output.colorspace = sys::WEBP_CSP_MODE::MODE_RGBA;
        config.output.is_external_memory = 1;
        config.output.u.RGBA = sys::WebPRGBABuffer {
            rgba: out.as_mut_ptr(),
            stride: (w * 4) as i32,
            size: out.len(),
        };
        assert_eq!(
            sys::WebPDecode(data.as_ptr(), data.len(), &mut config),
            sys::VP8StatusCode::VP8_STATUS_OK
        );
        out
    }
}

pub fn decode_ours(data: &[u8]) -> Vec<(Vec<u8>, u32)> {
    let mut d = WebPDecoder::new(Cursor::new(data.to_vec())).unwrap();
    let mut buf = vec![0u8; d.output_buffer_size().unwrap()];
    let has_alpha = d.has_alpha();
    let rgba = |buf: &[u8]| match has_alpha {
        true => buf.to_vec(),
        false => buf
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
    };
    if !d.is_animated() {
        d.read_image(&mut buf).unwrap();
        return vec![(rgba(&buf), 0)];
    }
    let mut frames = Vec::new();
    while let Ok(duration) = d.read_frame(&mut buf) {
        frames.push((rgba(&buf), duration));
    }
    frames
}

/// RGBA pixels with the colour under alpha 0 taken out, which no decoder has to keep.
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

/// PSNR of RGBA `a` against `b`, with the colour under fully transparent source pixels ignored.
fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut sum = 0u64;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let n = if pb[3] == 0 { 3 } else { 0 };
        for c in n..4 {
            let d = i64::from(pa[c]) - i64::from(pb[c]);
            sum += (d * d) as u64;
        }
    }
    if sum == 0 {
        return 99.;
    }
    let mse = sum as f64 / a.len() as f64;
    10. * (255f64 * 255. / mse).log10()
}

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

struct Side {
    bytes: usize,
    millis: f64,
    /// Per source frame: the dB of the frame shown at that time.
    db: Vec<f64>,
    frames: usize,
    identical: bool,
}

/// Judges `data` against the source: frame by frame as libwebp decodes it, matched to source
/// frames by time since identical frames merge.
fn judge(a: &Animation, data: &[u8]) -> (Vec<f64>, usize, bool) {
    let theirs = decode_libwebp(data);
    let ours = decode_ours(data);
    assert_eq!(
        theirs.len(),
        ours.len(),
        "{}: decoders disagree on the frame count",
        a.name
    );
    for (i, (t, o)) in theirs.iter().zip(&ours).enumerate() {
        assert!(
            masked(&t.0) == masked(&o.0),
            "{}: decoders disagree on frame {i}",
            a.name
        );
        if theirs.len() > 1 {
            assert_eq!(
                t.1, o.1,
                "{}: decoders disagree on the duration of frame {i}",
                a.name
            );
        }
    }
    let total: u32 = a.frames.iter().map(|f| f.1).sum();
    let shown: u32 = theirs.iter().map(|f| f.1).sum();
    assert!(
        theirs.len() == 1 || shown == total,
        "{}: plays {shown} ms, the source {total} ms",
        a.name
    );
    let mut db = Vec::new();
    let mut identical = true;
    let (mut at, mut i) = (0u32, 0usize);
    for (frame, duration) in &a.frames {
        while i + 1 < theirs.len() && theirs[..=i].iter().map(|f| f.1).sum::<u32>() <= at {
            i += 1;
        }
        let d = psnr(&theirs[i].0, frame);
        identical &= masked(&theirs[i].0) == masked(frame);
        db.push(d);
        at += duration;
    }
    (db, theirs.len(), identical)
}

fn run_side(a: &Animation, s: Settings, ours: bool) -> Side {
    let start = Instant::now();
    let data = if ours {
        encode_ours(a, s)
    } else {
        encode_libwebp(a, s)
    };
    let millis = start.elapsed().as_secs_f64() * 1000.;
    let (db, frames, identical) = judge(a, &data);
    Side {
        bytes: data.len(),
        millis,
        db,
        frames,
        identical,
    }
}

fn child(a_index: usize, args: &[String], ours: bool) -> (f64, u64) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(args)
        .arg("--child")
        .arg(a_index.to_string())
        .arg(if ours { "ours" } else { "libwebp" })
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.split_whitespace();
    (
        it.next().unwrap().parse().unwrap(),
        it.next().unwrap().parse().unwrap(),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut qualities = vec![75u8];
    let mut methods = vec![4u8];
    let mut lossless = false;
    let mut argb = false;
    let mut frames = 12;
    let mut size = (256, 192);
    let mut files = Vec::new();
    let mut child_mode = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-q" => {
                qualities = args[i + 1].split(',').map(|q| q.parse().unwrap()).collect();
                i += 1;
            }
            "-m" => {
                methods = args[i + 1].split(',').map(|m| m.parse().unwrap()).collect();
                i += 1;
            }
            "--lossless" => lossless = true,
            "--argb" => argb = true,
            "--frames" => {
                frames = args[i + 1].parse().unwrap();
                i += 1;
            }
            "--size" => {
                let (w, h) = args[i + 1].split_once('x').unwrap();
                size = (w.parse().unwrap(), h.parse().unwrap());
                i += 1;
            }
            "--child" => {
                child_mode = Some((args[i + 1].parse::<usize>().unwrap(), args[i + 2] == "ours"));
                i += 2;
            }
            f => files.push(f.to_string()),
        }
        i += 1;
    }
    let animations: Vec<Animation> = match files.is_empty() {
        true => synthetic(frames, size.0, size.1),
        false => files.iter().map(|f| load(f)).collect(),
    };
    let settings: Vec<Settings> = qualities
        .iter()
        .flat_map(|&quality| {
            methods.iter().map(move |&method| Settings {
                quality,
                method,
                lossless,
                argb,
            })
        })
        .collect();
    let base_args: Vec<String> = args
        .iter()
        .take_while(|a| *a != "--child")
        .cloned()
        .collect();

    if let Some((index, ours)) = child_mode {
        // One process per side and animation: its peak memory is that side's, over the settings.
        let a = &animations[index];
        let start = Instant::now();
        for s in &settings {
            std::hint::black_box(if ours {
                encode_ours(a, *s)
            } else {
                encode_libwebp(a, *s)
            });
        }
        println!(
            "{} {}",
            start.elapsed().as_secs_f64() * 1000.,
            peak_rss_kb()
        );
        return;
    }

    let mut failed = false;
    println!(
        "{:<24} {:>10} {:>8} {:>8} {:>6} {:>7} {:>7} {:>8} {:>8}",
        "case", "settings", "ours B", "libwebp", "size", "min dB", "lw min", "mean dB", "lw mean"
    );
    for (index, a) in animations.iter().enumerate() {
        for s in &settings {
            let o = run_side(a, *s, true);
            let l = run_side(a, *s, false);
            let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let ratio = o.bytes as f64 / l.bytes as f64;
            let label = format!(
                "q{} m{}{}",
                s.quality,
                s.method,
                if s.lossless { " ll" } else { "" }
            );
            println!(
                "{:<24} {:>10} {:>8} {:>8} {:>5.1}% {:>7.2} {:>7.2} {:>8.2} {:>8.2}  frames {}/{} {:.1} ms vs {:.1} ms",
                a.name, label, o.bytes, l.bytes, (ratio - 1.) * 100., min(&o.db), min(&l.db), mean(&o.db), mean(&l.db), o.frames, l.frames, o.millis, l.millis
            );
            let mut misses = Vec::new();
            if s.lossless && !o.identical {
                misses.push("not pixel-identical".to_string());
            }
            if ratio > 1.05 {
                misses.push(format!("{:.1}% larger", (ratio - 1.) * 100.));
            }
            for (i, (od, ld)) in o.db.iter().zip(&l.db).enumerate() {
                if !s.lossless && *od < ld - 0.5 {
                    misses.push(format!("frame {i}: {od:.2} dB vs {ld:.2}"));
                }
            }
            if !misses.is_empty() {
                failed = true;
                println!("  MISS {}", misses.join("; "));
            }
        }
        let (om, orss) = child(index, &base_args, true);
        let (lm, lrss) = child(index, &base_args, false);
        println!(
            "  process: ours {om:.1} ms {} MB peak, libwebp {lm:.1} ms {} MB peak",
            orss / 1024,
            lrss / 1024
        );
    }
    if failed {
        std::process::exit(1);
    }
}
