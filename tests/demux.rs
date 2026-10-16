//! `WebPDecoder`'s chunk walk against libwebp's `WebPDemux` over fixtures, generated chunk sequences and
//! their byte-changed and cut copies.

use std::io::Cursor;

use image_webp::WebPDecoder;
use libwebp_sys as webp;
use rand::{rngs::StdRng, Rng, SeedableRng};

#[derive(Debug, PartialEq)]
struct Outcome {
    flags: u32,
    canvas: (i32, i32),
    loop_count: i32,
    bgcolor: u32,
    frames: Vec<FrameOutcome>,
    chunks: Vec<(usize, Option<(usize, usize)>)>,
}

#[derive(Debug, PartialEq)]
struct FrameOutcome {
    offset: (i32, i32),
    size: (i32, i32),
    has_alpha: bool,
    duration: i32,
    dispose_to_background: bool,
    no_blend: bool,
    complete: bool,
    payload: (usize, usize),
}

const FOURCCS: [&[u8; 4]; 4] = [b"ICCP", b"EXIF", b"XMP ", b"ABCD"];

fn libwebp(data: &[u8]) -> Option<Outcome> {
    let input = webp::WebPData { bytes: data.as_ptr(), size: data.len() };
    let d = unsafe { webp::WebPDemuxInternal(&input, 0, std::ptr::null_mut(), webp::WEBP_DEMUX_ABI_VERSION as i32) };
    if d.is_null() {
        return None;
    }
    let get = |f| unsafe { webp::WebPDemuxGetI(d, f) };
    let n = get(webp::WebPFormatFeature::WEBP_FF_FRAME_COUNT);
    let mut frames = Vec::new();
    for i in 1..=n as i32 {
        let mut it = unsafe { std::mem::zeroed::<webp::WebPIterator>() };
        assert!(unsafe { webp::WebPDemuxGetFrame(d, i, &mut it) } != 0);
        frames.push(FrameOutcome {
            offset: (it.x_offset, it.y_offset),
            size: (it.width, it.height),
            has_alpha: it.has_alpha != 0,
            duration: it.duration,
            dispose_to_background: it.dispose_method == webp::WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND,
            no_blend: it.blend_method == webp::WebPMuxAnimBlend::WEBP_MUX_NO_BLEND,
            complete: it.complete != 0,
            payload: (it.fragment.bytes as usize - data.as_ptr() as usize, it.fragment.size),
        });
    }
    let chunks = FOURCCS
        .iter()
        .map(|cc| {
            let mut it = unsafe { std::mem::zeroed::<webp::WebPChunkIterator>() };
            let c = std::ffi::CString::new(&cc[..]).unwrap();
            if unsafe { webp::WebPDemuxGetChunk(d, c.as_ptr(), 1, &mut it) } == 0 {
                return (0, None);
            }
            (it.num_chunks as usize, Some((it.chunk.bytes as usize - data.as_ptr() as usize, it.chunk.size)))
        })
        .collect();
    let out = Outcome {
        flags: get(webp::WebPFormatFeature::WEBP_FF_FORMAT_FLAGS),
        canvas: (get(webp::WebPFormatFeature::WEBP_FF_CANVAS_WIDTH) as i32, get(webp::WebPFormatFeature::WEBP_FF_CANVAS_HEIGHT) as i32),
        loop_count: get(webp::WebPFormatFeature::WEBP_FF_LOOP_COUNT) as i32,
        bgcolor: get(webp::WebPFormatFeature::WEBP_FF_BACKGROUND_COLOR),
        frames,
        chunks,
    };
    unsafe { webp::WebPDemuxDelete(d) };
    Some(out)
}

/// The walk as `new` makes it, and as `start` and `finish` make it, which must agree.
fn ours(data: &[u8]) -> Option<Outcome> {
    let whole = WebPDecoder::new(Cursor::new(data)).ok().map(|d| outcome(&d));
    let started = WebPDecoder::start(Cursor::new(data)).ok().and_then(|mut d| {
        let early = outcome(&d);
        d.finish().ok()?;
        let late = outcome(&d);
        assert_eq!((early.flags, early.canvas), (late.flags, late.canvas));
        assert_eq!(early.frames.first(), late.frames.first());
        // A file that is no animation may hold its ANIM chunk after the image.
        if early.flags & image_webp::ANIMATION_FLAG != 0 {
            assert_eq!((early.loop_count, early.bgcolor), (late.loop_count, late.bgcolor));
        }
        Some(late)
    });
    assert_eq!(whole, started, "start + finish differs from new");
    whole
}

fn outcome<R: std::io::BufRead + std::io::Seek>(d: &WebPDecoder<R>) -> Outcome {
    let frames = d
        .frames()
        .iter()
        .map(|f| {
            let (start, mut size) = (f.image.offset as usize, f.image.size as usize);
            let start = if f.alpha.size > 0 {
                let inter = if f.image.offset > 0 { f.image.offset - (f.alpha.offset + f.alpha.size) } else { 0 };
                size += (f.alpha.size + inter) as usize;
                f.alpha.offset as usize
            } else {
                start
            };
            FrameOutcome {
                offset: (f.x_offset, f.y_offset),
                size: (f.width, f.height),
                has_alpha: f.has_alpha,
                duration: f.duration,
                dispose_to_background: f.dispose_to_background,
                no_blend: f.no_blend,
                complete: f.complete,
                payload: (start, size),
            }
        })
        .collect();
    let chunks = FOURCCS
        .iter()
        .map(|cc| {
            let n = d.chunks().iter().filter(|c| &c.fourcc == *cc).count();
            (n, d.chunk(cc, 1).map(|c| (c.offset as usize, c.size as usize)))
        })
        .map(|(n, c)| if n == 0 { (0, None) } else { (n, c) })
        .collect();
    Outcome {
        flags: d.format_flags(),
        canvas: (d.dimensions().0 as i32, d.dimensions().1 as i32),
        loop_count: d.animation_params().1,
        bgcolor: d.animation_params().0,
        frames,
        chunks,
    }
}

/// A file cut after its first frame's chunks starts, decodes its first frame as the whole
/// file would, and fails to finish; `new` refuses it.
fn check_cut(data: &[u8]) {
    let Ok(whole) = WebPDecoder::new(Cursor::new(data)) else { return };
    let f = whole.frames()[0].clone();
    let frame_end = (f.image.offset + f.image.size) as usize;
    for cut in [frame_end + 8, data.len() - 1] {
        if cut >= data.len() {
            continue;
        }
        let mut d = WebPDecoder::start(Cursor::new(&data[..cut])).expect("starts");
        assert_eq!(d.frames().first().map(|f| f.image), Some(f.image));
        assert!(!d.finished());
        assert!(d.finish().is_err());
        assert!(WebPDecoder::new(Cursor::new(&data[..cut])).is_err());
    }
}

fn check(data: &[u8], what: &str) -> bool {
    // Raw VP8/VP8L frames without a RIFF header are out of scope.
    if !data.starts_with(b"RIFF") {
        return false;
    }
    let (a, b) = (libwebp(data), ours(data));
    assert_eq!(a, b, "{what}: {} bytes, head {:02x?}", data.len(), &data[..data.len().min(64)]);
    a.is_some()
}

fn chunk(fourcc: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut c = fourcc.to_vec();
    c.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    c.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        c.push(0);
    }
    c
}

fn riff(body: &[u8]) -> Vec<u8> {
    let mut f = b"RIFF".to_vec();
    f.extend_from_slice(&(body.len() as u32 + 4).to_le_bytes());
    f.extend_from_slice(b"WEBP");
    f.extend_from_slice(body);
    f
}

fn walk(data: &[u8], mut at: usize, end: usize, out: &mut Vec<(Vec<u8>, Vec<u8>)>) {
    while at + 8 <= end {
        let size = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap()) as usize;
        let to = (at + 8 + size).min(end);
        out.push((data[at..at + 4].to_vec(), data[at + 8..to].to_vec()));
        at += 8 + size + (size & 1);
    }
}

fn fixtures() -> Vec<Vec<u8>> {
    let mut files = Vec::new();
    for dir in ["tests/images/gallery1", "tests/images/gallery2", "tests/images/animated", "tests/images/regression"] {
        let mut paths: Vec<_> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
        paths.sort();
        for p in paths.into_iter().filter(|p| p.extension().is_some_and(|e| e == "webp")) {
            files.push(std::fs::read(p).unwrap());
        }
    }
    files
}

fn vp8x(flags: u8, w: u32, h: u32) -> Vec<u8> {
    let mut p = vec![flags, 0, 0, 0];
    p.extend_from_slice(&(w - 1).to_le_bytes()[..3]);
    p.extend_from_slice(&(h - 1).to_le_bytes()[..3]);
    chunk(b"VP8X", &p)
}

fn anmf(x: u32, y: u32, w: u32, h: u32, sub: &[u8]) -> Vec<u8> {
    let mut p = Vec::new();
    for v in [x / 2, y / 2, w - 1, h - 1, 100] {
        p.extend_from_slice(&v.to_le_bytes()[..3]);
    }
    p.push(0);
    p.extend_from_slice(sub);
    chunk(b"ANMF", &p)
}

fn generated(rng: &mut StdRng, images: &[(Vec<u8>, Vec<u8>, u32, u32)]) -> Vec<u8> {
    let pick = |rng: &mut StdRng| &images[rng.gen_range(0..images.len())];
    let (_, _, w, h) = pick(rng);
    let (w, h) = if rng.gen_bool(0.8) { (*w, *h) } else { (rng.gen_range(1..300), rng.gen_range(1..300)) };
    let flags: u8 = if rng.gen_bool(0.1) { rng.gen() } else { [0x00, 0x02, 0x10, 0x12, 0x20, 0x08, 0x3e, 0x04][rng.gen_range(0..8)] };
    let alph = chunk(b"ALPH", &[0, 1, 2, 3, 4]);
    let meta = |rng: &mut StdRng, body: &mut Vec<u8>| {
        for (cc, p) in [(b"ICCP", 0.5), (b"EXIF", 0.5), (b"XMP ", 0.2), (b"ABCD", 0.2)] {
            if rng.gen_bool(p) {
                body.extend(chunk(cc, &vec![7; rng.gen_range(0..40)]));
            }
        }
    };
    let mut body = Vec::new();
    match rng.gen_range(0..4) {
        0 => {
            let (cc, p, iw, ih) = pick(rng);
            let alpha = rng.gen_bool(0.4);
            body.extend(vp8x(if rng.gen_bool(0.2) { flags } else { [0, 0x10, 0x28, 0x38][rng.gen_range(0..4)] }, *iw, *ih));
            meta(rng, &mut body);
            if alpha {
                body.extend(&alph);
            }
            body.extend(chunk(cc, p));
            meta(rng, &mut body);
        }
        1 => {
            body.extend(vp8x(if rng.gen_bool(0.2) { flags } else { [0x02, 0x12, 0x2a, 0x3e][rng.gen_range(0..4)] }, w.max(64), h.max(64)));
            meta(rng, &mut body);
            body.extend(chunk(b"ANIM", &[0xff, 0xff, 0xff, 0xff, 0, 0]));
            for _ in 0..rng.gen_range(1..5) {
                let (cc, p, fw, fh) = pick(rng);
                let mut sub = if rng.gen_bool(0.3) { alph.clone() } else { Vec::new() };
                sub.extend(chunk(cc, p));
                let (x, y) = if rng.gen_bool(0.8) { (0, 0) } else { (rng.gen_range(0..40), rng.gen_range(0..40)) };
                body.extend(anmf(x, y, *fw, *fh, &sub));
            }
            meta(rng, &mut body);
        }
        _ => {
            if rng.gen_bool(0.9) {
                body.extend(vp8x(flags, w, h));
            }
            for _ in 0..rng.gen_range(0..7) {
                let c = match rng.gen_range(0..10) {
                    0 => chunk(b"ICCP", &vec![7; rng.gen_range(0..40)]),
                    1 => chunk(b"EXIF", &vec![8; rng.gen_range(0..40)]),
                    2 => chunk(b"XMP ", &vec![9; rng.gen_range(0..9)]),
                    3 => chunk(b"ABCD", &vec![1; rng.gen_range(0..9)]),
                    4 => chunk(b"ANIM", &vec![0xff; [6, 5, 7, 8][rng.gen_range(0..4)]]),
                    5 | 6 => {
                        let (cc, p, fw, fh) = pick(rng);
                        let mut sub = if rng.gen_bool(0.3) { alph.clone() } else { Vec::new() };
                        sub.extend(chunk(cc, p));
                        if rng.gen_bool(0.2) {
                            sub.extend(chunk(b"ABCD", &[1, 2, 3]));
                        }
                        let (x, y) = if rng.gen_bool(0.7) { (0, 0) } else { (rng.gen_range(0..40), rng.gen_range(0..40)) };
                        anmf(x, y, *fw, *fh, &sub)
                    }
                    7 => alph.clone(),
                    _ => {
                        let (cc, p, _, _) = pick(rng);
                        chunk(cc, p)
                    }
                };
                body.extend(c);
            }
        }
    }
    let mut f = riff(&body);
    if rng.gen_bool(0.05) {
        f.extend_from_slice(&[0; 7]);
    }
    f
}

fn mutate(rng: &mut StdRng, f: &[u8]) -> Vec<u8> {
    let mut m = f.to_vec();
    match rng.gen_range(0..5) {
        0 => m.truncate(rng.gen_range(0..m.len())),
        1 => {
            for _ in 0..rng.gen_range(1..4) {
                let i = rng.gen_range(0..m.len().min(200));
                m[i] = rng.gen();
            }
        }
        2 => {
            let i = rng.gen_range(0..m.len());
            m[i] ^= 1 << rng.gen_range(0..8);
        }
        3 => {
            let i = rng.gen_range(0..m.len().min(200));
            m[i] = m[i].wrapping_add(if rng.gen() { 1 } else { 255 });
        }
        _ => {
            let i = rng.gen_range(4..8);
            m[i] = m[i].wrapping_add(rng.gen_range(1..4));
        }
    }
    m
}

#[test]
fn demux_matches_libwebp() {
    let files = fixtures();
    let mut images = Vec::new();
    for f in &files {
        assert!(check(f, "fixture"));
        check_cut(f);
        let mut cs = Vec::new();
        walk(f, 12, f.len(), &mut cs);
        let mut nested = Vec::new();
        for (cc, p) in &cs {
            if cc == b"ANMF" && p.len() > 16 {
                walk(p, 16, p.len(), &mut nested);
            }
        }
        for (cc, p) in cs.into_iter().chain(nested) {
            let lone = riff(&chunk(&cc, &p));
            if (cc == b"VP8 " || cc == b"VP8L") && images.len() < 12 {
                if let Some(o) = ours(&lone) {
                    images.push((cc, p, o.canvas.0 as u32, o.canvas.1 as u32));
                }
            }
        }
    }
    assert!(images.len() >= 4);
    let lossy = images.iter().find(|i| i.0 == b"VP8 ").unwrap();
    let lossless = images.iter().find(|i| i.0 == b"VP8L").unwrap();
    let alph = chunk(b"ALPH", &[0, 1, 2]);
    let (vp8, vp8l) = (chunk(b"VP8 ", &lossy.1), chunk(b"VP8L", &lossless.1));
    for d in [-1i64, 0, 1] {
        let mut p = lossy.1.clone();
        let bits = u32::from_le_bytes([p[0], p[1], p[2], 0]) & 0x1f | (((p.len() as i64 + d) as u32) << 5);
        p[..3].copy_from_slice(&bits.to_le_bytes()[..3]);
        check(&riff(&chunk(b"VP8 ", &p)), "partition length");
    }
    for (flags, w, h) in [(0x10, lossy.2, lossy.3), (0x00, lossy.2, lossy.3), (0x10, lossless.2, lossless.3)] {
        for seq in [
            [&alph[..], &alph, &vp8].concat(),
            [&vp8[..], &alph].concat(),
            [&alph[..], &vp8l].concat(),
            [&alph[..], &chunk(b"ABCD", &[1]), &vp8].concat(),
            [&vp8[..], &vp8].concat(),
            [&alph[..]].concat(),
        ] {
            check(&riff(&[vp8x(flags, w, h), seq.clone()].concat()), "chunk order");
            check(&riff(&seq), "chunk order");
            let anim = [vp8x(0x12, w, h), chunk(b"ANIM", &[0; 6]), anmf(0, 0, w, h, &seq)].concat();
            check(&riff(&anim), "chunk order in a frame");
        }
    }
    let mut rng = StdRng::seed_from_u64(0x5eed);
    let (mut valid, mut total) = (0, 0);
    for f in &files {
        for _ in 0..300 {
            valid += check(&mutate(&mut rng, f), "fixture mutation") as usize;
            total += 1;
        }
    }
    for _ in 0..30000 {
        let f = generated(&mut rng, &images);
        valid += check(&f, "generated") as usize;
        for _ in 0..3 {
            valid += check(&mutate(&mut rng, &f), "generated mutation") as usize;
        }
        total += 4;
    }
    assert!(valid > total / 10 && valid < total * 9 / 10, "{valid} of {total} accepted");
}
