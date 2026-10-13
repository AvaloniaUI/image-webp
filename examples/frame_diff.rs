//! Prints where `FrameDecoder` and libwebp's `WebPDecode` disagree on each file, RGBA at full
//! size: a still as one bitstream, an animation frame by frame as the demuxer hands them out.
use image_webp::{headers, FrameDecoder, WebPDecoder, ANIMATION_FLAG};
use libwebp_sys as webp;

fn fragments(data: &[u8]) -> Vec<Vec<u8>> {
    let Ok(d) = WebPDecoder::new(std::io::Cursor::new(data)) else { return vec![data.to_vec()] };
    if d.format_flags() & ANIMATION_FLAG == 0 {
        return vec![data.to_vec()];
    }
    d.frames()
        .iter()
        .map(|f| {
            let start = if f.alpha.size > 0 { f.alpha.offset } else { f.image.offset } as usize;
            data[start..(f.image.offset + f.image.size) as usize].to_vec()
        })
        .collect()
}

/// The differing pixels of one bitstream, `None` where libwebp fails.
fn diff(data: &[u8], shown: &mut usize) -> Option<usize> {
    let (mut w, mut h) = (0, 0);
    let p = unsafe { webp::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut w, &mut h) };
    if p.is_null() {
        return None;
    }
    let theirs = unsafe { std::slice::from_raw_parts(p, (w * h * 4) as usize) }.to_vec();
    unsafe { webp::WebPFree(p as *mut _) };
    let hd = headers::decoding(data).unwrap();
    let mut ours = Vec::new();
    let r = FrameDecoder::new(data, &hd, None, true).and_then(|mut d| {
        while let Some((_, rows)) = d.next_band()? {
            ours.extend_from_slice(rows);
        }
        Ok(())
    });
    if let Err(e) = r {
        println!("ours: {e:?}");
        return Some(theirs.len() / 4);
    }
    let mut n = 0;
    for (i, (a, b)) in theirs.chunks(4).zip(ours.chunks(4)).enumerate() {
        if a != b {
            n += 1;
            if *shown < 20 {
                *shown += 1;
                println!("x {} y {} libwebp {a:?} ours {b:?}", i as i32 % w, i as i32 / w);
            }
        }
    }
    Some(n)
}

fn main() {
    let mut total = 0;
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).unwrap();
        let mut shown = 0;
        let mut wrong = 0;
        let mut frames = 0;
        for frame in fragments(&data) {
            frames += 1;
            match diff(&frame, &mut shown) {
                Some(n) => wrong += n,
                None => println!("{path} frame {frames}: libwebp fails"),
            }
        }
        total += wrong;
        println!("{path}: {frames} frames, {wrong} wrong pixels");
    }
    std::process::exit(i32::from(total > 0));
}
