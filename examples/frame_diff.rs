//! Prints where `FrameDecoder` and libwebp's `WebPDecode` disagree on one file, RGBA at full size.
use image_webp::{headers, FrameDecoder};
use libwebp_sys as webp;

fn main() {
    let data = std::fs::read(std::env::args().nth(1).unwrap()).unwrap();
    let (mut w, mut h) = (0, 0);
    let p = unsafe { webp::WebPDecodeRGBA(data.as_ptr(), data.len(), &mut w, &mut h) };
    if p.is_null() {
        println!("libwebp fails");
        return;
    }
    let theirs = unsafe { std::slice::from_raw_parts(p, (w * h * 4) as usize) }.to_vec();
    let hd = headers::decoding(&data).unwrap();
    println!("{hd:?}");
    let mut ours = Vec::new();
    let r = FrameDecoder::new(&data, &hd, None, true).and_then(|mut d| {
        while let Some((_, rows)) = d.next_band()? {
            ours.extend_from_slice(rows);
        }
        Ok(())
    });
    if let Err(e) = r {
        println!("ours: {e:?}");
        return;
    }
    for (i, (a, b)) in theirs.chunks(4).zip(ours.chunks(4)).enumerate() {
        if a != b {
            println!("x {} y {} libwebp {a:?} ours {b:?}", i as i32 % w, i as i32 / w);
        }
    }
}
