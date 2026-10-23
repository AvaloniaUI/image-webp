//! Times still decodes of a WebP through `FrameDecoder` the way a caller reading bands does it, at
//! full size and scaled to 1/2, 1/4 and 1/8, with parts run ahead on `helpers` pool threads (0:
//! none), printing the fastest run and a checksum of the pixels (from a run of its own).
//!
//! cargo run --release --example threads_bench -- image.webp [runs] [case|all] [helpers] [ahead]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use image_webp::{headers, FrameDecoder, Runner};

fn fnv(h: u64, data: &[u8]) -> u64 {
    let (words, rest) = data.as_chunks::<8>();
    let h = words.iter().fold(h, |h, w| (h ^ u64::from_le_bytes(*w)).wrapping_mul(0x100_0000_01b3));
    rest.iter().fold(h, |h, &b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// A pool of `threads` helpers that take work only while one is idle.
fn pool(threads: usize) -> Option<Runner> {
    if threads == 0 {
        return None;
    }
    let idle = Arc::new(AtomicUsize::new(threads));
    let (tx, rx) = sync_channel::<Box<dyn FnOnce() + Send>>(threads);
    let rx = Arc::new(Mutex::new(rx));
    for _ in 0..threads {
        let (rx, idle) = (rx.clone(), idle.clone());
        std::thread::spawn(move || loop {
            let Ok(work) = rx.lock().unwrap().recv() else { return };
            work();
            idle.fetch_add(1, Ordering::AcqRel);
        });
    }
    let tx = Mutex::new(tx);
    Some(Arc::new(move |work: Box<dyn FnOnce() + Send>| {
        let mut n = idle.load(Ordering::Acquire);
        loop {
            if n == 0 {
                return false;
            }
            match idle.compare_exchange_weak(n, n - 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(now) => n = now,
            }
        }
        match tx.lock().unwrap().try_send(work) {
            Ok(()) => true,
            Err(_) => {
                idle.fetch_add(1, Ordering::AcqRel);
                false
            }
        }
    }))
}

fn decode(data: &'static [u8], size: Option<(u32, u32)>, alpha: bool, runner: &Option<Runner>, ahead: usize, sum: bool) -> u64 {
    let headers = headers::read_decoding(&mut std::io::Cursor::new(data), 0, data.len() as u64).unwrap();
    let mut frame = FrameDecoder::new(data, &headers, size, alpha).unwrap();
    if let Some(runner) = runner {
        frame.run_ahead(runner.clone(), ahead);
    }
    let mut h = 0xcbf2_9ce4_8422_2325;
    while let Some((_, rows)) = frame.next_band().unwrap() {
        h = if sum { fnv(h, rows) } else { h ^ u64::from(rows[rows.len() / 2]) };
    }
    h
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data: &'static [u8] = std::fs::read(&args[1]).unwrap().leak();
    let runs: usize = args.get(2).map_or(5, |r| r.parse().unwrap());
    let only = args.get(3).filter(|c| *c != "all");
    let helpers: usize = args.get(4).map_or(0, |h| h.parse().unwrap());
    let ahead: usize = args.get(5).map_or(4, |a| a.parse().unwrap());
    let runner = pool(helpers);
    let headers = headers::read_decoding(&mut std::io::Cursor::new(data), 0, data.len() as u64).unwrap();
    let (w, h) = (headers.width, headers.height);
    let alpha = headers.has_alpha;
    let side = |n: u32, k: u32| ((f64::from(n) / f64::from(k)).round_ties_even() as u32).max(1);
    let mut cases = vec![("full".to_string(), None)];
    for k in [2, 4, 8] {
        cases.push((format!("1/{k}"), Some((side(w, k), side(h, k)))));
    }
    for (name, size) in cases {
        if only.is_some_and(|only| *only != name) {
            continue;
        }
        let sum = decode(data, size, alpha, &runner, ahead, true);
        let mut best = f64::MAX;
        for _ in 0..runs {
            let t = Instant::now();
            decode(data, size, alpha, &runner, ahead, false);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let (ow, oh) = size.unwrap_or((w, h));
        println!(
            "{name:<5} {ow:>5}x{oh:<5} {} helpers {helpers} ahead {ahead} {best:8.2} ms  {sum:016x}",
            if alpha { "rgba" } else { "rgb " }
        );
    }
}
