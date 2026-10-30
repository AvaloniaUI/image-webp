//! Sets `decoder_simd` where the decoder has vector kernels for the target's baseline: SSE2 on
//! x86 and x86_64, NEON on aarch64, simd128 on wasm32.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(decoder_simd)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has = |f: &str| features.split(',').any(|x| x == f);
    let simd = match arch.as_str() {
        "x86" | "x86_64" => has("sse2") && has("fxsr"),
        "aarch64" => has("neon"),
        "wasm32" => has("simd128"),
        _ => false,
    };
    if simd {
        println!("cargo::rustc-cfg=decoder_simd");
    }
}
