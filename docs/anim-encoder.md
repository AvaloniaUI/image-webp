# Animated WebP encoder

Task brief for an agent. Goal: an animated WebP encoder in this crate, written in image-webp's
own style and conventions and aimed at speed, that replaces libwebp's `WebPAnimEncoder` +
`WebPMux` in image-processing. libwebp is the reference to measure against, not code to port
line by line.

## Why

image-processing (`../image-processing`) is removing libvips, sharp and libwebp. Animated WebP
output is the last large piece that still needs libwebp: `src/engine/webp_anim.rs` (`Encoder`,
`config`, `mux`) drives `WebPAnimEncoderNewInternal` / `WebPAnimEncoderAdd` /
`WebPAnimEncoderAssemble` and then `WebPMux*` to set the loop count and the ICCP/EXIF/XMP chunks.
Still frames are already coded by this crate (lossy `vp8enc`, lossless `vp8l`); what is missing
is everything between frames.

## What the caller does today (the contract to keep)

From `../image-processing/src/engine/webp_anim.rs` and `src/pipeline.rs` (~1667):

- Frames arrive as whole rows, top to bottom, frame after frame, RGB or RGBA (`bands` 3 or 4),
  all the same `width` × `page_height`. Sides above 16383 fail (`webp: image too large`).
- Options: `quality` (0-100), `effort` (libwebp `method`, default 4), `lossless`,
  `near_lossless` (quality becomes the near-lossless level), `kmin = 9`, `kmax = 10`,
  `minimize_size = 0`, `allow_mixed = 0`, `exact = 0`, `alpha_quality = 100`, `pass = 1`,
  `target_size = 0`. A lossless config keeps quality 75 unless `lossless` sets it.
- Timing: each frame's duration is its delay in ms, where a delay of 10 ms or less plays as 100
  (libvips' rule); timestamps accumulate; the final `Add(NULL, timestamp)` closes the last frame.
- Container: `VP8X` + `ANIM` (background colour and loop count; the loop count is set through
  the muxer afterwards) + `ANMF` frames, with ICCP, EXIF and XMP chunks when given.
- Errors map to the engine's codes; messages start with `webp:` and never name libwebp/libvips.

## How WebPAnimEncoder decides (reference behaviour; read libwebp `src/mux/anim_encode.c`)

Per added frame:
1. Compare with the previous canvas and find the changed rectangle (snapped to even x/y, as the
   format requires). An identical frame adds its duration to the previous frame instead of
   emitting one.
2. Candidates: sub-frame vs. full frame; previous frame disposed to background vs. not; blend
   vs. no-blend. For blending, pixels equal to the previous canvas are made transparent
   (lossless: exactly equal; lossy: within a quality-dependent threshold, and similar blocks are
   flattened) so they cost almost nothing.
3. Each candidate is encoded with the frame config and the smallest kept (with
   `minimize_size = 0` only the cheaper candidate set is tried; `allow_mixed = 0` means no
   lossy/lossless mixing).
4. Keyframes: after `kmin` frames since the last keyframe a full keyframe is also tried, at
   `kmax` it is forced. With `kmin = 9`, `kmax = 10` that is roughly every 10th frame.
5. Encoded frames are kept until `Assemble`, which writes the container.

## What to build

- An `AnimationEncoder` (name it in the crate's style) that takes frames one at a time
  (rows or a frame buffer), options as above, and writes the container. Reuse the crate's
  `vp8enc` and `vp8l` frame encoders and its chunk/RIFF writers (`encoder.rs`, `extended.rs`,
  `demux.rs` on branch `probe-demux` for the chunk layout); don't duplicate them.
- Decide sub-rectangles, dispose/blend and keyframes with your own design. Start simple
  (changed rectangle, blend vs. no-blend, keyframe interval like kmin/kmax) and add libwebp's
  refinements only where the bench shows the gate missed, one at a time, measured.
- Near-lossless frames need the near-lossless preprocessing, which this crate does not have
  yet (it is a separate task; design the encoder so it can plug in).
- Memory: hold only what the decisions need (previous canvas, current frame, the candidates in
  flight). Do not keep every encoded frame in memory if avoidable: the output goes to a sink
  (image-processing spools it), so frames can be written as they are decided and the RIFF/VP8X
  sizes fixed at the end (write to a `Write + Seek`, or return the header to patch). Report peak
  memory against libwebp.
- Speed: use the crate's SIMD paths and conventions; parallelise candidate encodes if it pays.

## Gates (Marc's, image-processing `docs/engine.md` and memory `feedback-quality-and-memory-not-byte-identity`)

Compared with libwebp's `WebPAnimEncoder` at the options above, on the same frames:
- Lossless: every decoded frame (composited canvas) pixel-identical.
- Lossy: per composited frame PSNR ≥ libwebp's − 0.5 dB, both against the source frame; total
  file size ≤ +5%.
- Frame count, durations, loop count, background, metadata chunks: equal in meaning (a frame
  merged for being identical must keep the total duration).
- Peak RSS ≤ libwebp's for the same animation; time reported (aim for faster).
- Decodes with libwebp and with this crate's decoder identically.

## How to measure

- A bench/example in this crate that encodes an animation with both (libwebp through
  `libwebp-sys` as a dev-dependency) and prints per-frame dB, total bytes, time and peak RSS,
  each side in its own process.
- Inputs: synthetic animations (moving shapes, small changes, full-frame noise, alpha fades,
  identical frames, 1-frame, 2×2, 16383 wide), real GIF/WebP animations decoded to RGBA (on
  `worker`: the corpora image-processing's benches use, see its `.claude/scratchpad/*learnings.md`
  and memory `worker-bench-setup`), at quality 30/75/90, effort 0/4/6, lossy, lossless.
- Heavy runs go through the queue on `worker` (docker), never on the laptop; see
  image-processing's CLAUDE.md and `.claude/scratchpad/brief-common-2.md`.

## Integration (after the gates hold)

In image-processing, `src/engine/webp_anim.rs` `Encoder` switches to this encoder (fork pinned
by git `rev`; Marc pushes the fork branch), the `WebPAnimEncoder`/`WebPMux` calls go, and
`test/engine-anim.test.js` / `test/webp.test.js` hold the new output (literals through
`test/parity.js`). Work on a branch of this repo (e.g. `anim-encoder`, from the newest fork
branch then in use); never push.
