# Performance round 2026-10-23: lossy decode and encode

Three builders in parallel off main c8b19b9. Each A/B is an interleaved run, taking the min of 5 cycles:u on the
x86 worker (pinned, 1 slot) or wall time on the M3. Every number below was measured twice with disjoint ranges.
Each builder's result then went to one refuter.

| Branch | Predicted | Result |
|---|---|---|
| recon: luma 4x4 workspace, i16 blocks, transforms clearing their blocks | x86 1-3% | x86 −5.3/−4.1/−8.3% (gallery1/3, gallery1/4, 15 MP photo) |
| modes: row read-ahead with hardcoded trees | x86 ~0, M3 ≤1.5%, 40% nothing kept | x86 +0.3/−0.1/−1.3%, M3 −4.5/−2.6/−3.9% |
| enc: 4x4 search in registers, last non-zero by mask | ~1-2% | x86 −5.6/−5.7/−7.1%, RGBA −2.6%; M3 NEON mask −2.5..−3.2% more |

These results are on top of c8b19b9:

- **Decode, x86:** 0.949, 0.959 and 0.904 of the base.
- **Decode, M3:** 0.944, 0.949 and 0.896 of the base. Against libwebp the M3 is now at 0.933, 0.949 and 0.978, so it is faster on all three files.
- **Encode, M3:** 3.png takes 397 ms here and 411 ms with libwebp. The RGBA file takes 341 ms here and 376 ms with libwebp.

Verification:

- **Suites on the combined chain:** x86 117 and aarch64 114.
- **Refuters:**
  - The decode refuters ran about 750k corrupted and mutated inputs against libwebp and the parent. They found 0 regressions and one pre-existing panic, an empty token partition. It is fixed in cfffb31.
  - The encode refuter ran about 354k encodes against libwebp and the base, all byte-identical. It also checked the last non-zero test exhaustively over 2^16 patterns.

Surprise: moderate, on the upside. Recon and the encoder search each gave 2-3x their prediction.

Open items:

- **Partition-size check parity.** If a bit flip lands in a partition-size entry, we still decode the file and libwebp rejects it.
- **aarch64-only divergence.** Our decoder and libwebp diverge on one variant of tag-moves-table.webp in the `decodes_as_libwebp` sweep. The fixture was left out.

Dropped variants:

- `#[inline(never)]` on large_value: +6-7%.
- Renormalisation shift as `7 ^ ilog2`: +3-7%.
