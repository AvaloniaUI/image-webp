#!/bin/bash
# Kernel tests, byte parity against libwebp, then 24 MP timing of both encoders.
set -u
if ! cargo build -q --release --examples 2>build.log; then grep -E '^error' -A8 build.log | head -40; exit 1; fi
cargo test -q --release --lib 2>&1 | grep -E "test result|panicked|FAILED" | head -5
for m in ${METHODS:-0 4 6}; do
  ./target/release/examples/lossy_vs_libwebp --synthetic -q 0,50,80,95,100 -m "$m" tests/reference/gallery1/*.png tests/reference/gallery2/*_a.png | tail -1
done
grep Cpus_allowed_list /proc/self/status 2>/dev/null
for which in libwebp ours libwebp ours libwebp ours; do
  ./target/release/examples/lossy_bench "$which" tests/reference/gallery1/3.png 6000x4000 80 4
done | awk '{t=$(NF-1); k=$1; if (!(k in best) || t < best[k]) best[k]=t; print} END {printf "best: libwebp %d ms, ours %d ms, ratio %.2f\n", best["libwebp"], best["ours"], best["ours"]/best["libwebp"]}'
if [ -n "${PROFILE:-}" ]; then
  CARGO_PROFILE_RELEASE_DEBUG=true cargo build -q --release --example lossy_bench
  perf record -g -o target/perf.data ./target/release/examples/lossy_bench ours tests/reference/gallery1/3.png 3000x2000 80 4 >/dev/null 2>&1
  perf report -i target/perf.data --no-children --percent-limit 1 --stdio 2>/dev/null | grep -E "^ +[0-9]+\.[0-9]+%" | sed -E 's/_R[A-Za-z0-9_]*lossy7encoder//' | head -25
fi
if [ -n "${MEM:-}" ]; then
  for which in libwebp ours ours-rows; do
    if [ "$(uname)" = Linux ]; then
      /usr/bin/time -f "$which peak RSS %M KB" ./target/release/examples/lossy_bench "$which" tests/reference/gallery1/3.png 6000x4000 80 4 2>&1 | grep -E "peak|B in"
    else
      /usr/bin/time -l ./target/release/examples/lossy_bench "$which" tests/reference/gallery1/3.png 6000x4000 80 4 2>&1 | grep -E "maximum resident|B in"
    fi
  done
  echo "(libwebp and ours hold the 72000 KB RGB input; ours-rows generates rows on demand)"
fi
