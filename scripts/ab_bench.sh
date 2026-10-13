#!/bin/bash
# Interleaved best-of-N timing of two lossy_bench binaries (A = baseline, B = this tree).
set -u
A=$1; B=$2; N=${N:-5}
for i in $(seq $N); do
  echo "A $($A ours tests/reference/gallery1/3.png 6000x4000 80 4)"
  echo "B $($B ours tests/reference/gallery1/3.png 6000x4000 80 4)"
done | awk '{t=$(NF-1); k=$1; if (!(k in best) || t < best[k]) best[k]=t; print} END {printf "best: A %d ms, B %d ms, B/A %.3f\n", best["A"], best["B"], best["B"]/best["A"]}'
