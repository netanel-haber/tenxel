#!/bin/sh
# Checks lexnet_ifelse.c and lexnet_fast.c against the real network. Run from the repo root.
#   sh verify.sh [fuzz_bytes]     default 200000000
# On a mismatch, the failing input is saved as fuzz_fail.bin.
set -e
CC=${CC:-cc}; BYTES=${1:-200000000}; OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
FLAGS="-O3 -march=native -w"
$CC $FLAGS -I. -o "$OUT/prove" prove.c -lm
$CC $FLAGS -I. -o "$OUT/prove_words" prove_words.c -lm
$CC $FLAGS -DLEXNET_NO_MAIN -Dlexnet_classify=ifelse_classify -c -o "$OUT/ie.o" lexnet_ifelse.c
$CC $FLAGS -DLEXNET_NO_MAIN -Dlexnet_classify=fast_classify -c -o "$OUT/fa.o" lexnet_fast.c
$CC $FLAGS -I. -c -o "$OUT/fuzz.o" fuzz.c
$CC $FLAGS -c -o "$OUT/exhaust.o" exhaust.c
$CC -o "$OUT/fuzz" "$OUT/fuzz.o" "$OUT/ie.o" "$OUT/fa.o" -lm
$CC -o "$OUT/exhaust" "$OUT/exhaust.o" "$OUT/ie.o" "$OUT/fa.o"
"$OUT/prove"
"$OUT/prove_words"
"$OUT/exhaust" 5 | tail -1
cat lexnet_hf/lexnet.c lexnet_hf/lexnet_kernel.h lexnet_ifelse.c lexnet_fast.c > "$OUT/seed.c"
"$OUT/fuzz" 1 $((BYTES / 20)) "$BYTES" "$OUT/seed.c"
echo "all checks passed"
