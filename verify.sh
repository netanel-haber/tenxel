#!/bin/sh
# Checks tenxel.c. Run from the repo root. Needs a C compiler, clang 19 and python3; uvx for the layout check.
#   sh verify.sh [fuzz_bytes] [oracle_megabytes]     default 300000000 and 20
# A failing input is saved as fuzz_fail.bin or oracle_fail.c.
set -e
CC=${CC:-cc}; BYTES=${1:-300000000}; MB=${2:-20}; OUT=$(mktemp -d); trap 'rm -rf "$OUT"' EXIT
FLAGS="-O2 -Wall -Wextra -Werror"
STYLE='{BasedOnStyle: LLVM, IndentWidth: 4, ColumnLimit: 120, AllowShortIfStatementsOnASingleLine: Never,
AllowShortLoopsOnASingleLine: false, AllowShortBlocksOnASingleLine: Never, SortIncludes: Never, ReflowComments: Never}'

if command -v uvx > /dev/null; then
    uvx --quiet --from clang-format==21.1.8 clang-format --style="$STYLE" --dry-run --Werror tenxel.c fuzz.c exhaust.c bench.c
    echo "ok: layout"
fi
for path in native portable; do
    arch=$([ $path = native ] && echo -march=native || true)
    $CC $FLAGS $arch -shared -fPIC -o "$OUT/tenxel_$path.so" tenxel.c
    $CC $FLAGS $arch -o "$OUT/fuzz_$path" fuzz.c
done
$CC $FLAGS -march=native -o "$OUT/tenxel" tenxel.c
$CC $FLAGS -march=native -o "$OUT/exhaust" exhaust.c
$CC $FLAGS -march=native -o "$OUT/bench" bench.c
cat tenxel.c /usr/include/*.h > "$OUT/corpus.c" 2> /dev/null || cp tenxel.c "$OUT/corpus.c"

"$OUT/tenxel" tenxel.c | sed 's/\x1b\[[0-9;]*m//g' | cmp - tenxel.c
echo "ok: colored output is the input plus escape codes"
python3 oracle.py --check "$OUT/tenxel_native.so" "$MB"
python3 oracle.py --check "$OUT/tenxel_portable.so" $((MB / 4 + 1)) 2
"$OUT/exhaust" 5 2> /dev/null
"$OUT/fuzz_native" 1 "$BYTES" "$OUT/corpus.c"
"$OUT/fuzz_portable" 2 $((BYTES / 4)) "$OUT/corpus.c"
echo "all checks passed"
