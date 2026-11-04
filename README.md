# lexnet-c

[Lexnet](https://huggingface.co/alpindale/lexnet) is a neural network that syntax-highlights C. Its weights turn out to encode an ordinary C lexer, so this repo writes that lexer out as plain C, then optimizes it. The original repo is included as the `lexnet_hf` submodule, pinned to commit `c59fe64`.

| File | What it is |
|---|---|
| `lexnet_ifelse.c` | The network as readable if/else C, with no weights or floats |
| `lexnet_fast.c` | The optimized version: SIMD over 64-byte blocks, with a perfect hash for words |
| `net2c.py` | Regenerates both C files from the weights, checking hashes before and after |
| `verify.sh` | Runs the equivalence checks: `prove.c`, `prove_words.c`, `exhaust.c` and `fuzz.c` |

## Setup

The weights are stored in Git LFS, so install it before cloning.

```sh
git lfs install
git clone --recursive https://github.com/netanel-haber/lexnet-c
cd lexnet-c
```

## Use

```sh
cc -O3 -march=native -o lexnet_fast lexnet_fast.c
./lexnet_fast file.c              # colored output
./lexnet_fast --classes file.c    # one class digit per byte
./lexnet_fast --bench 100 file.c

uv run net2c.py                   # regenerate the C files
sh verify.sh                      # check them against the network
```

`net2c.py` reads the word list and its classes from the weights and rebuilds the perfect hash. The C control flow is a template inside the script, because it was written by hand from the gates. The script checks the real weights against that logic and refuses to write anything if they disagree.

## Performance

Measured on a shared 2-core 2.1 GHz Xeon VM with AVX-512, best of N on both sides. The best original build used `-O3 -march=native -ffast-math`, with gcc PGO on three inputs and clang on the stripped corpus.

| Input | Best original | `lexnet_fast` | Speedup |
|---|---|---|---|
| Lua source, 1.0 MB | 138.5 ns/char | 1.10 ns/char | 126x |
| lua, curl, git, redis and sqlite, 53.5 MB | 149.4 ns/char | 1.05 ns/char | 143x |
| Same corpus with comments stripped, 38.4 MB | 147.5 ns/char | 1.28 ns/char | 116x |
| Random printable text, 8.0 MB | 141.4 ns/char | 0.92 ns/char | 154x |

On the Lua source, `lexnet_fast` takes 1.35 ns/char with AVX2 and 7.7 ns/char on the portable path, and `lexnet_ifelse` takes 15.3 ns/char. The slowest pattern found for `lexnet_fast` was `a a a a…` at 11.2 ns/char.

## How far to trust it

- **`lexnet_ifelse.c`** is checked against the real weights over every reachable state and every keyword threshold. Only the word accumulator values are sampled.
- **`lexnet_fast.c`** is tested, not proven: zero mismatches across the 53.5 MB corpus, 3.3 GB of adversarial fuzzing and exhaustive short strings.
- **The NEON path has not been run.** Run `sh verify.sh` on ARM before relying on it, or build with `-DLEXNET_FORCE_SCALAR` to use the portable path.

## Network quirks kept on purpose

Output matches the network byte for byte, so these stay:

- Only a word's first 14 bytes count, so `_Static_assertXYZ` is classed as a keyword.
- A newline ends a string or char literal even after a backslash, because there's no line splicing.

## License

Apache-2.0, the same as [lexnet](https://huggingface.co/alpindale/lexnet) by alpindale.
