# tenxel

A C lexer for syntax highlighting, in one C file. It gives every byte one of 11 classes, with clang's token boundaries, in about 3 CPU cycles per byte with AVX2.

## Credit

tenxel started as a sort of port of [lexnet](https://huggingface.co/alpindale/lexnet), a hand-built neural network by alpindale that highlights C. The first version rewrote the network as plain C and matched it byte for byte, mistakes included; it is kept at the tag `lexnet-exact`. tenxel has since moved to clang's rules and a different design, but the idea of classifying every byte, the 11 classes and the name (lexnet backwards) come from lexnet. tenxel is an independent project, not affiliated with lexnet or alpindale.

## Use

```sh
cc -O3 -march=native -o tenxel tenxel.c
./tenxel file.c    # prints the file, colored
```

From a program, call `tenxel_classify(src, n, cls)`. It writes one class per byte into `cls`, and is the whole interface. To build `tenxel.c` into a program, rename its `main`: `cc -c -O3 -march=native -Dmain=tenxel_main tenxel.c`.

| Class | Bytes |
|---|---|
| 0 space | Spaces, newlines and NULs, and line splices between tokens |
| 1 keyword | clang's keywords, such as `if`, `typeof`, `__attribute__`, `static_assert` |
| 2 type | Builtin type keywords, such as `int`, `bool`, `_BitInt`, `__int128` |
| 3 function | An identifier whose next token, past spaces and comments, is `(` |
| 4 identifier | Other identifiers, including `size_t`, `FILE` and `NULL` |
| 5 constant | Numbers, and `true`, `false`, `nullptr` |
| 6 string, 7 char | Literals with their prefixes (`u8"x"`, `L'x'`), and raw strings |
| 8 comment | |
| 9 preprocessor | A directive, except its comments and literals |
| 10 punctuation | Everything else |

## What it matches

Tokens are those of clang 19's raw lexer with `-std=gnu23` on x86-64 Linux:

- A backslash, optional spaces and a newline join lines anywhere, even inside an identifier or between `/` and `*`.
- `1e+5`, `0x1p-3`, `.5` and `1'000` are one number each; `...5` is `...` then `5`.
- Identifiers can hold `$`, UTF-8 and universal character names, by clang's Unicode tables.
- Unterminated literals and comments keep their class up to where clang ends them.
- Keywords are the 134 words clang treats as keywords in that mode.
- A directive starts at a `#` or `%:` with only spaces or comments before it on its line, and ends at the next newline outside a comment.

The one known difference is named universal character names (`\N{LATIN SMALL LETTER A}`): clang looks the name up, and tenxel always treats it as invalid.

## Performance

`bench.c` measures it on any file, in ns, CPU cycles and instructions per byte. Cycles and instructions come from the CPU's counters, so they don't depend on the clock speed:

```sh
cc -O3 -march=native -o bench bench.c
taskset -c 2 ./bench file.c
```

On an Intel i7-13800H (Raptor Lake, AVX2), on one core, the fastest of 40 runs:

| Input | Cycles/byte | Instructions/byte | ns/byte at 4.95 GHz |
|---|---|---|---|
| 715 KB of C (lexnet's C files and earlier code from this repo, repeated) | 2.9 | 11.1 | 0.57 |
| SQLite 3.54.1's `sqlite3.c`, 9.6 MB | 3.2 | 7.6 | 0.64 |

On `sqlite3.c`, tenxel matches clang on every byte. On one large block comment it takes 0.3 cycles/byte, and on its slowest pattern found, `a a a …`, 6. Without AVX2 it uses portable code, which takes 43 cycles/byte on the 715 KB of C.

Other lexers on the same machine, as multiples of tenxel's time:

| Lexer | 715 KB of C | `sqlite3.c` |
|---|---|---|
| [simd-lexing](https://github.com/Mateuszd6/simd-lexing), counting tokens only | 2.2× | 1.3× |
| [stb_c_lexer](https://github.com/nothings/stb/blob/master/stb_c_lexer.h) | 2.6× | 2.5× |
| [buster](https://github.com/buster14a/buster)'s C lexer, scalar path | 4.9× | 2.9× |
| clang 19's lexer, `clang_tokenize` from libclang | 9.8× | 5.8× |
| lexnet, its compiled kernel with gcc `-O3 -march=native -ffast-math` | 140× | |
| tree-sitter-c, parse only | 200× | |
| tree-sitter-c, parse and highlight query | 410× | |
| Pygments `CLexer` | 1,700× | |

The others do different work:

- stb_c_lexer, simd-lexing, buster and clang produce tokens, not a class per byte. stb_c_lexer also converts numbers and copies strings, and clang records each token's location.
- simd-lexing has no line splices outside literals and comments, no string prefixes and no UTF-8 identifiers.
- buster's fast path needs AVX-512, which this CPU lacks. On a Zen 4 laptop with it, its authors report 2.0 cycles/byte on `sqlite3.c`, fewer than tenxel's 3.2 here.
- tree-sitter builds a full syntax tree.
- lexnet gives the same output without `-ffast-math`, 6.4 times slower.

On the 229 headers in `/usr/include`, outside directives, stb_c_lexer agrees with clang on 99.993% of the bytes and tenxel on all of them. Most of stb_c_lexer's misses are multi-line comments that start on a directive line. It also has no line splices, string prefixes or raw strings, and doesn't recognize directives.

## Checks

`sh verify.sh [fuzz_bytes] [oracle_megabytes]` checks tenxel against clang with `oracle.py`, and its SIMD path against `lex_token`, its one-token-at-a-time lexer, with `fuzz.c` and `exhaust.c`. It needs clang 19, python3, and `uvx` for the layout check, which uses clang-format 21 with the style in `verify.sh`.

## License

Apache-2.0, the same as lexnet.
