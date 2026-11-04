# tenxel

A C lexer for syntax highlighting, in one C file. It gives every byte one of 11 classes, with clang's token boundaries, at about 0.6 ns per byte on AVX2.

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

Measured on an Intel i7-13800H (AVX2, capped at 2.9 GHz), on one core, best of up to 40 runs. The inputs are 715 KB of C (lexnet's C files and earlier code from this repo, repeated) and SQLite 3.54.1's `sqlite3.c` (9.6 MB). On `sqlite3.c`, tenxel matches clang on every byte.

| Lexer | 715 KB of C, ns/byte | `sqlite3.c`, ns/byte |
|---|---|---|
| tenxel | 0.56 | 0.62 |
| [simd-lexing](https://github.com/Mateuszd6/simd-lexing), counting tokens only | 1.26 | 0.78 |
| [stb_c_lexer](https://github.com/nothings/stb/blob/master/stb_c_lexer.h) | 1.43 | 1.57 |
| [buster](https://github.com/buster14a/buster)'s C lexer, scalar path | 2.73 | 1.81 |
| clang 19's lexer, `clang_tokenize` from libclang | 5.5 | 3.58 |
| lexnet, its compiled kernel with gcc `-O3 -march=native -ffast-math` | 79 | |
| tree-sitter-c, parse only | 114 | |
| tree-sitter-c, parse and highlight query | 229 | |
| Pygments `CLexer` | 965 | |

The others do different work:

- stb_c_lexer, simd-lexing, buster and clang produce tokens, not a class per byte. stb_c_lexer also converts numbers and copies strings, and clang records each token's location.
- simd-lexing has no line splices outside literals and comments, no string prefixes and no UTF-8 identifiers.
- buster's fast path needs AVX-512, which this CPU lacks. On a Zen 4 laptop its authors report 2.0 cycles/byte on `sqlite3.c`; tenxel takes 1.8 here.
- tree-sitter builds a full syntax tree.
- lexnet gives the same output without `-ffast-math`, at 507 ns/byte.

On the 229 headers in `/usr/include`, outside directives, stb_c_lexer agrees with clang on 99.993% of the bytes and tenxel on all of them. Most of stb_c_lexer's misses are multi-line comments that start on a directive line. It also has no line splices, string prefixes or raw strings, and doesn't recognize directives.

tenxel's other speeds: 0.04 ns/byte on one large block comment, 1.3 on its slowest pattern found (`a a a …`), and 10 with the portable code used without AVX2.

## Checks

| File | What it checks |
|---|---|
| `oracle.py` | `tenxel_classify` against clang, on real headers, mutated code and generated token soup; also the keyword table |
| `fuzz.c` | The SIMD path against `lex_token`, the one-token-at-a-time lexer in `tenxel.c`, on hundreds of MB of generated input placed against unmapped pages |
| `exhaust.c` | The same, on every string up to 5 bytes over the bytes that matter, placed across a 64-byte block boundary |
| `verify.sh` | Runs all three, and checks the layout |

`sh verify.sh [fuzz_bytes] [oracle_megabytes]` needs clang 19, python3, and `uvx` for the layout check, which uses clang-format 21 with the style in `verify.sh`.

## License

Apache-2.0, the same as lexnet.
