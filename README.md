# tenxel

A C lexer for syntax highlighting in one C file. It gives every byte one of 11 classes, with the same token boundaries as clang, at about 0.6 ns per byte.

It started as [lexnet](https://huggingface.co/alpindale/lexnet), a hand-built neural network that highlights C, rewritten as plain C. That version matched the network byte for byte, including its mistakes, and is kept at the tag `lexnet-exact`.

| File | What it is |
|---|---|
| `tenxel.c` | The lexer and its command line |
| `oracle.py` | The reference: the class of every byte according to clang's lexer, and a checker for `tenxel` |
| `fuzz.c`, `exhaust.c` | Check the SIMD paths against `lex_token`, the one-token-at-a-time lexer inside `tenxel.c` |
| `verify.sh` | Runs all the checks |

## Use

```sh
cc -O3 -march=native -o tenxel tenxel.c
./tenxel file.c    # the file, colored
```

From a program, call `tenxel_classify(src, n, cls)`, which writes one class per byte into `cls`. It is the whole interface. Compile `tenxel.c` into the program with its `main` renamed: `cc -c -O3 -march=native -Dmain=tenxel_main tenxel.c`.

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
| 9 preprocessor | A directive, from `#` or `%:` to the end of its line, except its comments and literals |
| 10 punctuation | Everything else |

## What it matches

Tokens are those of clang 19's raw lexer with `-std=gnu23` on x86-64 Linux, so:

- A backslash, optional spaces and a newline join lines anywhere: in strings, comments, identifiers and between `/` and `*`.
- `1e+5`, `0x1p-3`, `.5` and `1'000` are one number each; `...5` is `...` and `5`.
- `%:` starts a directive. A comment before `#` doesn't stop it from being one, and a multi-line comment inside one doesn't end it.
- Identifiers take `$`, UTF-8 and universal character names, with clang's Unicode tables.
- Unterminated literals and comments keep their class up to where clang ends them.

Keywords are the 134 words clang treats as keywords in that mode. A directive is a `#` or `%:` with only spaces or comments before it on its line, as clang's preprocessor decides.

Named universal character names (`\N{LATIN SMALL LETTER A}`) are the one known difference: clang looks the name up, and `tenxel` always treats it as invalid.

## Performance

Measured on an Intel i7-13800H with AVX2, capped at 2.9 GHz, on 715 KB of C (the original lexnet's C files and this repo's earlier ones, repeated), best of 40 calls of `tenxel_classify` on one performance core.

| Lexer | ns/byte |
|---|---|
| `tenxel` | 0.57 |
| [stb_c_lexer](https://github.com/nothings/stb/blob/master/stb_c_lexer.h) | 1.43 |
| The lexnet network, its compiled kernel with gcc `-O3 -march=native -ffast-math` | 79 |
| tree-sitter-c, parse only | 114 |
| tree-sitter-c, parse and highlight query | 229 |
| Pygments `CLexer` | 965 |

stb_c_lexer converts numbers and copies strings, which `tenxel` doesn't need to do; tree-sitter builds a full syntax tree. The network gives the same output with and without `-ffast-math`, and takes 507 ns/byte without it. On one large block comment `tenxel` takes 0.04 ns/byte, and on its slowest pattern found, `a a a …`, 1.3 ns/byte. The portable fallback, used without AVX2, takes 10 ns/byte.

On the 229 headers in `/usr/include`, outside directives, stb_c_lexer's tokens agree with clang's on 99.993% of the bytes and `tenxel`'s on all of them. stb_c_lexer's misses there are mostly multi-line comments that start on a directive line. It also has no line splices, string prefixes or raw strings, and doesn't tell a directive from other code.

## How far to trust it

- `lex_token`, which defines the output, is checked against clang on generated inputs: real headers, mutated code, and token soup made of every case above. The keyword table is checked against clang too.
- The SIMD paths are checked against `lex_token`: hundreds of MB of fuzzing with inputs flush against unmapped pages, and every string up to 5 bytes over the bytes that matter, placed across a block boundary.

`sh verify.sh [fuzz_bytes] [oracle_megabytes]` runs everything. It needs clang 19 and python3, plus `uvx` for the layout check, which uses clang-format 21 with the style written in `verify.sh`.

## License

Apache-2.0, the same as [lexnet](https://huggingface.co/alpindale/lexnet) by alpindale.
