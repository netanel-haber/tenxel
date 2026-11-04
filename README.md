# tenxel

A C lexer for syntax highlighting, in one C file. It gives every byte one of 11 classes, with clang's token boundaries, in about 2.5 to 4.5 CPU cycles per byte with AVX2.

It started as a sort of port of [lexnet](https://huggingface.co/alpindale/lexnet), alpindale's hand-built neural network that highlights C, and takes its idea, classes and name (backwards) from it. The port is at the tag `lexnet-exact`.

## Use

```sh
cc -O3 -march=native -o tenxel tenxel.c
./tenxel file.c    # prints the file, colored
```

From a program: `tenxel_classify(src, n, cls)` writes one class per byte into `cls`. Build it in with `-Dmain=tenxel_main`. The classes are 0 space, 1 keyword, 2 type, 3 function (an identifier before `(`), 4 identifier, 5 constant, 6 string, 7 char, 8 comment, 9 preprocessor and 10 punctuation.

Tokens match clang 19's raw lexer with `-std=gnu23`, including line splices, digraphs, literal prefixes, raw strings and UTF-8 identifiers, and keywords are clang's 134. The one known difference: a named character escape, `\N{...}`, is always treated as invalid.

## Speed

`taskset -c 2 ./bench file.c`, built from `bench.c`, reports cycles and instructions per byte from the CPU's counters. On SQLite's 9.6 MB `sqlite3.c`, on an Intel i7-13800H, with other lexers' time as multiples of tenxel's:

| Lexer | Time |
|---|---|
| tenxel | 2.5 cycles/byte |
| [simd-lexing](https://github.com/Mateuszd6/simd-lexing), counting tokens only | 1.5× |
| [stb_c_lexer](https://github.com/nothings/stb/blob/master/stb_c_lexer.h) | 3.1× |
| clang 19, `clang_tokenize` | 7.1× |

The others produce tokens rather than a class per byte. [buster](https://github.com/buster14a/buster)'s lexer needs AVX-512, which this CPU lacks; its authors report 2.0 cycles/byte on `sqlite3.c`.

## Checks

`sh verify.sh` checks tenxel against clang (`oracle.py`) and its SIMD path against its simple lexer (`fuzz.c`, `exhaust.c`). It needs clang 19, python3 and `uvx`.

## License

Apache-2.0, the same as lexnet.
