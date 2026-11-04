/* fuzz.c - tenxel_classify against lex_token alone, on generated inputs.
 *
 * lex_token is the whole definition of the output, and oracle.py checks it against clang. This checks that the block
 * paths agree with it, much faster than clang can. Inputs sit flush against unmapped guard pages, so a read or write
 * out of bounds faults. On a mismatch the input is saved as fuzz_fail.bin.
 *
 * Usage: fuzz SEED BYTES [corpus.c]
 */
#define main tenxel_main /* this file has its own */
#include "tenxel.c"
#undef main
#include <sys/mman.h>
#include <unistd.h>

static u64 state[4];

static u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }

static u64 rnd(void) { /* xoshiro256** */
    u64 r = rotl(state[1] * 5, 7) * 9;
    u64 t = state[1] << 17;
    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotl(state[3], 45);
    return r;
}

static size_t rn(size_t n) { return n ? rnd() % n : 0; }

static const char *PIECES[] = {
    "0",
    "1",
    "42",
    "0x1F",
    "0x1p-3",
    "1e+5",
    "1E-5",
    "1.5f",
    ".5",
    "1.",
    "1..2",
    "...",
    "..",
    "..5",
    "...5",
    "1'000",
    "1'a",
    "1'_",
    "1'$",
    "0x1e+5",
    "1e+e+5",
    "1'e+5",
    "1\\\n2",
    "1e\\\n+5",
    "1'\\\n0",
    ".\\\n5",
    "x.5",
    "a..5",
    "\"s\"",
    "\"a\\\"b\"",
    "\"a\\\\\"",
    "\"unterminated",
    "\"a\\\nb\"",
    "\"a\\\\\nb\"",
    "'c'",
    "'\\''",
    "''",
    "'",
    "'ab'",
    "don't",
    "L\"w\"",
    "u8\"s\"",
    "u'c'",
    "U'c'",
    "u8'c'",
    "L'c'",
    "u\\\n8\"s\"",
    "R\"(raw)\"",
    "R\"x(a)\"b)x\"",
    "R\"(un",
    "R\"bad delim(x)\"",
    "u8R\"(z)\"",
    "LR\"-(q)-\"",
    "R\"(a\\\nb)\"",
    "R\"\\\n(x)\"",
    "//c",
    "// c \\\n still",
    "//\\\n",
    "/**/",
    "/* c */",
    "/*/ x */",
    "/* a\nb */",
    "/***/",
    "/* open",
    "*/",
    "/\\\n* c *\\\n/",
    "/\\\n/ c",
    "/* *\\ \n/",
    "/* *\\\n\\\n/",
    "#",
    "##",
    "%:",
    "%:%:",
    "%:%",
    "#define X 1",
    "# include <a.h>",
    "#if 0",
    "#x\\\ny",
    "#define S \"a\" L'b' 1'0 u8\"c\" R\"(d)\" /* e\n f */ g",
    "<:",
    ":>",
    "<%",
    "%>",
    "%=",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ";",
    ",",
    "->",
    "+",
    "-",
    "*",
    "/",
    "=",
    "<",
    ">",
    "!",
    "~",
    "?",
    ":",
    "&",
    "|",
    "^",
    "%",
    "@",
    "`",
    "\\",
    "\\u00e9",
    "\\u03b1x",
    "\\u0041",
    "\\u00a0",
    "\\ud800",
    "\\U0001F600",
    "\\U00110000",
    "\\u{e9}",
    "\\u{3b1}",
    "\\U{e9}",
    "\\u{}",
    "\\u12",
    "\\N{X}",
    "\\N{}",
    "\\N{A'B}",
    "\\x",
    "\xc3\xa9",
    "\xce\xb1",
    "\xe2\x86\x92",
    "\xf0\x9f\x98\x80",
    "\xc2\xb7",
    "\xc2\xa0",
    "\xe3\x80\x80",
    "\xe2\x88\x82",
    "\xff",
    "\xc0\xaf",
    "\xed\xa0\x80",
    "\xe2\x86",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\r",
    "\n\r",
    "\f",
    "\v",
    "\\\n",
    "\\ \n",
    "\\\r\n",
    "\\\t\r",
    "\x1a",
    "\x7f",
    "$",
    "a$",
    "$a",
    "x",
    "foo",
    "foo(",
    "foo (",
    "foo\n(",
    "foo /**/ (",
    "foo // c\n(",
    "size_t",
    "NULL",
    "intx",
    "__typeof_unqual_",
    "__builtin_va_argX",
    "in\\\nt",
    "_Static_assertXYZ",
};
#define N_PIECES (sizeof PIECES / sizeof *PIECES)
static const u8 HOT[] = {'/', '*', '"', '\'', '\\', '\n', '#', '(', ' ', '.',  'a',  '0', '_', '\t', '\r',
                         'e', '+', '%', ':',  'u',  'L',  'R', '8', 0,   0x80, 0xff, '$', '{', '}',  'x'};

static const u8 *corpus;
static size_t corpus_len;

/* One piece: a keyword, or one of PIECES */
static const char *piece(size_t *len) {
    if (rn(3) == 0) {
        const Keyword *k;
        do
            k = &KEYWORDS[rn(KEYWORD_SLOTS)];
        while (!k->word[0]);
        *len = strlen(k->word);
        return k->word;
    }
    const char *t = PIECES[rn(N_PIECES)];
    *len = strlen(t);
    return t;
}

static size_t emit_piece(u8 *p, size_t room) {
    size_t len;
    const char *t = piece(&len);
    if (rn(8) == 0) { /* a NUL */
        t = "";
        len = 1;
    }
    if (len > room)
        len = room;
    memcpy(p, t, len);
    return len;
}

/* A long run of one short unit, crossing many blocks */
static size_t run_of(u8 *p, size_t room) {
    static const char *UNITS[] = {"a", "0",  " ",  "\n", "/",  "*",  "\"", "'",    "\\", "#",  "(",
                                  ".", "ab", "a ", "a(", "//", "/*", "*/", "\\\n", "1.", "e+", "%:"};
    const char *u = UNITS[rn(sizeof UNITS / sizeof *UNITS)];
    size_t len = strlen(u);
    size_t n = (1 + rn(rn(4) ? 80 : 400)) * len;
    if (n > room)
        n = room;
    for (size_t i = 0; i < n; i++)
        p[i] = (u8)u[i % len];
    return n;
}

static size_t generate(u8 *b, size_t cap, int kind) {
    size_t n = 0;
    size_t want = rn(rn(2) ? 300 : 3000);
    if (rn(4) == 0)
        want = rn(70);
    else if (rn(8) == 0)
        want = rn(cap);
    if (want > cap)
        want = cap;

    switch (kind) {
    case 0:
        for (; n < want; n++)
            b[n] = (u8)rnd();
        break;
    case 1:
        for (; n < want; n++)
            b[n] = HOT[rn(sizeof HOT)];
        break;
    case 2:
        while (n < want)
            n += emit_piece(b + n, want - n);
        break;
    case 3:
        while (n < want)
            n += rn(3) ? emit_piece(b + n, want - n) : run_of(b + n, want - n);
        break;
    case 4: /* pieces that straddle 64-byte blocks */
        while (n < want) {
            size_t to = 64 - n % 64;
            size_t pad = to > 20 ? to - 1 - rn(19) : 0;
            if (pad > want - n)
                pad = want - n;
            memset(b + n, " a0.;\n("[rn(7)], pad);
            n += pad;
            for (int i = 0; i < 2 && n < want; i++)
                n += emit_piece(b + n, want - n);
        }
        break;
    default: /* real code, mutated */
        if (!corpus_len)
            return generate(b, cap, 2);
        if (want > corpus_len)
            want = corpus_len;
        memcpy(b, corpus + rn(corpus_len - want + 1), want);
        n = want;
        for (size_t m = kind == 5 ? rn(4) : rn(1 + want / 8); m && n; m--) {
            size_t at = rn(n);
            size_t len;
            const char *t = piece(&len);
            if (rn(2) && n + len < cap) {
                memmove(b + at + len, b + at, n - at);
                memcpy(b + at, t, len);
                n += len;
            } else {
                b[at] = HOT[rn(sizeof HOT)];
            }
        }
        break;
    }
    return n;
}

/* lex_token alone, the way tenxel_classify starts */
static void classify_plain(const u8 *src, size_t n, u8 *cls) {
    Lexer L = {src, src + n, cls, true, false};
    const u8 *p = src;
    if (n >= 3 && !memcmp(src, "\xEF\xBB\xBF", 3)) {
        memset(cls, C_SPACE, 3);
        p += 3;
    }
    while (p < L.end)
        p = lex_token(&L, p);
}

static long page;

/* cap bytes between two unmapped pages */
static u8 *guarded(size_t cap) {
    size_t size = (cap + page - 1) / page * page;
    u8 *m = mmap(0, size + 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        perror("mmap");
        exit(2);
    }
    mprotect(m, page, PROT_NONE);
    mprotect(m + page + size, page, PROT_NONE);
    return m + page;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: fuzz SEED BYTES [corpus.c]\n");
        return 2;
    }
    u64 seed = strtoull(argv[1], 0, 0);
    size_t budget = strtoull(argv[2], 0, 0);
    for (int i = 0; i < 4; i++) { /* splitmix64 */
        seed += 0x9E3779B97F4A7C15ULL;
        u64 z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        state[i] = z ^ (z >> 31);
    }
    if (argc > 3) {
        FILE *f = fopen(argv[3], "rb");
        if (!f) {
            perror(argv[3]);
            return 2;
        }
        fseek(f, 0, SEEK_END);
        corpus_len = (size_t)ftell(f);
        fseek(f, 0, SEEK_SET);
        u8 *c = malloc(corpus_len);
        if (fread(c, 1, corpus_len, f) != corpus_len)
            return 2;
        fclose(f);
        corpus = c;
    }

    enum { CAP = 1 << 17 };
    page = sysconf(_SC_PAGESIZE);
    size_t size = (CAP + page - 1) / page * page;
    u8 *in = guarded(CAP);
    u8 *want = guarded(CAP);
    u8 *got = guarded(CAP);
    static u8 tmp[CAP];
    unsigned long cases = 0;
    size_t bytes = 0;
    while (bytes < budget) {
        size_t n = generate(tmp, CAP, (int)rn(7));
        u8 *s = rn(2) ? in : in + size - n; /* flush against the leading or the trailing guard page */
        memcpy(s, tmp, n);
        classify_plain(s, n, want + size - n);
        tenxel_classify(s, n, got + size - n);
        if (memcmp(want + size - n, got + size - n, n)) {
            size_t i = 0;
            while (want[size - n + i] == got[size - n + i])
                i++;
            printf("MISMATCH n=%zu at %zu: lex_token says %d, tenxel_classify %d\n", n, i, want[size - n + i],
                   got[size - n + i]);
            FILE *f = fopen("fuzz_fail.bin", "wb");
            fwrite(s, 1, n, f);
            fclose(f);
            return 1;
        }
        bytes += n;
        cases++;
    }
    printf("ok: fuzz seed %s: %lu inputs, %zu bytes, block paths match lex_token\n", argv[1], cases, bytes);
    return 0;
}
