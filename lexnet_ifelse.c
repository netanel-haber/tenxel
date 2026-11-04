/* lexnet_ifelse.c - the lexnet "neural network", written as the program it always was.
 *
 * No weights, no floats, no matrices. Same output as the network on every byte of input.
 * It keeps the network's two-pass shape: a left-to-right pass that tracks what we are
 * inside, then a right-to-left pass that handles anything needing to see what comes next.
 *
 * Build: cc -O2 -o lexnet_ifelse lexnet_ifelse.c
 * Usage: lexnet_ifelse [--classes | --bench N] file.c
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum Class {
    C_SPACE = 0, C_KEYWORD = 1, C_TYPE = 2, C_FUNCTION = 3, C_IDENT = 4, C_CONSTANT = 5,
    C_STRING = 6, C_CHAR = 7, C_COMMENT = 8, C_PREPROC = 9, C_PUNCT = 10,
};

/* ---- character tests ---- */
static inline bool is_alpha(unsigned c) { return ((c | 32) - 'a' < 26) || c == '_'; }
static inline bool is_digit(unsigned c) { return c - '0' < 10; }
static inline bool is_ident(unsigned c) { return is_alpha(c) || is_digit(c); }
/* Note: newline is deliberately not "space" here, and bytes >= 128 are nothing at all. */
static inline bool is_space(unsigned c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r'; }

/* ---- the word list ----
 * The network packs a word's first 14 characters into two integers, so characters past the
 * 14th are invisible to it. Only one keyword is 14 long, which makes "_Static_assertXYZ" a
 * keyword. That is a bug in the network and it is preserved here on purpose. */
static int word_kind(const unsigned char *s, size_t n) {
    if (n > 14) n = 14;
#define W(lit) (n == sizeof(lit) - 1 && memcmp(s, lit, n) == 0)
    if (W("do") || W("if") || W("for") || W("auto") || W("case") || W("else") || W("enum") ||
        W("goto") || W("break") || W("const") || W("union") || W("while") || W("extern") ||
        W("inline") || W("return") || W("sizeof") || W("static") || W("struct") || W("switch") ||
        W("typeof") || W("alignas") || W("alignof") || W("default") || W("typedef") ||
        W("_Atomic") || W("continue") || W("register") || W("restrict") || W("volatile") ||
        W("_Alignas") || W("_Alignof") || W("_Generic") || W("_Noreturn") || W("constexpr") ||
        W("_Thread_local") || W("thread_local") || W("static_assert") || W("_Static_assert"))
        return C_KEYWORD;
    if (W("int") || W("bool") || W("char") || W("long") || W("void") || W("FILE") || W("float") ||
        W("off_t") || W("pid_t") || W("short") || W("_Bool") || W("double") || W("int8_t") ||
        W("signed") || W("size_t") || W("time_t") || W("int16_t") || W("int32_t") ||
        W("int64_t") || W("ssize_t") || W("uint8_t") || W("va_list") || W("wchar_t") ||
        W("intptr_t") || W("uint16_t") || W("uint32_t") || W("uint64_t") || W("unsigned") ||
        W("_Complex") || W("ptrdiff_t") || W("uintptr_t"))
        return C_TYPE;
    if (W("EOF") || W("true") || W("NULL") || W("false") || W("nullptr"))
        return C_CONSTANT;
#undef W
    return 0;
}

/* ---- left-to-right pass ---- */
enum Mode { CODE, BLOCK_COMMENT, LINE_COMMENT, STRING, CHAR_LIT };

typedef struct {
    enum Mode mode;
    bool last_star;    /* previous byte was a '*' inside a block comment */
    bool escaped;      /* previous byte was an unescaped backslash inside a literal */
    bool last_slash;   /* previous byte was a '/' that could open a comment */
    bool line_start;   /* only whitespace so far on this line */
    bool preproc;      /* inside a # directive */
    bool last_bs;      /* inside a directive, previous byte was a backslash */
    bool word, number;
} Forward;

enum { F_STRING = 1, F_CHAR = 2, F_COMMENT = 4, F_PREPROC = 8, F_WORD = 16, F_NUMBER = 32, F_OPENER = 64 };

static const Forward FORWARD_START = { .mode = CODE, .line_start = true };

static inline unsigned forward_step(Forward *s, unsigned c) {
    bool newline = c == '\n';
    unsigned flags = 0;

    /* A newline ends line comments and literals, even an escaped one: there is no line
     * splicing here, so the newline itself belongs to neither. */
    if (newline && s->mode != BLOCK_COMMENT) {
        s->mode = CODE;
        s->escaped = false;
    }

    bool inside = s->mode != CODE;   /* inside something, as of the start of this byte */
    bool was_slash = s->last_slash;
    s->last_slash = false;

    if (s->mode == BLOCK_COMMENT) {
        flags |= F_COMMENT;
        if (s->last_star && c == '/') s->mode = CODE;
        s->last_star = c == '*';
    } else if (s->mode == LINE_COMMENT) {
        flags |= F_COMMENT;
    } else if (s->mode == STRING || s->mode == CHAR_LIT) {
        flags |= s->mode == STRING ? F_STRING : F_CHAR;
        if (s->escaped) s->escaped = false;
        else if (c == '\\') s->escaped = true;
        else if (c == (s->mode == STRING ? '"' : '\'')) s->mode = CODE;
    } else if (was_slash && c == '*') {       /* "/ *" opens; the star can't also close it */
        s->mode = BLOCK_COMMENT;
        flags |= F_COMMENT | F_OPENER;
    } else if (was_slash && c == '/') {
        s->mode = LINE_COMMENT;
        flags |= F_COMMENT | F_OPENER;
    } else if (c == '"') {
        s->mode = STRING;
        flags |= F_STRING;
    } else if (c == '\'') {
        s->mode = CHAR_LIT;
        flags |= F_CHAR;
    } else if (c == '/') {
        s->last_slash = true;
    }

    /* A directive runs to the end of the line. It continues if the newline follows a
     * backslash, whatever else that backslash was doing. */
    bool directive_goes_on = s->preproc && (!newline || s->last_bs);
    s->preproc = directive_goes_on || (c == '#' && s->line_start && !inside);
    s->last_bs = directive_goes_on && c == '\\';
    if (s->preproc) flags |= F_PREPROC;

    s->line_start = newline || (s->line_start && is_space(c));

    /* Words and numbers. Once a number starts, letters and dots stay part of it. */
    bool word = false, number = false;
    if (!inside) {
        if ((s->word && is_ident(c)) || (is_alpha(c) && !s->number)) word = true;
        else if ((is_ident(c) || c == '.') && (s->number || is_digit(c))) number = true;
    }
    s->word = word;
    s->number = number;
    if (word) flags |= F_WORD;
    if (number) flags |= F_NUMBER;
    return flags;
}

/* ---- right-to-left pass ---- */
typedef struct {
    bool paren_ahead;   /* this byte starts: spaces, then '(' */
    bool opener;        /* this byte is the second byte of a comment opener */
    bool word;
    int word_class;
} Backward;

/* kind: word_kind() of the word ending here; only read when this byte is the word's last. */
static inline int backward_step(Backward *r, unsigned c, unsigned flags, int kind) {
    int word_class = 0;
    if (flags & F_WORD) {
        if (r->word) word_class = r->word_class;
        else if (kind) word_class = kind;
        else if (r->paren_ahead) word_class = C_FUNCTION;
        else word_class = C_IDENT;
    }
    bool comment = (flags & F_COMMENT) || (c == '/' && r->opener);

    r->paren_ahead = c == '(' || (is_space(c) && r->paren_ahead);
    r->opener = flags & F_OPENER;
    r->word = flags & F_WORD;
    r->word_class = word_class;

    /* Priority order, highest first. */
    if (flags & F_STRING) return C_STRING;
    if (flags & F_CHAR) return C_CHAR;
    if (comment) return C_COMMENT;
    if (flags & F_PREPROC) return C_PREPROC;
    if (flags & F_WORD) return word_class;
    if (flags & F_NUMBER) return C_CONSTANT;
    if (is_space(c) || c == '\n') return C_SPACE;
    return C_PUNCT;
}

void lexnet_classify(const unsigned char *src, size_t n, unsigned char *cls) {
    unsigned char *flags = malloc(n + 1);
    Forward f = FORWARD_START;
    for (size_t i = 0; i < n; i++)
        flags[i] = (unsigned char)forward_step(&f, src[i]);

    Backward r = {0};
    for (size_t i = n; i-- > 0;) {
        int kind = 0;
        if ((flags[i] & F_WORD) && !r.word) {
            size_t start = i;
            while (start > 0 && (flags[start - 1] & F_WORD)) start--;
            kind = word_kind(src + start, i - start + 1);
        }
        cls[i] = (unsigned char)backward_step(&r, src[i], flags[i], kind);
    }
    free(flags);
}

/* ---- command line: same interface as the original ---- */
#ifndef LEXNET_NO_MAIN
static const char *ANSI[11] = {
    "", "\x1b[1;35m", "\x1b[36m", "\x1b[1;34m", "", "\x1b[33m",
    "\x1b[32m", "\x1b[32m", "\x1b[2;3;37m", "\x1b[38;5;208m", "\x1b[90m",
};

int main(int argc, char **argv) {
    int mode = 0, reps = 1;
    const char *path = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--classes")) mode = 1;
        else if (!strcmp(argv[i], "--bench") && i + 1 < argc) mode = 2, reps = atoi(argv[++i]);
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: lexnet_ifelse [--classes|--bench N] file.c\n"); return 2; }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *src = malloc(n + 1), *cls = malloc(n + 1);
    if (fread(src, 1, n, f) != n) return 1;
    fclose(f);

    if (mode == 2) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < reps; i++) lexnet_classify(src, n, cls);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double sec = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
        printf("%zu chars x %d: %.1f ms  (%.2f ns/char, %.1f M chars/s)\n",
               n, reps, sec * 1e3, sec / reps / (double)n * 1e9, (double)n * reps / sec / 1e6);
        return 0;
    }
    lexnet_classify(src, n, cls);
    if (mode == 1) {
        for (size_t t = 0; t < n; t++) putchar("0123456789A"[cls[t]]);
        return 0;
    }
    int cur = -1;
    for (size_t t = 0; t < n; t++) {
        if (cls[t] != cur) { fputs("\x1b[0m", stdout); fputs(ANSI[cls[t]], stdout); cur = cls[t]; }
        putchar(src[t]);
    }
    fputs("\x1b[0m", stdout);
    return 0;
}
#endif
