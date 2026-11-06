/* exhaust.c - every string up to a given length over the bytes that matter, inside padding that places it across a
 * 64-byte block boundary: tenxel_classify against lex_token alone, with the input flush against an unmapped page.
 * Progress goes to stderr.
 *
 * Usage: exhaust MAX_LENGTH
 */
#define main tenxel_main /* this file has its own */
#include "tenxel.c"
#undef main
#include <sys/mman.h>
#include <unistd.h>

static const u8 ALPHABET[] = {'/', '*', '"', '\'', '\\', '\n', '#', '(', ' ', '.', 'u', '0', 'e', '+', '%', ':'};
#define N_ALPHABET sizeof ALPHABET

/* The blocks' keyword table (WORDS, made by words.py) against keyword_class, which lex_token uses:
 * every keyword, each of its prefixes, and it with a letter more */
static int check_words(void) {
    for (size_t i = 0; i < KEYWORD_SLOTS; i++) {
        size_t len = strlen(KEYWORDS[i].word);
        u8 w[80] = {0};
        memcpy(w, KEYWORDS[i].word, len);
        w[len] = 'x';
        for (size_t l = 1; l <= len + 1; l++) {
            int want = keyword_class(w, l);
            int got = (int)word_class(w, (unsigned)l) ^ C_IDENT;
            if (len && want != got) {
                printf("MISMATCH: the word table says %d for %.*s, keyword_class %d: run words.py\n", got, (int)l, w,
                       want);
                return 1;
            }
        }
    }
    return 0;
}

static void classify_plain(const u8 *src, size_t n, u8 *cls) {
    Lexer L = {src, src + n, cls, true, false};
    const u8 *p = src;
    while (p < L.end)
        p = lex_token(&L, p);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fputs("usage: exhaust MAX_LENGTH\n", stderr);
        return 2;
    }
    int max = atoi(argv[1]);
    if (check_words())
        return 1;
    long page = sysconf(_SC_PAGESIZE);
    u8 *m = mmap(0, 3 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(m + 2 * page, page, PROT_NONE);
    static const int PADDING[] = {0, 1, 60, 61, 62, 63, 64}; /* bytes before the string */
    static u8 x[256];
    static u8 want[256];
    static u8 got[256];
    unsigned long tested = 0;

    for (int len = 0; len <= max; len++) {
        unsigned long total = 1;
        for (int i = 0; i < len; i++)
            total *= N_ALPHABET;
        for (int pi = 0; pi < 7; pi++) {
            for (int fill = 0; fill < 3; fill++) {
                int pad = PADDING[pi];
                if (len >= 5 && (fill || (pad != 62 && pad != 0)))
                    continue;
                for (unsigned long w = 0; w < total; w++) {
                    memset(x, " a;"[fill], (size_t)pad);
                    unsigned long t = w;
                    for (int i = 0; i < len; i++) {
                        x[pad + i] = ALPHABET[t % N_ALPHABET];
                        t /= N_ALPHABET;
                    }
                    size_t n = (size_t)(pad + len);
                    u8 *s = m + 2 * page - n;
                    memcpy(s, x, n);
                    classify_plain(s, n, want);
                    tenxel_classify(s, n, got);
                    tested++;
                    if (memcmp(want, got, n)) {
                        printf("MISMATCH n=%zu:", n);
                        for (size_t i = 0; i < n; i++)
                            printf(" %02x", s[i]);
                        printf("\nlex_token       ");
                        for (size_t i = 0; i < n; i++)
                            putchar("0123456789A"[want[i]]);
                        printf("\ntenxel_classify ");
                        for (size_t i = 0; i < n; i++)
                            putchar("0123456789A"[got[i]]);
                        puts("");
                        return 1;
                    }
                }
            }
        }
        fprintf(stderr, "length %d done, %lu tested\n", len, tested);
    }
    printf("ok: exhaust: %lu strings up to length %d, block paths match lex_token\n", tested, max);
    return 0;
}
