// Exhaustive: every string up to length L over a small hot alphabet, inside padded contexts that
// place it across a 64-byte block boundary. fast vs ifelse, flush against a trailing guard page.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
typedef unsigned char u8;
void ifelse_classify(const u8 *, size_t, u8 *);
void fast_classify(const u8 *, size_t, u8 *);
static const u8 A[] = { '/','*','"','\'','\\','\n','#','(',' ','.','a','0',0x80 };
#define NA 13
int main(int c, char **v) {
    int L = atoi(v[1]); long pg = sysconf(_SC_PAGESIZE);
    u8 *m = mmap(0, 3 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); mprotect(m + 2 * pg, pg, PROT_NONE);
    static u8 x[256], a[256], b[256]; unsigned long tested = 0;
    static const int pre[] = { 0, 1, 60, 61, 62, 63, 64 };     // padding before the string
    for (int len = 0; len <= L; len++) {
        unsigned long total = 1; for (int i = 0; i < len; i++) total *= NA;
        for (int pi = 0; pi < 7; pi++) for (int fill = 0; fill < 3; fill++) {
            int p = pre[pi]; if (len >= 7 && fill) continue;  if (len >= 7 && p != 62 && p != 0) continue;
            for (unsigned long w = 0; w < total; w++) {
                memset(x, " a;"[fill], (size_t)p); unsigned long t = w;
                for (int i = 0; i < len; i++) { x[p + i] = A[t % NA]; t /= NA; }
                size_t n = (size_t)(p + len); u8 *s = m + 2 * pg - n; memcpy(s, x, n);
                ifelse_classify(s, n, a); fast_classify(s, n, b); tested++;
                if (memcmp(a, b, n)) { printf("MISMATCH n=%zu:", n); for (size_t i = 0; i < n; i++) printf(" %02x", s[i]); printf("\nwant "); for (size_t i = 0; i < n; i++) putchar("0123456789A"[a[i]]); printf("\ngot  "); for (size_t i = 0; i < n; i++) putchar("0123456789A"[b[i]]); puts(""); return 1; }
            }
        }
        printf("len %d done, %lu tested\n", len, tested); fflush(stdout);
    }
    puts("exhaustive ok"); return 0;
}
