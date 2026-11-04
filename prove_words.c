// Closes the gap in prove.c, whose automaton search pins h1 = h2 = 0.
// A. Accumulator step: on any identifier byte, for every (W, d) and a grid of in-range h values,
//    the REAL k_fwd_step updates h1, h2, d exactly as base-64 packing of bytes 0-7 and 8-13.
//    A word start resets them no matter what garbage they hold.
// B. Keyword units: with d = 0 the accumulators hold, so any (h1, h2) can be injected. ge/le are
//    step functions of h, so testing every threshold, its neighbours and the extremes covers all h.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define main orig_main
#include "lexnet_hf/lexnet.c"
#undef main

static const char *KW[] = {"do","if","for","auto","case","else","enum","goto","break","const","union","while","extern","inline","return","sizeof","static","struct","switch","typeof","alignas","alignof","default","typedef","_Atomic","continue","register","restrict","volatile","_Alignas","_Alignof","_Generic","_Noreturn","constexpr","_Thread_local","thread_local","static_assert","_Static_assert",0};
static const char *TY[] = {"int","bool","char","long","void","FILE","float","off_t","pid_t","short","_Bool","double","int8_t","signed","size_t","time_t","int16_t","int32_t","int64_t","ssize_t","uint8_t","va_list","wchar_t","intptr_t","uint16_t","uint32_t","uint64_t","unsigned","_Complex","ptrdiff_t","uintptr_t",0};
static const char *CO[] = {"EOF","true","NULL","false","nullptr",0};
typedef struct { uint64_t h1, h2; int cls; } E;
static E tab[128]; static int nt;
static Net net; static double kc[K_N_COEF], kb[K_N_BIAS];
static int dig(unsigned c) { return (int)net.emb[c * 14 + 13]; }
static void add(const char **l, int cls) {
    for (; *l; l++) { uint64_t a = 0, b = 0; size_t n = strlen(*l);
        for (size_t i = 0; i < n; i++) { if (i < 8) a = a * 64 + (uint64_t)dig((u_int8_t)(*l)[i]); else b = b * 64 + (uint64_t)dig((u_int8_t)(*l)[i]); }
        tab[nt++] = (E){a, b, cls}; }
}
static int want_cls(uint64_t a, uint64_t b) { for (int i = 0; i < nt; i++) if (tab[i].h1 == a && tab[i].h2 == b) return tab[i].cls; return 0; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static int cmpu(const void *x, const void *y) { uint64_t a = *(const uint64_t *)x, b = *(const uint64_t *)y; return a < b ? -1 : a > b; }

int main(void) {
    net = load("lexnet_hf/lexnet_weights.bin"); gather(&net, kc, kb);
    add(KW, 1); add(TY, 2); add(CO, 3);
    printf("table: %d words\n", nt);
    for (int i = 0; i < nt; i++) for (int j = i + 1; j < nt; j++) if (tab[i].h1 == tab[j].h1 && tab[i].h2 == tab[j].h2) { puts("FAIL: duplicate"); return 1; }

    // A. accumulators
    static const char *ID = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    unsigned long checks = 0; double P64[9]; P64[0] = 1; for (int i = 1; i < 9; i++) P64[i] = P64[i - 1] * 64;
    for (int W = 0; W < 2; W++) for (int d = 0; d <= 14; d++) for (const char *c = ID; *c; c++) for (int it = 0; it < 4000; it++) {
        int pos = d >= 1 ? 15 - d : 14;                // bytes consumed so far
        int p1 = pos > 7 ? 8 : pos, p2 = pos > 8 ? pos - 8 : 0; if (p2 > 6) p2 = 6;
        uint64_t m1 = (uint64_t)P64[p1], m2 = (uint64_t)P64[p2];
        uint64_t a, b;
        switch (it % 6) { case 0: a = 0; b = 0; break; case 1: a = m1 - 1; b = m2 - 1; break; case 2: a = m1 / 2; b = m2 / 2; break;
            case 3: a = rnd() % m1; b = m2 - 1; break; case 4: a = m1 - 1; b = rnd() % m2; break; default: a = rnd() % m1; b = rnd() % m2; }
        if (!W) { a = rnd() >> 16; b = rnd() >> 28; if (it % 2) { a = it & 1 ? (1ULL << 48) - 1 : a; b = (1ULL << 36) - 1; } }  // garbage must not survive a word start
        FState p, s; memset(&p, 0, sizeof p); p.W = W; p.d = d; p.h1 = (double)a; p.h2 = (double)b;
        k_fwd_step(kc, kb, net.emb + (unsigned)*c * 14, &p, &s);
        int dg = dig((unsigned)*c); uint64_t ea, eb; int ed;
        if (!W) { ea = (uint64_t)dg; eb = 0; ed = 14; }
        else { ed = d > 0 ? d - 1 : 0; if (d >= 8) { ea = a * 64 + (uint64_t)dg; eb = b; } else if (d >= 2) { ea = a; eb = b * 64 + (uint64_t)dg; } else { ea = a; eb = b; } }
        if (s.h1 != (double)ea || s.h2 != (double)eb || (int)s.d != ed || s.W != ((W || !(*c >= '0' && *c <= '9')) ? 1.0 : 0.0)) {
            printf("FAIL A: W=%d d=%d c=%c h1=%llu h2=%llu -> h1=%.0f h2=%.0f d=%.0f W=%.0f want %llu %llu %d\n", W, d, *c, (unsigned long long)a, (unsigned long long)b, s.h1, s.h2, s.d, s.W, (unsigned long long)ea, (unsigned long long)eb, ed);
            return 1;
        }
        checks++;
    }
    printf("ok: accumulator step: %lu checks over all (W, d, identifier byte), zero disagreements\n", checks);

    // B. keyword units over every interval of (h1, h2)
    static uint64_t A[4096], B[4096]; int na = 0, nb = 0;
#define PUSH(arr, n, v) do { if ((int64_t)(v) >= 0) arr[n++] = (v); } while (0)
    for (int i = 0; i < nt; i++) { for (int k = -2; k <= 2; k++) { PUSH(A, na, tab[i].h1 + (uint64_t)(int64_t)k); PUSH(B, nb, tab[i].h2 + (uint64_t)(int64_t)k); } }
    for (int k = 0; k < 8; k++) { PUSH(A, na, (uint64_t)k); PUSH(B, nb, (uint64_t)k); }
    PUSH(A, na, (1ULL << 48) - 1); PUSH(A, na, 1ULL << 48); PUSH(B, nb, (1ULL << 36) - 1); PUSH(B, nb, 1ULL << 36);
    qsort(A, (size_t)na, 8, cmpu); qsort(B, (size_t)nb, 8, cmpu);
    int ua = 0, ub = 0; for (int i = 0; i < na; i++) if (!i || A[i] != A[i - 1]) A[ua++] = A[i]; for (int i = 0; i < nb; i++) if (!i || B[i] != B[i - 1]) B[ub++] = B[i];
    unsigned long kchecks = 0, hits = 0;
    for (int i = 0; i < ua; i++) for (int j = 0; j < ub; j++) {
        FState p, s; memset(&p, 0, sizeof p); p.W = 1; p.d = 0; p.h1 = (double)A[i]; p.h2 = (double)B[j];
        uint16_t bits = k_fwd_step(kc, kb, net.emb + 'q' * 14, &p, &s);
        if (s.h1 != p.h1 || s.h2 != p.h2) { puts("FAIL B: hold did not hold"); return 1; }
        int kw = bits >> 7 & 1, ty = bits >> 8 & 1, co = bits >> 9 & 1, e = want_cls(A[i], B[j]);
        if (kw != (e == 1) || ty != (e == 2) || co != (e == 3)) { printf("FAIL B: h1=%llu h2=%llu bits %d%d%d want class %d\n", (unsigned long long)A[i], (unsigned long long)B[j], kw, ty, co, e); return 1; }
        kchecks++; hits += e != 0;
    }
    printf("ok: keyword units: %d x %d = %lu (h1, h2) points covering every threshold interval, %lu true matches, zero disagreements\n", ua, ub, kchecks, hits);
    puts("PROVED: word bit at a word's last byte <=> its first 14 bytes are one of the 74 table words");
    return 0;
}
