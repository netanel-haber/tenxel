// Exhaustive equivalence proof: real kernel + real weights vs lexnet_ifelse.c.
// 1. Forward: BFS over the product automaton (network mode state, position counter, if/else state)
//    on all 256 bytes; every output flag must agree on every reachable edge.
// 2. Backward: every reachable right-state x every reachable forward output, with word
//    kind flags toggled; class and next state must agree.
// 3. Numeric: every table word, plus near misses, through both whole pipelines; injectivity of the
//    base-64 packing, which turns "the table holds N words" into a statement about ALL strings.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#define main orig_main
#include "lexnet_hf/lexnet.c"
#undef main
#define LEXNET_NO_MAIN
#include "lexnet_ifelse.c"

#define NF 15
static uint32_t pack(const FState *s) {
    const double *d = (const double *)s; uint32_t b = 0;
    for (int i = 0; i < NF; i++) {
        if (d[i] != 0 && d[i] != 1) { puts("FAIL: non-boolean unit"); exit(1); }
        if (d[i] == 1) b |= 1u << i;
    }
    return b;
}
typedef struct { uint32_t net; int d; Forward me; } Pair;
static int key_eq(const Pair *a, const Pair *b) {
    return a->net == b->net && a->d == b->d && !memcmp(&a->me, &b->me, sizeof a->me);
}
static uint32_t mine_pack(const Forward *m) {
    return (uint32_t)m->mode | m->last_star << 3 | m->escaped << 4 | m->last_slash << 5 | m->line_start << 6 |
           m->preproc << 7 | m->last_bs << 8 | m->word << 9 | m->number << 10;
}
#define MAXP 20000
static Pair q[MAXP]; static int qt;
static uint16_t bitset[1 << 15]; // reachable net outputs

static void fail(const char *w, int c, const Pair *p) {
    printf("FAIL %s at byte %d net=%x d=%d mine=%x\n", w, c, p->net, p->d, mine_pack(&p->me)); exit(1);
}
int main(void) {
    Net net = load("lexnet_hf/lexnet_weights.bin");
    double kc[K_N_COEF], kb[K_N_BIAS]; gather(&net, kc, kb);

    // Character predicates against the embedding, all 256 bytes
    for (int b = 0; b < 256; b++) {
        const double *x = net.emb + (b < 128 ? b : 127) * 14;
        unsigned c = (unsigned)b;
        if ((x[0] != 0) != is_alpha(c) || (x[1] != 0) != is_digit(c) || (x[2] != 0) != is_ident(c) ||
            (x[3] != 0) != (is_ident(c) || c == '.') || (x[4] != 0) != is_space(c) || (x[5] != 0) != (c == '\n') ||
            (x[6] != 0) != (c == '"') || (x[7] != 0) != (c == '\'') || (x[8] != 0) != (c == '\\') ||
            (x[9] != 0) != (c == '/') || (x[10] != 0) != (c == '*') || (x[11] != 0) != (c == '#') || (x[12] != 0) != (c == '(')) {
            printf("FAIL: predicate differs on byte %d\n", b); return 1;
        }
    }
    puts("ok: character predicates match the embedding on all 256 bytes");

    // Base-64 injectivity
    { int seen[64] = {0};
      for (int b = 0; b < 128; b++) { const double *x = net.emb + b * 14; int v = (int)x[13];
        if (x[13] != v || (v != 0) != (x[2] != 0) || v < 0 || v > 63) { puts("FAIL: x13 range"); return 1; }
        if (v) { if (seen[v]) { puts("FAIL: x13 collision"); return 1; } seen[v] = 1; } }
      puts("ok: x[13] maps the 63 identifier chars one-to-one onto 1..63 (no zero digit, so packing is injective)"); }

    // Forward product BFS
    FState a, b; k_init_state(&a, net.h0);
    Pair s0 = { pack(&a), (int)((double *)&a)[15], FORWARD_START };
    q[qt++] = s0; unsigned long edges = 0; int maxd = 0;
    for (int h = 0; h < qt; h++) {
        Pair p = q[h];
        for (int c = 0; c < 256; c++) {
            double *d = (double *)&a; memcpy(&a, net.h0, sizeof a);
            for (int i = 0; i < NF; i++) d[i] = (p.net >> i) & 1;
            d[15] = p.d; d[16] = 0; d[17] = 0;
            uint16_t bits = k_fwd_step(kc, kb, net.emb + (c < 128 ? c : 127) * 14, &a, &b);
            Pair n; memset(&n, 0, sizeof n); n.net = pack(&b); n.d = (int)((double *)&b)[15]; n.me = p.me;
            if (((double *)&b)[15] != n.d) { puts("FAIL: d non-integer"); return 1; }
            unsigned f = forward_step(&n.me, (unsigned)c);
            int fC = bits & 1, fL = bits >> 1 & 1, fS = bits >> 2 & 1, fQ = bits >> 3 & 1, fP = bits >> 4 & 1, fW = bits >> 5 & 1, fN = bits >> 6 & 1, fop = bits >> 10 & 1;
            if (fS != !!(f & F_STRING)) fail("string", c, &p);
            if (fQ != !!(f & F_CHAR)) fail("char", c, &p);
            if ((fC | fL) != !!(f & F_COMMENT)) fail("comment", c, &p);
            if (fP != !!(f & F_PREPROC)) fail("preproc", c, &p);
            if (fW != !!(f & F_WORD)) fail("word", c, &p);
            if (fN != !!(f & F_NUMBER)) fail("number", c, &p);
            if (fop != !!(f & F_OPENER)) fail("opener", c, &p);
            bitset[bits & 0x7fff] = 1; edges++;
            int found = 0; for (int i = 0; i < qt; i++) if (key_eq(&q[i], &n)) { found = 1; break; }
            if (!found) { if (qt == MAXP) { puts("overflow"); return 1; } q[qt++] = n; if (n.d > maxd) maxd = n.d; }
        }
    }
    printf("ok: forward product automaton: %d reachable state pairs, %lu edges checked, zero disagreements\n", qt, edges);

    // Word position invariant: with W set, d = max(15-pos,0). Follows word run from each start.
    { const char *w = "abcdefghijklmnopqrstuvwxyz"; FState x; k_init_state(&x, net.h0); FState y; int ok = 1;
      for (int i = 0; i < 26; i++) { k_fwd_step(kc, kb, net.emb + w[i] * 14, &x, &y); x = y;
        int pos = i + 1, want = pos == 1 ? 14 : (15 - pos < 0 ? 0 : 15 - pos); if ((int)x.d != want) ok = 0; }
      printf("%s: position counter tracks word offset, chars 0-7 go to h1, 8-13 to h2\n", ok ? "ok" : "FAIL"); if (!ok) return 1; }

    // Backward: all reachable forward outputs, KW/TY/CONST each free (they can fire off-word too, e.g. on the
    // byte after "continue", where h1 is held).
    static uint16_t bl[1 << 15]; int nb = 0;
    for (int b0 = 0; b0 < 1 << 15; b0++) if (bitset[b0]) {
        int base = b0 & ~(7 << 7);
        for (int k = 0; k < 4; k++) {
            int bb = base | (k ? 1 << (6 + k) : 0); int dup = 0;
            for (int i = 0; i < nb; i++) if (bl[i] == bb) dup = 1;
            if (!dup) bl[nb++] = (uint16_t)bb;
        }
    }
    typedef struct { BState n; Backward m; } RP;
    static RP rq[100000]; int rt = 0; memset(&rq[0], 0, sizeof rq[0]); rt = 1; unsigned long bedges = 0;
    for (int h = 0; h < rt; h++) {
        for (int i = 0; i < nb; i++) {
            uint16_t bits = bl[i];
            RP p = rq[h], n; memset(&n, 0, sizeof n); n.m = p.m;
            int cn = k_bwd_step(kc, kb, bits, &p.n, &n.n);
            unsigned f = 0;
            if (bits >> 2 & 1) f |= F_STRING;
            if (bits >> 3 & 1) f |= F_CHAR;
            if (bits & 3) f |= F_COMMENT;
            if (bits >> 4 & 1) f |= F_PREPROC;
            if (bits >> 5 & 1) f |= F_WORD;
            if (bits >> 6 & 1) f |= F_NUMBER;
            if (bits >> 10 & 1) f |= F_OPENER;
            // pick a byte consistent with the raw flags
            unsigned c = 0x7f;
            if (bits >> 11 & 1) c = '/'; else if (bits >> 12 & 1) c = ' '; else if (bits >> 13 & 1) c = '\n'; else if (bits >> 14 & 1) c = '(';
            int kind = (bits >> 7 & 1) ? C_KEYWORD : (bits >> 8 & 1) ? C_TYPE : (bits >> 9 & 1) ? C_CONSTANT : 0;
            int cm = backward_step(&n.m, c, f, kind);
            if (cn != cm) { printf("FAIL backward class net=%d mine=%d bits=%x\n", cn, cm, bits); return 1; }
            if ((n.n.W != 0) != n.m.word || (n.n.lps != 0) != n.m.paren_ahead || (n.n.copen2 != 0) != n.m.opener) { printf("FAIL backward state bits=%x\n", bits); return 1; }
            bedges++;
            int found = 0; for (int j = 0; j < rt; j++) if (!memcmp(&rq[j].n, &n.n, sizeof n.n) && !memcmp(&rq[j].m, &n.m, sizeof n.m)) { found = 1; break; }
            if (!found) rq[rt++] = n;
        }
    }
    printf("ok: backward: %d forward outputs x %d right-states, %lu edges checked, zero disagreements\n", nb, rt, bedges);

    // Word table: every word and near misses, full pipeline on both sides
    static const char *words[] = { "do","if","for","auto","case","else","enum","goto","break","const","union","while","extern","inline","return","sizeof","static","struct","switch","typeof","alignas","alignof","default","typedef","_Atomic","continue","register","restrict","volatile","_Alignas","_Alignof","_Generic","_Noreturn","constexpr","_Thread_local","thread_local","static_assert","_Static_assert",
      "int","bool","char","long","void","FILE","float","off_t","pid_t","short","_Bool","double","int8_t","signed","size_t","time_t","int16_t","int32_t","int64_t","ssize_t","uint8_t","va_list","wchar_t","intptr_t","uint16_t","uint32_t","uint64_t","unsigned","_Complex","ptrdiff_t","uintptr_t",
      "EOF","true","NULL","false","nullptr" };
    const char *alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_";
    unsigned char *buf = malloc(1 << 28), *c1 = malloc(1 << 28), *c2 = malloc(1 << 28); uint16_t *bits = malloc(2u << 28);
    size_t n = 0; unsigned long cases = 0;
#define ADD(s, l) do { memcpy(buf + n, s, l); n += (size_t)(l); buf[n++] = ';'; buf[n++] = ' '; cases++; } while (0)
    for (size_t i = 0; i < sizeof words / sizeof *words; i++) {
        char t[64]; size_t L = strlen(words[i]);
        ADD(words[i], L);
        for (size_t k = 1; k < L; k++) ADD(words[i], k);                  // prefixes
        for (int a1 = 0; a1 < 63; a1++) {
            memcpy(t, words[i], L); t[L] = alpha[a1]; ADD(t, L + 1);      // one char longer
            for (int a2 = 0; a2 < 63; a2 += 7) { t[L + 1] = alpha[a2]; memset(t + L + 2, 'q', 8); ADD(t, L + 2 + (size_t)(a2 % 9)); } // longer
            t[0] = alpha[a1]; memcpy(t + 1, words[i], L); ADD(t, L + 1);  // prepended
            for (size_t k = 0; k < L; k++) { memcpy(t, words[i], L); t[k] = alpha[a1]; ADD(t, L); } // every single substitution
            for (size_t k = 0; k + 1 < L; k++) { memcpy(t, words[i], L); memmove(t + k, t + k + 1, L - k - 1); ADD(t, L - 1); } // deletions
        }
    }
    classify_compiled(&net, kc, kb, buf, n, c1, bits);
    lexnet_classify(buf, n, c2);
    if (memcmp(c1, c2, n)) { for (size_t i = 0; i < n; i++) if (c1[i] != c2[i]) { printf("FAIL word at %zu: %.40s\n", i, buf + (i > 20 ? i - 20 : 0)); return 1; } }
    printf("ok: %lu word variants (%zu bytes) classify identically\n", cases, n);
    puts("PROVED");
    return 0;
}
