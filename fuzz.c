// Differential fuzz: real network (compiled kernel + real weights) vs lexnet_ifelse vs lexnet_fast.
// Inputs sit flush against unmapped guard pages, so any out-of-bounds read or write faults.
// usage: fuzz <seed> <net_budget_bytes> <fast_budget_bytes> <corpus.c>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#define main orig_main
#include "lexnet_hf/lexnet.c"
#undef main

typedef unsigned char u8;
void ifelse_classify(const u8 *, size_t, u8 *);
void fast_classify(const u8 *, size_t, u8 *);

static uint64_t st[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rnd(void) {
    uint64_t r = rotl(st[1] * 5, 7) * 9, t = st[1] << 17;
    st[2] ^= st[0]; st[3] ^= st[1]; st[1] ^= st[2]; st[0] ^= st[3]; st[2] ^= t; st[3] = rotl(st[3], 45);
    return r;
}
static inline size_t rn(size_t n) { return n ? rnd() % n : 0; }

#define MAXN (1u << 20)
static long PG;
static u8 *guarded(size_t cap) {          // [guard][cap][guard]
    size_t sz = (cap + PG - 1) / PG * PG;
    u8 *m = mmap(0, sz + 2 * PG, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { perror("mmap"); exit(2); }
    mprotect(m, PG, PROT_NONE); mprotect(m + PG + sz, PG, PROT_NONE);
    return m + PG;
}
static u8 *inbuf, *outa, *outb, *outc; static size_t CAP;
static u8 *corp; static size_t corpn;

static const char *TOK[] = { "do","if","for","auto","case","else","enum","goto","break","const","union","while","extern",
 "inline","return","sizeof","static","struct","switch","typeof","alignas","alignof","default","typedef","_Atomic","continue",
 "register","restrict","volatile","_Alignas","_Alignof","_Generic","_Noreturn","constexpr","_Thread_local","thread_local",
 "static_assert","_Static_assert","int","bool","char","long","void","FILE","float","off_t","pid_t","short","_Bool","double",
 "int8_t","signed","size_t","time_t","int16_t","int32_t","int64_t","ssize_t","uint8_t","va_list","wchar_t","intptr_t",
 "uint16_t","uint32_t","uint64_t","unsigned","_Complex","ptrdiff_t","uintptr_t","EOF","true","NULL","false","nullptr",
 "_Static_assertXYZ","_Static_asser","_Static_assert_","thread_localx","uintptr_t_","continueX","continu","a","x1","_","__",
 "abcdefghijklmn","abcdefghijklmno","abcdefghijklmnopqrstuvwxyz0123456789","0","1","0x1F","1.5e3","1e","3.","..","...",".5","1a2b","0b1",
 "/*","*/","//","/","*","/**/","/*/","\"","'","\\","\\\n","\\\\","\\\"","\\'","\n","\r\n","\n\n"," ","  ","\t","\v","\f","\r",
 "(",")","{","}","[","]",";",",","#","#define X","# include <a.h>","#if 0","  #","\n#","\n  #","/**/#","(\n","( "," (","  (","\t(",
 "\x80","\xff","\x7f","\0x","$","@","`","\xc3\xa9","-","+","=","<",">","->","~","!" };
#define NTOK (sizeof TOK / sizeof *TOK)
static const u8 HOT[] = { '/','*','"','\'','\\','\n','#','(',' ','.','a','0','_','\t','\r','\v','\f',0,0x80,0xff,0x7f,'z','9' };

static size_t emit_tok(u8 *p, size_t room) {
    const char *t = TOK[rn(NTOK)]; size_t l = t[0] ? strlen(t) : 1;   // "\0x" has leading NUL
    if (t[0] == 0) l = 2;
    if (l > room) l = room;
    memcpy(p, t, l); return l;
}
static size_t run_of(u8 *p, size_t room) {        // long homogeneous runs crossing many blocks
    static const char *U[] = { "a","0"," ","\n","\t","/","*","\"","'","\\","#","(",".","ab","a ","a(","//","/*","*/","\\\n","1.","\xff","\0" };
    const char *u = U[rn(sizeof U / sizeof *U)]; size_t l = strlen(u); if (!l) l = 1;
    size_t n = 1 + rn(rn(4) ? 80 : 400); n *= l; if (n > room) n = room;
    for (size_t i = 0; i < n; i++) p[i] = (u8)u[i % l];
    return n;
}
static size_t gen(u8 *b, size_t cap, int kind) {
    size_t n = 0;
    size_t want = rn(4) == 0 ? rn(70) : rn(8) == 0 ? rn(cap) : rn(rn(2) ? 300 : 3000);
    if (want > cap) want = cap;
    switch (kind) {
    case 0: for (; n < want; n++) b[n] = (u8)rnd(); break;
    case 1: for (; n < want; n++) b[n] = HOT[rn(sizeof HOT)]; break;
    case 2: while (n < want) n += emit_tok(b + n, want - n); break;
    case 3: while (n < want) { n += rn(3) ? emit_tok(b + n, want - n) : run_of(b + n, want - n); } break;
    case 4: {                                     // tokens forced to straddle 64-byte blocks
        while (n < want) {
            size_t to = 64 - n % 64, pad = to > 20 ? to - 1 - rn(19) : 0;
            static const char FP[] = " a0.;\n(";
            char f = FP[rn(sizeof FP - 1)]; if (pad > want - n) pad = want - n;
            memset(b + n, f, pad); n += pad;
            if (n < want) n += emit_tok(b + n, want - n);
            if (n < want) n += emit_tok(b + n, want - n);
        } } break;
    case 5: case 6: {                             // mutated real code
        if (want > corpn) want = corpn;
        size_t o = rn(corpn - want + 1); memcpy(b, corp + o, want); n = want;
        size_t muts = kind == 5 ? rn(4) : rn(1 + want / 8);
        for (size_t m = 0; m < muts && n; m++) {
            size_t at = rn(n);
            switch (rn(5)) {
            case 0: b[at] = (u8)rnd(); break;
            case 1: b[at] = HOT[rn(sizeof HOT)]; break;
            case 2: b[at] ^= (u8)(1 << rn(8)); break;
            case 3: { size_t l = 1 + rn(40); if (at + l < n) { memmove(b + at, b + at + l, n - at - l); n -= l; } } break;
            case 4: { size_t l = 1 + rn(12); if (n + l < cap) { memmove(b + at + l, b + at, n - at); for (size_t i = 0; i < l; i++) b[at + i] = HOT[rn(sizeof HOT)]; n += l; } } break;
            }
        } } break;
    }
    return n;
}

static Net net; static double kc[K_N_COEF], kb[K_N_BIAS]; static uint16_t *bits; static double *feats;
static void report(const char *who, const u8 *s, size_t n, const u8 *want, const u8 *got) {
    size_t i = 0; while (i < n && want[i] == got[i]) i++;
    size_t a = i > 48 ? i - 48 : 0, e = i + 16 < n ? i + 16 : n;
    printf("MISMATCH %s n=%zu at=%zu want=%d got=%d\nbytes:", who, n, i, want[i], got[i]);
    for (size_t k = a; k < e; k++) printf(" %02x", s[k]);
    printf("\nwant : "); for (size_t k = a; k < e; k++) putchar("0123456789A"[want[k]]);
    printf("\ngot  : "); for (size_t k = a; k < e; k++) putchar("0123456789A"[got[k]]);
    puts(""); fflush(stdout);
    FILE *f = fopen("fuzz_fail.bin", "wb"); fwrite(s, 1, n, f); fclose(f);
    exit(1);
}
int main(int argc, char **argv) {
    uint64_t seed = strtoull(argv[1], 0, 0); size_t net_budget = strtoull(argv[2], 0, 0), fast_budget = strtoull(argv[3], 0, 0);
    PG = sysconf(_SC_PAGESIZE); CAP = MAXN;
    for (int i = 0; i < 4; i++) { seed += 0x9E3779B97F4A7C15ULL; uint64_t z = seed; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; st[i] = z ^ (z >> 31); }
    FILE *f = fopen(argv[4], "rb"); fseek(f, 0, SEEK_END); corpn = (size_t)ftell(f); fseek(f, 0, SEEK_SET); corp = malloc(corpn); if (fread(corp, 1, corpn, f) != corpn) return 2; fclose(f);
    net = load("lexnet_hf/lexnet_weights.bin"); gather(&net, kc, kb);
    bits = malloc(2 * (size_t)MAXN + 64); feats = malloc((size_t)MAXN * (size_t)net.n_fout * 8);
    size_t sz = (CAP + PG - 1) / PG * PG;
    inbuf = guarded(CAP); outa = guarded(CAP); outb = guarded(CAP); outc = guarded(CAP);
    static u8 tmp[MAXN]; unsigned long cases[2] = {0}, kindc[7] = {0}, interp_cases = 0; size_t netb = 0, fastb = 0;
    while (netb < net_budget || fastb < fast_budget) {
        int kind = (int)rn(7); size_t n = gen(tmp, MAXN / 8, kind); kindc[kind]++;
        // flush right against the trailing guard, or left against the leading guard
        int flush_start = (int)(rnd() & 1);
        u8 *s = flush_start ? inbuf : inbuf + sz - n; memcpy(s, tmp, n);
        u8 *oa = outa + sz - n, *ob = outb + sz - n, *oc = outc + sz - n;
        ifelse_classify(s, n, ob);
        fast_classify(s, n, oc);
        if (memcmp(ob, oc, n)) report("fast-vs-ifelse", s, n, ob, oc);
        fastb += n; cases[1]++;
        if (netb < net_budget) {
            classify_compiled(&net, kc, kb, s, n, oa, bits);
            if (memcmp(oa, ob, n)) report("ifelse-vs-net", s, n, oa, ob);
            if (n && n < 4096 && rn(4) == 0) {        // the true definition: the interpreter over every matrix
                use_interp = 1; classify(&net, s, n, oa, feats); use_interp = 0; interp_cases++;
                if (memcmp(oa, ob, n)) report("ifelse-vs-interp", s, n, oa, ob);
            }
            netb += n; cases[0]++;
        }
    }
    printf("seed %s: net-checked %lu cases %zu bytes (%lu via interpreter); fast-vs-ifelse %lu cases %zu bytes; kinds", argv[1], cases[0], netb, interp_cases, cases[1], fastb);
    for (int i = 0; i < 7; i++) printf(" %lu", kindc[i]);
    puts(""); return 0;
}
