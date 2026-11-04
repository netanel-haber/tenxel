/* lexnet_fast.c - the lexnet network, compiled all the way down.
 *
 * Output is byte-identical to the network. Instead of one forward and one backward pass
 * per byte, it makes a single forward pass over 64-byte blocks:
 *
 *   1. SIMD turns a block into bitmasks (identifier chars, digits, spaces, ...).
 *   2. Numbers are found by ripple carry: adding a number's start bit to the "number
 *      char" mask flips exactly that token, with no per-byte work.
 *   3. Each word costs one multiply into a perfect hash, with no data-dependent branches.
 *      The "(" lookahead the network's backward pass did is a bit shift.
 *   4. Default classes come from a table; word classes are XORed in as bit planes.
 *   5. Comments, strings and directives leave the block loop and jump with SIMD search.
 *
 * Blocks never carry state between them: a block stops before any token that might run
 * past its end, and the next block is loaded unaligned from exactly there.
 *
 * Build: cc -O3 -march=native -o lexnet_fast lexnet_fast.c
 * Paths: AVX-512 VBMI, AVX2, NEON, and a portable fallback, chosen at compile time.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint8_t u8;
typedef uint64_t u64;

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

enum { C_SPACE = 0, C_KEYWORD = 1, C_TYPE = 2, C_FUNCTION = 3, C_IDENT = 4, C_CONSTANT = 5,
       C_STRING = 6, C_CHAR = 7, C_COMMENT = 8, C_PREPROC = 9, C_PUNCT = 10 };

/* Byte kinds, for the mask step and the slow path. */
enum { K_IDENT = 1, K_DIGIT = 2, K_DOT = 4, K_SPACE = 8, K_LPAREN = 16, K_SLASH = 32, K_STAR = 64,
       K_QH = 128 /* " ' # */ };

#define R8(x) x,x,x,x,x,x,x,x
#define KD K_IDENT | K_DIGIT
#define KI K_IDENT
static const u8 KIND[256] __attribute__((aligned(64))) = {
    /* 00 */ 0,0,0,0,0,0,0,0, 0,K_SPACE,0,K_SPACE,K_SPACE,K_SPACE,0,0,
    /* 10 */ R8(0), R8(0),
    /* 20 */ K_SPACE,0,K_QH,K_QH,0,0,0,K_QH, K_LPAREN,0,K_STAR,0,0,0,K_DOT,K_SLASH,
    /* 30 */ KD,KD,KD,KD,KD,KD,KD,KD, KD,KD,0,0,0,0,0,0,
    /* 40 */ 0,KI,KI,KI,KI,KI,KI,KI, KI,KI,KI,KI,KI,KI,KI,KI,
    /* 50 */ KI,KI,KI,KI,KI,KI,KI,KI, KI,KI,KI,0,0,0,0,KI,
    /* 60 */ 0,KI,KI,KI,KI,KI,KI,KI, KI,KI,KI,KI,KI,KI,KI,KI,
    /* 70 */ KI,KI,KI,KI,KI,KI,KI,KI, KI,KI,KI,0,0,0,0,0,
    /* 80+ all zero */
};

/* Default class: identifier chars 4, whitespace and newline 0, everything else 10. */
#define P 10
#define S 0
#define I 4
static const u8 DEFCLASS[256] __attribute__((aligned(64))) = {
    P,P,P,P,P,P,P,P, P,S,S,S,S,S,P,P, R8(P),R8(P),
    S,P,P,P,P,P,P,P, P,P,P,P,P,P,P,P,
    I,I,I,I,I,I,I,I, I,I,P,P,P,P,P,P,
    P,I,I,I,I,I,I,I, I,I,I,I,I,I,I,I,
    I,I,I,I,I,I,I,I, I,I,I,P,P,P,P,I,
    P,I,I,I,I,I,I,I, I,I,I,I,I,I,I,I,
    I,I,I,I,I,I,I,I, I,I,I,P,P,P,P,P,
    R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),
    R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),R8(P),
};
#undef P
#undef S
#undef I

/* ---- perfect hash over the 74 words ----
 * A word is at most 14 significant bytes. .a holds bytes 0-7, .b holds bytes 8-13 with the
 * class in its top byte. Anything past 14 is ignored, which keeps the network's quirk:
 * "_Static_assertXYZ" counts as a keyword. */
typedef struct { u64 a, b; } WordEntry;
/* perfect hash: 69803 tries */
#define HASH_MUL 0xBE32047A1F762F35ULL
#define HASH_BITS 8
static const WordEntry WORDS[1 << HASH_BITS] = {
    [3] = {0x0000745F656D6974ULL, 0x0200000000000000ULL}, /* time_t */
    [10] = {0x0000686374697773ULL, 0x0100000000000000ULL}, /* switch */
    [12] = {0x000000745F646970ULL, 0x0200000000000000ULL}, /* pid_t */
    [16] = {0x00006E7275746572ULL, 0x0100000000000000ULL}, /* return */
    [18] = {0x656C6974616C6F76ULL, 0x0100000000000000ULL}, /* volatile */
    [26] = {0x00745F7261686377ULL, 0x0200000000000000ULL}, /* wchar_t */
    [36] = {0x000000004C4C554EULL, 0x0500000000000000ULL}, /* NULL */
    [41] = {0x70786574736E6F63ULL, 0x0100000000000072ULL}, /* constexpr */
    [43] = {0x5F6461657268545FULL, 0x0100006C61636F6CULL}, /* _Thread_local */
    [47] = {0x745F3631746E6975ULL, 0x0200000000000000ULL}, /* uint16_t */
    [48] = {0x007473696C5F6176ULL, 0x0200000000000000ULL}, /* va_list */
    [49] = {0x000000006F746F67ULL, 0x0100000000000000ULL}, /* goto */
    [59] = {0x7265747369676572ULL, 0x0100000000000000ULL}, /* register */
    [64] = {0x00745F3233746E69ULL, 0x0200000000000000ULL}, /* int32_t */
    [69] = {0x0000006B61657262ULL, 0x0100000000000000ULL}, /* break */
    [71] = {0x00000074736E6F63ULL, 0x0100000000000000ULL}, /* const */
    [72] = {0x666F6E67696C415FULL, 0x0100000000000000ULL}, /* _Alignof */
    [73] = {0x64656E6769736E75ULL, 0x0200000000000000ULL}, /* unsigned */
    [76] = {0x5F66666964727470ULL, 0x0200000000000074ULL}, /* ptrdiff_t */
    [84] = {0x0000000065736C65ULL, 0x0100000000000000ULL}, /* else */
    [85] = {0x007274706C6C756EULL, 0x0500000000000000ULL}, /* nullptr */
    [86] = {0x00000074726F6873ULL, 0x0200000000000000ULL}, /* short */
    [90] = {0x00000074616F6C66ULL, 0x0200000000000000ULL}, /* float */
    [96] = {0x745F727470746E69ULL, 0x0200000000000000ULL}, /* intptr_t */
    [98] = {0x0063696D6F74415FULL, 0x0100000000000000ULL}, /* _Atomic */
    [101] = {0x73616E67696C415FULL, 0x0100000000000000ULL}, /* _Alignas */
    [108] = {0x6C5F646165726874ULL, 0x010000006C61636FULL}, /* thread_local */
    [118] = {0x0000000000726F66ULL, 0x0100000000000000ULL}, /* for */
    [120] = {0x0000745F38746E69ULL, 0x0200000000000000ULL}, /* int8_t */
    [124] = {0x00006E7265747865ULL, 0x0100000000000000ULL}, /* extern */
    [127] = {0x0000745F657A6973ULL, 0x0200000000000000ULL}, /* size_t */
    [128] = {0x000000656C696877ULL, 0x0100000000000000ULL}, /* while */
    [131] = {0x00000000454C4946ULL, 0x0200000000000000ULL}, /* FILE */
    [135] = {0x0000000000746E69ULL, 0x0200000000000000ULL}, /* int */
    [136] = {0x7463697274736572ULL, 0x0100000000000000ULL}, /* restrict */
    [146] = {0x000000006D756E65ULL, 0x0100000000000000ULL}, /* enum */
    [149] = {0x745F3233746E6975ULL, 0x0200000000000000ULL}, /* uint32_t */
    [150] = {0x615F636974617473ULL, 0x0100007472657373ULL}, /* static_assert */
    [151] = {0x636972656E65475FULL, 0x0100000000000000ULL}, /* _Generic */
    [152] = {0x72757465726F4E5FULL, 0x010000000000006EULL}, /* _Noreturn */
    [154] = {0x00745F657A697373ULL, 0x0200000000000000ULL}, /* ssize_t */
    [160] = {0x0000656E696C6E69ULL, 0x0100000000000000ULL}, /* inline */
    [161] = {0x0000746375727473ULL, 0x0100000000000000ULL}, /* struct */
    [164] = {0x0000006C6F6F425FULL, 0x0200000000000000ULL}, /* _Bool */
    [165] = {0x0073616E67696C61ULL, 0x0100000000000000ULL}, /* alignas */
    [171] = {0x65756E69746E6F63ULL, 0x0100000000000000ULL}, /* continue */
    [173] = {0x0000656C62756F64ULL, 0x0200000000000000ULL}, /* double */
    [174] = {0x0000000065757274ULL, 0x0500000000000000ULL}, /* true */
    [176] = {0x0066656465707974ULL, 0x0100000000000000ULL}, /* typedef */
    [177] = {0x0000666F65707974ULL, 0x0100000000000000ULL}, /* typeof */
    [178] = {0x0000000065736163ULL, 0x0100000000000000ULL}, /* case */
    [179] = {0x78656C706D6F435FULL, 0x0200000000000000ULL}, /* _Complex */
    [181] = {0x00666F6E67696C61ULL, 0x0100000000000000ULL}, /* alignof */
    [182] = {0x00746C7561666564ULL, 0x0100000000000000ULL}, /* default */
    [185] = {0x0000006E6F696E75ULL, 0x0100000000000000ULL}, /* union */
    [190] = {0x0000636974617473ULL, 0x0100000000000000ULL}, /* static */
    [193] = {0x000000745F66666FULL, 0x0200000000000000ULL}, /* off_t */
    [195] = {0x5F6369746174535FULL, 0x0100747265737361ULL}, /* _Static_assert */
    [202] = {0x00745F3631746E69ULL, 0x0200000000000000ULL}, /* int16_t */
    [207] = {0x000064656E676973ULL, 0x0200000000000000ULL}, /* signed */
    [212] = {0x00745F38746E6975ULL, 0x0200000000000000ULL}, /* uint8_t */
    [224] = {0x745F3436746E6975ULL, 0x0200000000000000ULL}, /* uint64_t */
    [228] = {0x00000065736C6166ULL, 0x0500000000000000ULL}, /* false */
    [229] = {0x0000000064696F76ULL, 0x0200000000000000ULL}, /* void */
    [231] = {0x000000006F747561ULL, 0x0100000000000000ULL}, /* auto */
    [236] = {0x0000000000464F45ULL, 0x0500000000000000ULL}, /* EOF */
    [237] = {0x00745F3436746E69ULL, 0x0200000000000000ULL}, /* int64_t */
    [240] = {0x0000000000006669ULL, 0x0100000000000000ULL}, /* if */
    [243] = {0x0000000072616863ULL, 0x0200000000000000ULL}, /* char */
    [245] = {0x000000006C6F6F62ULL, 0x0200000000000000ULL}, /* bool */
    [246] = {0x5F727470746E6975ULL, 0x0200000000000074ULL}, /* uintptr_t */
    [251] = {0x0000000000006F64ULL, 0x0100000000000000ULL}, /* do */
    [252] = {0x00000000676E6F6CULL, 0x0200000000000000ULL}, /* long */
    [255] = {0x0000666F657A6973ULL, 0x0100000000000000ULL}, /* sizeof */
};

static const u64 MASK_LO[15] = {
    0, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFULL, 0xFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFULL,
    ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL, ~0ULL,
};
static const u64 MASK_HI[15] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFFFF, 0xFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFULL, 0xFFFFFFFFFFFFULL,
};

static inline u64 load64(const u8 *p) { u64 v; memcpy(&v, p, 8); return v; }

/* Reads 16 bytes at w. Returns the word's class, or 0 if it isn't in the table. */
static inline unsigned lookup(const u8 *w, unsigned len) {
    unsigned L = len < 14 ? len : 14;
    u64 a = load64(w) & MASK_LO[L], b = load64(w + 8) & MASK_HI[L];
    const WordEntry *e = &WORDS[((a + b * 0x9E3779B97F4A7C15ULL) * HASH_MUL) >> (64 - HASH_BITS)];
    u64 diff = (e->a ^ a) | ((e->b ^ b) << 8);
    return diff ? 0 : (unsigned)(e->b >> 56);
}

/* ---- SIMD layer ----
 * vec_eq(block, c): 64-bit mask of bytes equal to c.
 * block_masks(): everything the main loop needs about 64 bytes.
 * store_classes(): default classes, XORed with three bit planes. */
typedef struct { u64 ident, digit, x3, space, lparen, slash, star, qh; } Masks;

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VBMI__)
#include <immintrin.h>
#define LEXNET_PATH "avx512-vbmi"
typedef __m512i Block;
static inline Block vec_load(const u8 *p) { return _mm512_loadu_si512(p); }
static inline u64 vec_eq(Block v, u8 c) { return _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8((char)c)); }

static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    __m512i v = _mm512_loadu_si512(p);
    __mmask64 ascii = ~_mm512_movepi8_mask(v);
    __m512i k = _mm512_maskz_permutex2var_epi8(ascii, _mm512_load_si512(KIND), v, _mm512_load_si512(KIND + 64));
#define T(bits) _mm512_test_epi8_mask(k, _mm512_set1_epi8((char)(bits)))
    m->ident = T(K_IDENT);
    m->digit = T(K_DIGIT);
    m->x3 = T(K_IDENT | K_DOT);
    m->space = T(K_SPACE);
    m->lparen = T(K_LPAREN);
    m->slash = T(K_SLASH);
    m->star = T(K_STAR);
    m->qh = _mm512_movepi8_mask(k);
#undef T
    *def = _mm512_mask_blend_epi8(ascii, _mm512_set1_epi8(C_PUNCT),
                                  _mm512_permutex2var_epi8(_mm512_load_si512(DEFCLASS), v, _mm512_load_si512(DEFCLASS + 64)));
}
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2) {
    __m512i x = _mm512_or_si512(_mm512_maskz_set1_epi8(d0, 1),
                _mm512_or_si512(_mm512_maskz_set1_epi8(d1, 2), _mm512_maskz_set1_epi8(d2, 4)));
    _mm512_storeu_si512(o, _mm512_xor_si512(def, x));
}

#elif defined(__AVX2__)
#include <immintrin.h>
#define LEXNET_PATH "avx2"
typedef struct { __m256i a, b; } Block;
static inline Block vec_load(const u8 *p) {
    Block v = { _mm256_loadu_si256((const __m256i *)p), _mm256_loadu_si256((const __m256i *)(p + 32)) };
    return v;
}
static inline u64 mm256_mask(__m256i x) { return (uint32_t)_mm256_movemask_epi8(x); }
static inline u64 vec_eq(Block v, u8 c) {
    __m256i k = _mm256_set1_epi8((char)c);
    return mm256_mask(_mm256_cmpeq_epi8(v.a, k)) | mm256_mask(_mm256_cmpeq_epi8(v.b, k)) << 32;
}
static inline __m256i in_range(__m256i v, char lo, char hi) {   /* lo <= v <= hi, signed */
    return _mm256_and_si256(_mm256_cmpgt_epi8(v, _mm256_set1_epi8((char)(lo - 1))),
                            _mm256_cmpgt_epi8(_mm256_set1_epi8((char)(hi + 1)), v));
}
#define EQ(v, c) _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)(c)))
static inline void half_masks(__m256i v, u64 *m, __m256i *def) {
    __m256i digit = in_range(v, '0', '9');
    __m256i alpha = _mm256_or_si256(in_range(_mm256_or_si256(v, _mm256_set1_epi8(32)), 'a', 'z'), EQ(v, '_'));
    __m256i ident = _mm256_or_si256(alpha, digit);
    __m256i nlsp = in_range(v, 9, 13);
    __m256i nl = EQ(v, '\n');
    __m256i sp20 = EQ(v, ' ');
    __m256i space = _mm256_or_si256(sp20, _mm256_andnot_si256(nl, nlsp));
    __m256i dot = EQ(v, '.');
    __m256i q = _mm256_or_si256(EQ(v, '"'), _mm256_or_si256(EQ(v, '\''), EQ(v, '#')));
    m[0] = mm256_mask(ident);
    m[1] = mm256_mask(digit);
    m[2] = mm256_mask(_mm256_or_si256(ident, dot));
    m[3] = mm256_mask(space);
    m[4] = mm256_mask(EQ(v, '('));
    m[5] = mm256_mask(EQ(v, '/'));
    m[6] = mm256_mask(EQ(v, '*'));
    m[7] = mm256_mask(q);
    __m256i d = _mm256_blendv_epi8(_mm256_set1_epi8(C_PUNCT), _mm256_set1_epi8(C_IDENT), ident);
    *def = _mm256_andnot_si256(_mm256_or_si256(nlsp, sp20), d);
}
static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    Block v = vec_load(p);
    u64 x[8], y[8];
    half_masks(v.a, x, &def->a);
    half_masks(v.b, y, &def->b);
    m->ident = x[0] | y[0] << 32; m->digit = x[1] | y[1] << 32; m->x3 = x[2] | y[2] << 32;
    m->space = x[3] | y[3] << 32; m->lparen = x[4] | y[4] << 32; m->slash = x[5] | y[5] << 32;
    m->star = x[6] | y[6] << 32; m->qh = x[7] | y[7] << 32;
}
static inline __m256i expand_mask(uint32_t m) {
    __m256i v = _mm256_set1_epi32((int)m);
    const __m256i shuf = _mm256_setr_epi8(0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3);
    const __m256i bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    return _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, shuf), bits), bits);
}
static inline __m256i planes(uint32_t d0, uint32_t d1, uint32_t d2) {
    return _mm256_or_si256(_mm256_and_si256(expand_mask(d0), _mm256_set1_epi8(1)),
           _mm256_or_si256(_mm256_and_si256(expand_mask(d1), _mm256_set1_epi8(2)),
                           _mm256_and_si256(expand_mask(d2), _mm256_set1_epi8(4))));
}
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2) {
    _mm256_storeu_si256((__m256i *)o, _mm256_xor_si256(def.a, planes((uint32_t)d0, (uint32_t)d1, (uint32_t)d2)));
    _mm256_storeu_si256((__m256i *)(o + 32), _mm256_xor_si256(def.b, planes((uint32_t)(d0 >> 32), (uint32_t)(d1 >> 32), (uint32_t)(d2 >> 32))));
}

#elif defined(__ARM_NEON) && defined(__aarch64__) && !defined(LEXNET_FORCE_SCALAR)
#include <arm_neon.h>
#define LEXNET_PATH "neon"
typedef struct { uint8x16_t v[4]; } Block;
static inline Block vec_load(const u8 *p) {
    Block b = { { vld1q_u8(p), vld1q_u8(p + 16), vld1q_u8(p + 32), vld1q_u8(p + 48) } };
    return b;
}
static inline u64 movemask4(uint8x16_t a, uint8x16_t b, uint8x16_t c, uint8x16_t d) {
    /* Inputs are 0x00/0xFF per byte. Interleaves one bit per byte into 64 bits. */
    uint8x16_t t0 = vsriq_n_u8(b, a, 1), t1 = vsriq_n_u8(d, c, 1);
    uint8x16_t t2 = vsriq_n_u8(t1, t0, 2);
    uint8x16_t t3 = vsriq_n_u8(t2, t2, 4);
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(t3), 4)), 0);
}
static inline u64 vec_eq(Block x, u8 c) {
    uint8x16_t k = vdupq_n_u8(c);
    return movemask4(vceqq_u8(x.v[0], k), vceqq_u8(x.v[1], k), vceqq_u8(x.v[2], k), vceqq_u8(x.v[3], k));
}
static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    Block x = vec_load(p);
    uint8x16_t id[4], dg[4], dt[4], sp[4], lp[4], sl[4], st[4], qh[4];
    for (int i = 0; i < 4; i++) {
        uint8x16_t v = x.v[i];
        uint8x16_t digit = vcltq_u8(vsubq_u8(v, vdupq_n_u8('0')), vdupq_n_u8(10));
        uint8x16_t lower = vcltq_u8(vsubq_u8(vorrq_u8(v, vdupq_n_u8(32)), vdupq_n_u8('a')), vdupq_n_u8(26));
        uint8x16_t ident = vorrq_u8(digit, vorrq_u8(lower, vceqq_u8(v, vdupq_n_u8('_'))));
        uint8x16_t ws = vcltq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(5));
        uint8x16_t sp20 = vceqq_u8(v, vdupq_n_u8(' '));
        uint8x16_t nl = vceqq_u8(v, vdupq_n_u8('\n'));
        id[i] = ident; dg[i] = digit;
        dt[i] = vorrq_u8(ident, vceqq_u8(v, vdupq_n_u8('.')));
        sp[i] = vorrq_u8(sp20, vbicq_u8(ws, nl));
        lp[i] = vceqq_u8(v, vdupq_n_u8('('));
        sl[i] = vceqq_u8(v, vdupq_n_u8('/'));
        st[i] = vceqq_u8(v, vdupq_n_u8('*'));
        qh[i] = vorrq_u8(vceqq_u8(v, vdupq_n_u8('"')), vorrq_u8(vceqq_u8(v, vdupq_n_u8('\'')), vceqq_u8(v, vdupq_n_u8('#'))));
        uint8x16_t d = vbslq_u8(ident, vdupq_n_u8(C_IDENT), vdupq_n_u8(C_PUNCT));
        def->v[i] = vbicq_u8(d, vorrq_u8(ws, sp20));
    }
#define MM(a) movemask4(a[0], a[1], a[2], a[3])
    m->ident = MM(id); m->digit = MM(dg); m->x3 = MM(dt); m->space = MM(sp);
    m->lparen = MM(lp); m->slash = MM(sl); m->star = MM(st); m->qh = MM(qh);
#undef MM
}
static inline uint8x16_t bits_to_bytes(uint16_t m, u8 val) {
    static const u8 tbl[16] = {1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
    uint8x8_t lo = vdup_n_u8((u8)m), hi = vdup_n_u8((u8)(m >> 8));
    uint8x16_t b = vtstq_u8(vcombine_u8(lo, hi), vld1q_u8(tbl));
    return vandq_u8(b, vdupq_n_u8(val));
}
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2) {
    for (int i = 0; i < 4; i++) {
        uint16_t a = (uint16_t)(d0 >> (16 * i)), b = (uint16_t)(d1 >> (16 * i)), c = (uint16_t)(d2 >> (16 * i));
        uint8x16_t x = vorrq_u8(bits_to_bytes(a, 1), vorrq_u8(bits_to_bytes(b, 2), bits_to_bytes(c, 4)));
        vst1q_u8(o + 16 * i, veorq_u8(def.v[i], x));
    }
}

#else
#define LEXNET_PATH "portable"
typedef const u8 *Block;
static inline Block vec_load(const u8 *p) { return p; }
static inline u64 vec_eq(Block v, u8 c) {
    u64 m = 0;
    for (int i = 0; i < 64; i++) m |= (u64)(v[i] == c) << i;
    return m;
}
static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    Masks x = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 64; i++) {
        u64 bit = 1ULL << i;
        unsigned k = KIND[p[i]];
        if (k & K_IDENT) x.ident |= bit;
        if (k & K_DIGIT) x.digit |= bit;
        if (k & (K_IDENT | K_DOT)) x.x3 |= bit;
        if (k & K_SPACE) x.space |= bit;
        if (k & K_LPAREN) x.lparen |= bit;
        if (k & K_SLASH) x.slash |= bit;
        if (k & K_STAR) x.star |= bit;
        if (k & K_QH) x.qh |= bit;
    }
    *m = x;
    *def = p;
}
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2) {
    for (int i = 0; i < 64; i++)
        o[i] = (u8)(DEFCLASS[def[i]] ^ ((d0 >> i & 1) | (d1 >> i & 1) << 1 | (d2 >> i & 1) << 2));
}
#endif

/* ---- jumps out of the block loop ---- */

/* First byte in [p,end) equal to a, b or c. */
static inline const u8 *find3(const u8 *p, const u8 *end, u8 a, u8 b, u8 c) {
    while (end - p >= 64) {
        Block v = vec_load(p);
        u64 m = vec_eq(v, a) | vec_eq(v, b) | vec_eq(v, c);
        if (m) return p + __builtin_ctzll(m);
        p += 64;
    }
    while (p < end && *p != a && *p != b && *p != c) p++;
    return p;
}

/* Fills a class for a run, usually short. */
static inline void fill(u8 *o, unsigned c, size_t n) { memset(o, (int)c, n); }

/* A string or char literal, p at the opening quote. Returns the byte after it. The newline
 * ending an unterminated literal isn't part of it, not even after a backslash. */
static const u8 *literal(const u8 *src, const u8 *p, const u8 *end, u8 *cls) {
    u8 quote = *p;
    const u8 *i = p + 1, *stop;
    for (;;) {
        i = find3(i, end, quote, '\\', '\n');
        if (i >= end) { stop = end; break; }
        if (*i == quote) { stop = i + 1; break; }
        if (*i == '\n') { stop = i; break; }
        if (i + 1 >= end) { stop = end; break; }
        if (i[1] == '\n') { stop = i + 1; break; }
        i += 2;
    }
    fill(cls + (p - src), quote == '"' ? C_STRING : C_CHAR, (size_t)(stop - p));
    return stop;
}

/* A block comment, p at its '/'. The search starts after the opening star, so a slash
 * right behind it can't close the comment. *pp ends if a bare newline passes. */
static const u8 *block_comment(const u8 *src, const u8 *p, const u8 *end, u8 *cls, bool *pp) {
    const u8 *i = p + 2, *stop = end;
    while (end - i >= 65) {
        u64 m = vec_eq(vec_load(i), '*') & vec_eq(vec_load(i + 1), '/');
        if (m) { stop = i + __builtin_ctzll(m) + 2; goto found; }
        i += 64;
    }
    for (; i + 1 < end; i++)
        if (i[0] == '*' && i[1] == '/') { stop = i + 2; break; }
found:
    fill(cls + (p - src), C_COMMENT, (size_t)(stop - p));
    if (*pp) {
        const u8 *nl = p;
        while ((nl = memchr(nl, '\n', (size_t)(stop - nl))) != NULL) {
            if (nl[-1] != '\\') { *pp = false; break; }
            nl++;
        }
    }
    return stop;
}

static inline const u8 *line_end(const u8 *p, const u8 *end) {
    const u8 *nl = memchr(p, '\n', (size_t)(end - p));
    return nl ? nl : end;
}

/* First quote, comment opener or newline inside a directive. */
static inline const u8 *directive_next(const u8 *p, const u8 *end) {
    while (end - p >= 65) {
        Block v = vec_load(p), w = vec_load(p + 1);
        u64 m = vec_eq(v, '"') | vec_eq(v, '\'') | vec_eq(v, '\n') |
                (vec_eq(v, '/') & (vec_eq(w, '*') | vec_eq(w, '/')));
        if (m) return p + __builtin_ctzll(m);
        p += 64;
    }
    for (; p < end; p++) {
        if (*p == '"' || *p == '\'' || *p == '\n') return p;
        if (*p == '/' && p + 1 < end && (p[1] == '*' || p[1] == '/')) return p;
    }
    return end;
}

/* A directive, p at its '#'. Everything is class 9 except literals and comments; it ends at
 * a newline, unless a backslash is right before that newline. */
static const u8 *directive(const u8 *src, const u8 *p, const u8 *end, u8 *cls) {
    const u8 *seg = p;
    for (;;) {
        const u8 *q = directive_next(seg, end);
        fill(cls + (seg - src), C_PREPROC, (size_t)(q - seg));
        if (q >= end) return end;
        if (*q == '\n') {
            if (q[-1] == '\\') { cls[q - src] = C_PREPROC; seg = q + 1; continue; }
            cls[q - src] = C_SPACE;
            return q + 1;
        }
        if (*q != '/') { seg = literal(src, q, end, cls); continue; }
        if (q[1] == '/') {
            const u8 *nl = line_end(q, end);
            fill(cls + (q - src), C_COMMENT, (size_t)(nl - q));
            seg = nl;
            continue;
        }
        bool pp = true;
        seg = block_comment(src, q, end, cls, &pp);
        if (!pp) return seg;
    }
}

/* Only whitespace since the last newline? Reads raw bytes: the network's flag ignores comments. */
static inline bool at_line_start(const u8 *src, const u8 *p) {
    while (p > src && (KIND[p[-1]] & K_SPACE)) p--;
    return p == src || p[-1] == '\n';
}

/* The rare one: a run of word chars and spaces too long for a block. */
static const u8 *long_run(const u8 *src, const u8 *p, const u8 *end, u8 *cls) {
    while (p < end) {
        unsigned k = KIND[*p];
        if (k & K_SPACE) {
            const u8 *i = p;
            while (i < end && (KIND[*i] & K_SPACE)) i++;
            fill(cls + (p - src), C_SPACE, (size_t)(i - p));
            p = i;
        } else if (k & K_DIGIT) {
            const u8 *i = p;
            while (i < end && (KIND[*i] & (K_IDENT | K_DOT))) i++;
            fill(cls + (p - src), C_CONSTANT, (size_t)(i - p));
            p = i;
        } else if (k & K_IDENT) {
            const u8 *i = p;
            while (i < end && (KIND[*i] & K_IDENT)) i++;
            size_t len = (size_t)(i - p);
            u8 key[16] = {0};
            memcpy(key, p, len < 14 ? len : 14);
            unsigned c = lookup(key, len < 14 ? (unsigned)len : 14);
            if (!c) {
                const u8 *a = i;
                while (a < end && (KIND[*a] & K_SPACE)) a++;
                c = (a < end && *a == '(') ? C_FUNCTION : C_IDENT;
            }
            fill(cls + (p - src), c, len);
            p = i;
        } else if (k & K_DOT) {
            cls[p - src] = C_PUNCT;
            p++;
        } else {
            break;
        }
    }
    return p;
}

/* ---- the main loop ----
 * One block: find where it has to stop, label every word and number before that point, write
 * the classes, then advance. Each byte is either finished here or relabelled by the next
 * step, so storing a whole block is fine even when only part of it counts. */
static inline unsigned code_block(const u8 *b, u8 *o, const u8 *real, const u8 *end, const Masks *mp, Block def) {
    (void)real; (void)end;
    Masks m = *mp;
    u64 runs = m.x3 | m.space | 1ULL << 63;
    unsigned stop = 64 - (unsigned)(~runs ? __builtin_clzll(~runs) : 64);
    u64 special = m.qh | (m.slash & ((m.slash | m.star) >> 1));
    unsigned q = special ? (unsigned)__builtin_ctzll(special) : 64;
    unsigned limit = q < stop ? q : stop;
    if (limit == 0) return 0;

    u64 region = (1ULL << limit) - 1;
    u64 starts = m.ident & ~(m.ident << 1);
    u64 num_starts = starts & m.digit;
    u64 number = (m.x3 & ~(m.x3 + num_starts)) | num_starts;   /* ripple carry over each run */
    u64 word = m.ident & ~number & region;
    u64 ws = word & ~(word << 1), we = word & ~(word >> 1);

    /* Classes are XORed against 4, so a plain identifier costs nothing. */
    u64 d0 = number & m.ident & region, d1 = 0, d2 = 0;
    while (ws) {
        unsigned s = (unsigned)__builtin_ctzll(ws), e = (unsigned)__builtin_ctzll(we);
        ws &= ws - 1;
        we &= we - 1;
        unsigned c = lookup(b + s, e - s + 1);
        unsigned a = e + 1;
        a += (unsigned)__builtin_ctzll(~(m.space >> a));
        unsigned fn = (unsigned)(m.lparen >> a) & 1;
        unsigned x = (c ? c : 4 - fn) ^ 4;
        u64 w = (2ULL << e) - (1ULL << s);
        d0 |= w & -(u64)(x & 1);
        d1 |= w & -(u64)(x >> 1 & 1);
        d2 |= w & -(u64)(x >> 2 & 1);
    }
    store_classes(o, def, d0, d1, d2);
    /* A dot inside a number starts from 10, not 4, so it can't be XORed. They're rare. */
    u64 dots = number & ~m.ident & region;
    while (UNLIKELY(dots)) {
        o[__builtin_ctzll(dots)] = C_CONSTANT;
        dots &= dots - 1;
    }
    return limit;
}

void lexnet_classify(const u8 *src, size_t n, u8 *cls) {
    const u8 *p = src, *end = src + n;
    while (p < end) {
        size_t left = (size_t)(end - p);
        unsigned limit;
        if (LIKELY(left >= 80)) {
            Masks m;
            Block def;
            block_masks(p, &m, &def);
            limit = code_block(p, cls + (p - src), p, end, &m, def);
        } else {
            /* The last few bytes go through a zero-padded copy, and NUL behaves like end of file. */
            u8 in[96] = {0}, out[64];
            memcpy(in, p, left);
            Masks m;
            Block def;
            block_masks(in, &m, &def);
            limit = code_block(in, out, p, end, &m, def);
            if (limit > left) limit = (unsigned)left;
            memcpy(cls + (p - src), out, limit);
        }
        p += limit;
        if (p >= end) break;
        if (limit && (KIND[*p] & (K_IDENT | K_DOT | K_SPACE))) continue;   /* block ran out */

        unsigned k = KIND[*p];
        if (k & (K_IDENT | K_DOT | K_SPACE)) {
            p = long_run(src, p, end, cls);
        } else if (*p == '"' || *p == '\'') {
            p = literal(src, p, end, cls);
        } else if (*p == '#') {
            if (at_line_start(src, p)) {
                p = directive(src, p, end, cls);
            } else {
                cls[p - src] = C_PUNCT;
                p++;
            }
        } else if (*p == '/' && p + 1 < end && p[1] == '/') {
            const u8 *nl = line_end(p, end);
            fill(cls + (p - src), C_COMMENT, (size_t)(nl - p));
            p = nl;
        } else if (*p == '/' && p + 1 < end && p[1] == '*') {
            bool pp = false;
            p = block_comment(src, p, end, cls, &pp);
        }
        /* Anything else: the byte stopped a block (bit 63) without being special, or just
         * follows one. The next block handles it. */
    }
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
    if (!path) { fprintf(stderr, "usage: lexnet_fast [--classes|--bench N] file.c\n"); return 2; }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    u8 *src = malloc(n + 1), *cls = malloc(n + 1);
    if (fread(src, 1, n, f) != n) return 1;
    fclose(f);

    if (mode == 2) {
        double best = 1e30;
        for (int i = 0; i < reps; i++) {
            struct timespec a, b;
            clock_gettime(CLOCK_MONOTONIC, &a);
            lexnet_classify(src, n, cls);
            clock_gettime(CLOCK_MONOTONIC, &b);
            double sec = (double)(b.tv_sec - a.tv_sec) + (double)(b.tv_nsec - a.tv_nsec) * 1e-9;
            if (sec < best) best = sec;
        }
        printf("%s: %zu chars, best of %d: %.2f ms  (%.3f ns/char, %.0f MB/s)\n",
               LEXNET_PATH, n, reps, best * 1e3, best / (double)n * 1e9, (double)n / best / 1e6);
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
