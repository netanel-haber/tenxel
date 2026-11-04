/* tenxel.c - a C lexer for syntax highlighting: one class per byte, the same as clang.
 *
 * It started as a sort of port of lexnet by alpindale (https://huggingface.co/alpindale/lexnet), a hand-built neural
 * network that highlights C, and keeps its idea and its 11 classes. It is not affiliated with lexnet or alpindale.
 *
 * Token boundaries match clang 19's raw lexer (-std=gnu23, x86-64 Linux): line splices anywhere, digraphs, string
 * prefixes, raw strings, C23 numbers, UCNs and UTF-8 identifiers. On top of clang's tokens it marks clang's keywords
 * and builtin types, directives (a # or %: that is the first token on its line, up to the next newline outside a
 * comment) and functions (identifiers whose next token, past spaces and comments, is "("). oracle.py checks all of
 * this against clang. The one known gap: a named UCN (\N{...}) is always treated as invalid.
 *
 * Plain code goes through in 64-byte blocks: SIMD turns a block into bitmasks, words come from bit operations and
 * one perfect-hash lookup each, numbers from one addition, and the classes are stored a block at a time. Literals,
 * comments and directives are scanned with SIMD searches. Anything rarer (splices, UCNs, UTF-8, literal prefixes,
 * digit separators, exponent signs) goes through lex_token, which lexes one token the plain way and is the whole
 * definition of the output.
 *
 * Build: cc -O3 -march=native -o tenxel tenxel.c
 * Usage: tenxel file.c, which prints the file colored. tenxel_classify() is the whole interface.
 * SIMD: AVX2, or a portable fallback without it, chosen by the compiler's target.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint8_t u8;
typedef uint64_t u64;

enum Class {
    C_SPACE = 0,
    C_KEYWORD = 1,
    C_TYPE = 2,
    C_FUNCTION = 3,
    C_IDENT = 4,
    C_CONSTANT = 5,
    C_STRING = 6,
    C_CHAR = 7,
    C_COMMENT = 8,
    C_PREPROC = 9,
    C_PUNCT = 10,
};

/* ---- keywords: what clang 19 treats as a keyword with -std=gnu23 on x86-64 Linux ----
 * C_TYPE marks the builtin type specifiers, C_CONSTANT the literal keywords. Each word sits in the slot its first 16
 * bytes hash to (keyword_slot), and no two share a slot. */
typedef struct {
    char word[39];
    u8 cls;
} Keyword;

#define KEYWORD_MAX 36
#define KEYWORD_SLOTS 512
static const Keyword KEYWORDS[KEYWORD_SLOTS] = {
    [7] = {"__fastcall", C_KEYWORD},
    [9] = {"__builtin_offsetof", C_KEYWORD},
    [18] = {"__builtin_FUNCTION", C_KEYWORD},
    [23] = {"struct", C_KEYWORD},
    [24] = {"__complex__", C_TYPE},
    [32] = {"__label__", C_KEYWORD},
    [35] = {"__int128", C_TYPE},
    [38] = {"__typeof_unqual", C_KEYWORD},
    [42] = {"_Imaginary", C_TYPE},
    [50] = {"__builtin_omp_required_simd_align", C_KEYWORD},
    [51] = {"long", C_TYPE},
    [52] = {"__typeof_unqual__", C_KEYWORD},
    [53] = {"_Alignof", C_KEYWORD},
    [58] = {"__builtin_bit_cast", C_KEYWORD},
    [61] = {"__volatile__", C_KEYWORD},
    [65] = {"__attribute", C_KEYWORD},
    [74] = {"bool", C_TYPE},
    [75] = {"__module_private__", C_KEYWORD},
    [81] = {"_Atomic", C_KEYWORD},
    [94] = {"_Decimal128", C_TYPE},
    [100] = {"__builtin_types_compatible_p", C_KEYWORD},
    [102] = {"__builtin_convertvector", C_KEYWORD},
    [105] = {"_Nullable", C_KEYWORD},
    [108] = {"nullptr", C_CONSTANT},
    [111] = {"__builtin_COLUMN", C_KEYWORD},
    [112] = {"void", C_TYPE},
    [114] = {"float", C_TYPE},
    [117] = {"restrict", C_KEYWORD},
    [119] = {"while", C_KEYWORD},
    [120] = {"else", C_KEYWORD},
    [122] = {"__real", C_KEYWORD},
    [123] = {"__builtin_choose_expr", C_KEYWORD},
    [129] = {"__extension__", C_KEYWORD},
    [131] = {"__restrict", C_KEYWORD},
    [133] = {"__attribute__", C_KEYWORD},
    [134] = {"__is_bitwise_cloneable", C_KEYWORD},
    [135] = {"__objc_no", C_CONSTANT},
    [137] = {"__builtin_vectorelements", C_KEYWORD},
    [138] = {"_Thread_local", C_KEYWORD},
    [139] = {"__float128", C_TYPE},
    [146] = {"inline", C_KEYWORD},
    [152] = {"_Decimal64", C_TYPE},
    [156] = {"alignas", C_KEYWORD},
    [162] = {"continue", C_KEYWORD},
    [164] = {"_BitInt", C_TYPE},
    [166] = {"typeof_unqual", C_KEYWORD},
    [170] = {"__asm", C_KEYWORD},
    [171] = {"unsigned", C_TYPE},
    [172] = {"int", C_TYPE},
    [176] = {"false", C_CONSTANT},
    [182] = {"for", C_KEYWORD},
    [190] = {"goto", C_KEYWORD},
    [191] = {"__stdcall", C_KEYWORD},
    [192] = {"__inline", C_KEYWORD},
    [194] = {"_Complex", C_TYPE},
    [196] = {"__builtin_FILE", C_KEYWORD},
    [201] = {"_Decimal32", C_TYPE},
    [205] = {"__builtin_available", C_KEYWORD},
    [207] = {"_Generic", C_KEYWORD},
    [208] = {"__regcall", C_KEYWORD},
    [211] = {"do", C_KEYWORD},
    [212] = {"__imag", C_KEYWORD},
    [214] = {"default", C_KEYWORD},
    [215] = {"__alignof", C_KEYWORD},
    [220] = {"__thread", C_KEYWORD},
    [222] = {"register", C_KEYWORD},
    [230] = {"__builtin_ptrauth_type_discriminator", C_KEYWORD},
    [234] = {"_Static_assert", C_KEYWORD},
    [238] = {"if", C_KEYWORD},
    [241] = {"__builtin_LINE", C_KEYWORD},
    [250] = {"case", C_KEYWORD},
    [253] = {"__vectorcall", C_KEYWORD},
    [256] = {"__restrict__", C_KEYWORD},
    [267] = {"__inline__", C_KEYWORD},
    [283] = {"_Nonnull", C_KEYWORD},
    [305] = {"break", C_KEYWORD},
    [311] = {"char", C_TYPE},
    [312] = {"typedef", C_KEYWORD},
    [315] = {"volatile", C_KEYWORD},
    [322] = {"_Nullable_result", C_KEYWORD},
    [326] = {"asm", C_KEYWORD},
    [333] = {"__alignof__", C_KEYWORD},
    [334] = {"__real__", C_KEYWORD},
    [336] = {"__signed", C_TYPE},
    [338] = {"_Noreturn", C_KEYWORD},
    [345] = {"__typeof", C_KEYWORD},
    [349] = {"static", C_KEYWORD},
    [350] = {"__is_nothrow_destructible", C_KEYWORD},
    [352] = {"__pascal", C_KEYWORD},
    [358] = {"__const", C_KEYWORD},
    [368] = {"true", C_CONSTANT},
    [369] = {"thread_local", C_KEYWORD},
    [371] = {"_Bool", C_TYPE},
    [377] = {"__private_extern__", C_KEYWORD},
    [379] = {"auto", C_KEYWORD},
    [390] = {"__objc_yes", C_CONSTANT},
    [393] = {"_Null_unspecified", C_KEYWORD},
    [395] = {"alignof", C_KEYWORD},
    [402] = {"__FUNCTION__", C_KEYWORD},
    [404] = {"typeof", C_KEYWORD},
    [405] = {"constexpr", C_KEYWORD},
    [410] = {"__signed__", C_TYPE},
    [417] = {"__complex", C_TYPE},
    [419] = {"__typeof__", C_KEYWORD},
    [420] = {"__fp16", C_TYPE},
    [421] = {"signed", C_TYPE},
    [422] = {"union", C_KEYWORD},
    [423] = {"sizeof", C_KEYWORD},
    [424] = {"__imag__", C_KEYWORD},
    [429] = {"__bf16", C_TYPE},
    [433] = {"__cdecl", C_KEYWORD},
    [434] = {"double", C_TYPE},
    [436] = {"return", C_KEYWORD},
    [438] = {"__const__", C_KEYWORD},
    [439] = {"__asm__", C_KEYWORD},
    [440] = {"short", C_TYPE},
    [444] = {"_Float16", C_TYPE},
    [446] = {"__builtin_va_arg", C_KEYWORD},
    [447] = {"__thiscall", C_KEYWORD},
    [449] = {"__volatile", C_KEYWORD},
    [451] = {"switch", C_KEYWORD},
    [453] = {"_Alignas", C_KEYWORD},
    [458] = {"__funcref", C_KEYWORD},
    [460] = {"_ExtInt", C_TYPE},
    [467] = {"const", C_KEYWORD},
    [468] = {"__ibm128", C_TYPE},
    [470] = {"__PRETTY_FUNCTION__", C_KEYWORD},
    [478] = {"static_assert", C_KEYWORD},
    [490] = {"__auto_type", C_KEYWORD},
    [491] = {"__func__", C_KEYWORD},
    [496] = {"__builtin_FILE_NAME", C_KEYWORD},
    [501] = {"__is_destructible", C_KEYWORD},
    [510] = {"extern", C_KEYWORD},
    [511] = {"enum", C_KEYWORD},
};

static u64 load64(const void *p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

static unsigned keyword_slot(u64 a, u64 b) {
    return (unsigned)(((a + b * 0x9E3779B97F4A7C15ULL) * 0x391E370324CC3535ULL) >> 55);
}

/* The class of the word of len bytes at w: C_KEYWORD, C_TYPE, C_CONSTANT or C_IDENT. Reads 16 bytes at w. */
static inline int keyword_class(const u8 *w, size_t len) {
    /* MASK[8 + k] keeps the low k bytes of a word, for k from -8 to 16 */
    static const u64 MASK[25] = {
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
        0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x00000000000000FF,
        0x000000000000FFFF, 0x0000000000FFFFFF, 0x00000000FFFFFFFF, 0x000000FFFFFFFFFF, 0x0000FFFFFFFFFFFF,
        0x00FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    if (len > KEYWORD_MAX)
        return C_IDENT;
    size_t head = len < 16 ? len : 16;
    u64 a = load64(w) & MASK[8 + head];
    u64 b = load64(w + 8) & MASK[head];
    const Keyword *k = &KEYWORDS[keyword_slot(a, b)];
    if (load64(k->word) != a || load64(k->word + 8) != b)
        return C_IDENT;
    if (len >= 16 && (memcmp(k->word + 16, w + 16, len - 16) || k->word[len]))
        return C_IDENT; /* a shorter word compares equal on its zero padding; a 16-byte one doesn't */
    return k->cls;
}

/* ---- Unicode: which code points can start an identifier, and which are whitespace ----
 * From clang/lib/Lex/UnicodeCharSets.h at llvmorg-19.1.7: ID_START is XIDStartRanges (Unicode 15.1) merged with
 * MathematicalNotationProfileIDStartRanges, without ASCII; UNICODE_SPACE is UnicodeWhitespaceCharRanges. */
typedef struct {
    uint32_t lo, hi;
} CodeRange;
static const CodeRange ID_START[658] = {
    {0x000AA, 0x000AA}, {0x000B5, 0x000B5}, {0x000BA, 0x000BA}, {0x000C0, 0x000D6}, {0x000D8, 0x000F6},
    {0x000F8, 0x002C1}, {0x002C6, 0x002D1}, {0x002E0, 0x002E4}, {0x002EC, 0x002EC}, {0x002EE, 0x002EE},
    {0x00370, 0x00374}, {0x00376, 0x00377}, {0x0037B, 0x0037D}, {0x0037F, 0x0037F}, {0x00386, 0x00386},
    {0x00388, 0x0038A}, {0x0038C, 0x0038C}, {0x0038E, 0x003A1}, {0x003A3, 0x003F5}, {0x003F7, 0x00481},
    {0x0048A, 0x0052F}, {0x00531, 0x00556}, {0x00559, 0x00559}, {0x00560, 0x00588}, {0x005D0, 0x005EA},
    {0x005EF, 0x005F2}, {0x00620, 0x0064A}, {0x0066E, 0x0066F}, {0x00671, 0x006D3}, {0x006D5, 0x006D5},
    {0x006E5, 0x006E6}, {0x006EE, 0x006EF}, {0x006FA, 0x006FC}, {0x006FF, 0x006FF}, {0x00710, 0x00710},
    {0x00712, 0x0072F}, {0x0074D, 0x007A5}, {0x007B1, 0x007B1}, {0x007CA, 0x007EA}, {0x007F4, 0x007F5},
    {0x007FA, 0x007FA}, {0x00800, 0x00815}, {0x0081A, 0x0081A}, {0x00824, 0x00824}, {0x00828, 0x00828},
    {0x00840, 0x00858}, {0x00860, 0x0086A}, {0x00870, 0x00887}, {0x00889, 0x0088E}, {0x008A0, 0x008C9},
    {0x00904, 0x00939}, {0x0093D, 0x0093D}, {0x00950, 0x00950}, {0x00958, 0x00961}, {0x00971, 0x00980},
    {0x00985, 0x0098C}, {0x0098F, 0x00990}, {0x00993, 0x009A8}, {0x009AA, 0x009B0}, {0x009B2, 0x009B2},
    {0x009B6, 0x009B9}, {0x009BD, 0x009BD}, {0x009CE, 0x009CE}, {0x009DC, 0x009DD}, {0x009DF, 0x009E1},
    {0x009F0, 0x009F1}, {0x009FC, 0x009FC}, {0x00A05, 0x00A0A}, {0x00A0F, 0x00A10}, {0x00A13, 0x00A28},
    {0x00A2A, 0x00A30}, {0x00A32, 0x00A33}, {0x00A35, 0x00A36}, {0x00A38, 0x00A39}, {0x00A59, 0x00A5C},
    {0x00A5E, 0x00A5E}, {0x00A72, 0x00A74}, {0x00A85, 0x00A8D}, {0x00A8F, 0x00A91}, {0x00A93, 0x00AA8},
    {0x00AAA, 0x00AB0}, {0x00AB2, 0x00AB3}, {0x00AB5, 0x00AB9}, {0x00ABD, 0x00ABD}, {0x00AD0, 0x00AD0},
    {0x00AE0, 0x00AE1}, {0x00AF9, 0x00AF9}, {0x00B05, 0x00B0C}, {0x00B0F, 0x00B10}, {0x00B13, 0x00B28},
    {0x00B2A, 0x00B30}, {0x00B32, 0x00B33}, {0x00B35, 0x00B39}, {0x00B3D, 0x00B3D}, {0x00B5C, 0x00B5D},
    {0x00B5F, 0x00B61}, {0x00B71, 0x00B71}, {0x00B83, 0x00B83}, {0x00B85, 0x00B8A}, {0x00B8E, 0x00B90},
    {0x00B92, 0x00B95}, {0x00B99, 0x00B9A}, {0x00B9C, 0x00B9C}, {0x00B9E, 0x00B9F}, {0x00BA3, 0x00BA4},
    {0x00BA8, 0x00BAA}, {0x00BAE, 0x00BB9}, {0x00BD0, 0x00BD0}, {0x00C05, 0x00C0C}, {0x00C0E, 0x00C10},
    {0x00C12, 0x00C28}, {0x00C2A, 0x00C39}, {0x00C3D, 0x00C3D}, {0x00C58, 0x00C5A}, {0x00C5D, 0x00C5D},
    {0x00C60, 0x00C61}, {0x00C80, 0x00C80}, {0x00C85, 0x00C8C}, {0x00C8E, 0x00C90}, {0x00C92, 0x00CA8},
    {0x00CAA, 0x00CB3}, {0x00CB5, 0x00CB9}, {0x00CBD, 0x00CBD}, {0x00CDD, 0x00CDE}, {0x00CE0, 0x00CE1},
    {0x00CF1, 0x00CF2}, {0x00D04, 0x00D0C}, {0x00D0E, 0x00D10}, {0x00D12, 0x00D3A}, {0x00D3D, 0x00D3D},
    {0x00D4E, 0x00D4E}, {0x00D54, 0x00D56}, {0x00D5F, 0x00D61}, {0x00D7A, 0x00D7F}, {0x00D85, 0x00D96},
    {0x00D9A, 0x00DB1}, {0x00DB3, 0x00DBB}, {0x00DBD, 0x00DBD}, {0x00DC0, 0x00DC6}, {0x00E01, 0x00E30},
    {0x00E32, 0x00E32}, {0x00E40, 0x00E46}, {0x00E81, 0x00E82}, {0x00E84, 0x00E84}, {0x00E86, 0x00E8A},
    {0x00E8C, 0x00EA3}, {0x00EA5, 0x00EA5}, {0x00EA7, 0x00EB0}, {0x00EB2, 0x00EB2}, {0x00EBD, 0x00EBD},
    {0x00EC0, 0x00EC4}, {0x00EC6, 0x00EC6}, {0x00EDC, 0x00EDF}, {0x00F00, 0x00F00}, {0x00F40, 0x00F47},
    {0x00F49, 0x00F6C}, {0x00F88, 0x00F8C}, {0x01000, 0x0102A}, {0x0103F, 0x0103F}, {0x01050, 0x01055},
    {0x0105A, 0x0105D}, {0x01061, 0x01061}, {0x01065, 0x01066}, {0x0106E, 0x01070}, {0x01075, 0x01081},
    {0x0108E, 0x0108E}, {0x010A0, 0x010C5}, {0x010C7, 0x010C7}, {0x010CD, 0x010CD}, {0x010D0, 0x010FA},
    {0x010FC, 0x01248}, {0x0124A, 0x0124D}, {0x01250, 0x01256}, {0x01258, 0x01258}, {0x0125A, 0x0125D},
    {0x01260, 0x01288}, {0x0128A, 0x0128D}, {0x01290, 0x012B0}, {0x012B2, 0x012B5}, {0x012B8, 0x012BE},
    {0x012C0, 0x012C0}, {0x012C2, 0x012C5}, {0x012C8, 0x012D6}, {0x012D8, 0x01310}, {0x01312, 0x01315},
    {0x01318, 0x0135A}, {0x01380, 0x0138F}, {0x013A0, 0x013F5}, {0x013F8, 0x013FD}, {0x01401, 0x0166C},
    {0x0166F, 0x0167F}, {0x01681, 0x0169A}, {0x016A0, 0x016EA}, {0x016EE, 0x016F8}, {0x01700, 0x01711},
    {0x0171F, 0x01731}, {0x01740, 0x01751}, {0x01760, 0x0176C}, {0x0176E, 0x01770}, {0x01780, 0x017B3},
    {0x017D7, 0x017D7}, {0x017DC, 0x017DC}, {0x01820, 0x01878}, {0x01880, 0x018A8}, {0x018AA, 0x018AA},
    {0x018B0, 0x018F5}, {0x01900, 0x0191E}, {0x01950, 0x0196D}, {0x01970, 0x01974}, {0x01980, 0x019AB},
    {0x019B0, 0x019C9}, {0x01A00, 0x01A16}, {0x01A20, 0x01A54}, {0x01AA7, 0x01AA7}, {0x01B05, 0x01B33},
    {0x01B45, 0x01B4C}, {0x01B83, 0x01BA0}, {0x01BAE, 0x01BAF}, {0x01BBA, 0x01BE5}, {0x01C00, 0x01C23},
    {0x01C4D, 0x01C4F}, {0x01C5A, 0x01C7D}, {0x01C80, 0x01C88}, {0x01C90, 0x01CBA}, {0x01CBD, 0x01CBF},
    {0x01CE9, 0x01CEC}, {0x01CEE, 0x01CF3}, {0x01CF5, 0x01CF6}, {0x01CFA, 0x01CFA}, {0x01D00, 0x01DBF},
    {0x01E00, 0x01F15}, {0x01F18, 0x01F1D}, {0x01F20, 0x01F45}, {0x01F48, 0x01F4D}, {0x01F50, 0x01F57},
    {0x01F59, 0x01F59}, {0x01F5B, 0x01F5B}, {0x01F5D, 0x01F5D}, {0x01F5F, 0x01F7D}, {0x01F80, 0x01FB4},
    {0x01FB6, 0x01FBC}, {0x01FBE, 0x01FBE}, {0x01FC2, 0x01FC4}, {0x01FC6, 0x01FCC}, {0x01FD0, 0x01FD3},
    {0x01FD6, 0x01FDB}, {0x01FE0, 0x01FEC}, {0x01FF2, 0x01FF4}, {0x01FF6, 0x01FFC}, {0x02071, 0x02071},
    {0x0207F, 0x0207F}, {0x02090, 0x0209C}, {0x02102, 0x02102}, {0x02107, 0x02107}, {0x0210A, 0x02113},
    {0x02115, 0x02115}, {0x02118, 0x0211D}, {0x02124, 0x02124}, {0x02126, 0x02126}, {0x02128, 0x02128},
    {0x0212A, 0x02139}, {0x0213C, 0x0213F}, {0x02145, 0x02149}, {0x0214E, 0x0214E}, {0x02160, 0x02188},
    {0x02202, 0x02202}, {0x02207, 0x02207}, {0x0221E, 0x0221E}, {0x02C00, 0x02CE4}, {0x02CEB, 0x02CEE},
    {0x02CF2, 0x02CF3}, {0x02D00, 0x02D25}, {0x02D27, 0x02D27}, {0x02D2D, 0x02D2D}, {0x02D30, 0x02D67},
    {0x02D6F, 0x02D6F}, {0x02D80, 0x02D96}, {0x02DA0, 0x02DA6}, {0x02DA8, 0x02DAE}, {0x02DB0, 0x02DB6},
    {0x02DB8, 0x02DBE}, {0x02DC0, 0x02DC6}, {0x02DC8, 0x02DCE}, {0x02DD0, 0x02DD6}, {0x02DD8, 0x02DDE},
    {0x03005, 0x03007}, {0x03021, 0x03029}, {0x03031, 0x03035}, {0x03038, 0x0303C}, {0x03041, 0x03096},
    {0x0309D, 0x0309F}, {0x030A1, 0x030FA}, {0x030FC, 0x030FF}, {0x03105, 0x0312F}, {0x03131, 0x0318E},
    {0x031A0, 0x031BF}, {0x031F0, 0x031FF}, {0x03400, 0x04DBF}, {0x04E00, 0x0A48C}, {0x0A4D0, 0x0A4FD},
    {0x0A500, 0x0A60C}, {0x0A610, 0x0A61F}, {0x0A62A, 0x0A62B}, {0x0A640, 0x0A66E}, {0x0A67F, 0x0A69D},
    {0x0A6A0, 0x0A6EF}, {0x0A717, 0x0A71F}, {0x0A722, 0x0A788}, {0x0A78B, 0x0A7CA}, {0x0A7D0, 0x0A7D1},
    {0x0A7D3, 0x0A7D3}, {0x0A7D5, 0x0A7D9}, {0x0A7F2, 0x0A801}, {0x0A803, 0x0A805}, {0x0A807, 0x0A80A},
    {0x0A80C, 0x0A822}, {0x0A840, 0x0A873}, {0x0A882, 0x0A8B3}, {0x0A8F2, 0x0A8F7}, {0x0A8FB, 0x0A8FB},
    {0x0A8FD, 0x0A8FE}, {0x0A90A, 0x0A925}, {0x0A930, 0x0A946}, {0x0A960, 0x0A97C}, {0x0A984, 0x0A9B2},
    {0x0A9CF, 0x0A9CF}, {0x0A9E0, 0x0A9E4}, {0x0A9E6, 0x0A9EF}, {0x0A9FA, 0x0A9FE}, {0x0AA00, 0x0AA28},
    {0x0AA40, 0x0AA42}, {0x0AA44, 0x0AA4B}, {0x0AA60, 0x0AA76}, {0x0AA7A, 0x0AA7A}, {0x0AA7E, 0x0AAAF},
    {0x0AAB1, 0x0AAB1}, {0x0AAB5, 0x0AAB6}, {0x0AAB9, 0x0AABD}, {0x0AAC0, 0x0AAC0}, {0x0AAC2, 0x0AAC2},
    {0x0AADB, 0x0AADD}, {0x0AAE0, 0x0AAEA}, {0x0AAF2, 0x0AAF4}, {0x0AB01, 0x0AB06}, {0x0AB09, 0x0AB0E},
    {0x0AB11, 0x0AB16}, {0x0AB20, 0x0AB26}, {0x0AB28, 0x0AB2E}, {0x0AB30, 0x0AB5A}, {0x0AB5C, 0x0AB69},
    {0x0AB70, 0x0ABE2}, {0x0AC00, 0x0D7A3}, {0x0D7B0, 0x0D7C6}, {0x0D7CB, 0x0D7FB}, {0x0F900, 0x0FA6D},
    {0x0FA70, 0x0FAD9}, {0x0FB00, 0x0FB06}, {0x0FB13, 0x0FB17}, {0x0FB1D, 0x0FB1D}, {0x0FB1F, 0x0FB28},
    {0x0FB2A, 0x0FB36}, {0x0FB38, 0x0FB3C}, {0x0FB3E, 0x0FB3E}, {0x0FB40, 0x0FB41}, {0x0FB43, 0x0FB44},
    {0x0FB46, 0x0FBB1}, {0x0FBD3, 0x0FC5D}, {0x0FC64, 0x0FD3D}, {0x0FD50, 0x0FD8F}, {0x0FD92, 0x0FDC7},
    {0x0FDF0, 0x0FDF9}, {0x0FE71, 0x0FE71}, {0x0FE73, 0x0FE73}, {0x0FE77, 0x0FE77}, {0x0FE79, 0x0FE79},
    {0x0FE7B, 0x0FE7B}, {0x0FE7D, 0x0FE7D}, {0x0FE7F, 0x0FEFC}, {0x0FF21, 0x0FF3A}, {0x0FF41, 0x0FF5A},
    {0x0FF66, 0x0FF9D}, {0x0FFA0, 0x0FFBE}, {0x0FFC2, 0x0FFC7}, {0x0FFCA, 0x0FFCF}, {0x0FFD2, 0x0FFD7},
    {0x0FFDA, 0x0FFDC}, {0x10000, 0x1000B}, {0x1000D, 0x10026}, {0x10028, 0x1003A}, {0x1003C, 0x1003D},
    {0x1003F, 0x1004D}, {0x10050, 0x1005D}, {0x10080, 0x100FA}, {0x10140, 0x10174}, {0x10280, 0x1029C},
    {0x102A0, 0x102D0}, {0x10300, 0x1031F}, {0x1032D, 0x1034A}, {0x10350, 0x10375}, {0x10380, 0x1039D},
    {0x103A0, 0x103C3}, {0x103C8, 0x103CF}, {0x103D1, 0x103D5}, {0x10400, 0x1049D}, {0x104B0, 0x104D3},
    {0x104D8, 0x104FB}, {0x10500, 0x10527}, {0x10530, 0x10563}, {0x10570, 0x1057A}, {0x1057C, 0x1058A},
    {0x1058C, 0x10592}, {0x10594, 0x10595}, {0x10597, 0x105A1}, {0x105A3, 0x105B1}, {0x105B3, 0x105B9},
    {0x105BB, 0x105BC}, {0x10600, 0x10736}, {0x10740, 0x10755}, {0x10760, 0x10767}, {0x10780, 0x10785},
    {0x10787, 0x107B0}, {0x107B2, 0x107BA}, {0x10800, 0x10805}, {0x10808, 0x10808}, {0x1080A, 0x10835},
    {0x10837, 0x10838}, {0x1083C, 0x1083C}, {0x1083F, 0x10855}, {0x10860, 0x10876}, {0x10880, 0x1089E},
    {0x108E0, 0x108F2}, {0x108F4, 0x108F5}, {0x10900, 0x10915}, {0x10920, 0x10939}, {0x10980, 0x109B7},
    {0x109BE, 0x109BF}, {0x10A00, 0x10A00}, {0x10A10, 0x10A13}, {0x10A15, 0x10A17}, {0x10A19, 0x10A35},
    {0x10A60, 0x10A7C}, {0x10A80, 0x10A9C}, {0x10AC0, 0x10AC7}, {0x10AC9, 0x10AE4}, {0x10B00, 0x10B35},
    {0x10B40, 0x10B55}, {0x10B60, 0x10B72}, {0x10B80, 0x10B91}, {0x10C00, 0x10C48}, {0x10C80, 0x10CB2},
    {0x10CC0, 0x10CF2}, {0x10D00, 0x10D23}, {0x10E80, 0x10EA9}, {0x10EB0, 0x10EB1}, {0x10F00, 0x10F1C},
    {0x10F27, 0x10F27}, {0x10F30, 0x10F45}, {0x10F70, 0x10F81}, {0x10FB0, 0x10FC4}, {0x10FE0, 0x10FF6},
    {0x11003, 0x11037}, {0x11071, 0x11072}, {0x11075, 0x11075}, {0x11083, 0x110AF}, {0x110D0, 0x110E8},
    {0x11103, 0x11126}, {0x11144, 0x11144}, {0x11147, 0x11147}, {0x11150, 0x11172}, {0x11176, 0x11176},
    {0x11183, 0x111B2}, {0x111C1, 0x111C4}, {0x111DA, 0x111DA}, {0x111DC, 0x111DC}, {0x11200, 0x11211},
    {0x11213, 0x1122B}, {0x1123F, 0x11240}, {0x11280, 0x11286}, {0x11288, 0x11288}, {0x1128A, 0x1128D},
    {0x1128F, 0x1129D}, {0x1129F, 0x112A8}, {0x112B0, 0x112DE}, {0x11305, 0x1130C}, {0x1130F, 0x11310},
    {0x11313, 0x11328}, {0x1132A, 0x11330}, {0x11332, 0x11333}, {0x11335, 0x11339}, {0x1133D, 0x1133D},
    {0x11350, 0x11350}, {0x1135D, 0x11361}, {0x11400, 0x11434}, {0x11447, 0x1144A}, {0x1145F, 0x11461},
    {0x11480, 0x114AF}, {0x114C4, 0x114C5}, {0x114C7, 0x114C7}, {0x11580, 0x115AE}, {0x115D8, 0x115DB},
    {0x11600, 0x1162F}, {0x11644, 0x11644}, {0x11680, 0x116AA}, {0x116B8, 0x116B8}, {0x11700, 0x1171A},
    {0x11740, 0x11746}, {0x11800, 0x1182B}, {0x118A0, 0x118DF}, {0x118FF, 0x11906}, {0x11909, 0x11909},
    {0x1190C, 0x11913}, {0x11915, 0x11916}, {0x11918, 0x1192F}, {0x1193F, 0x1193F}, {0x11941, 0x11941},
    {0x119A0, 0x119A7}, {0x119AA, 0x119D0}, {0x119E1, 0x119E1}, {0x119E3, 0x119E3}, {0x11A00, 0x11A00},
    {0x11A0B, 0x11A32}, {0x11A3A, 0x11A3A}, {0x11A50, 0x11A50}, {0x11A5C, 0x11A89}, {0x11A9D, 0x11A9D},
    {0x11AB0, 0x11AF8}, {0x11C00, 0x11C08}, {0x11C0A, 0x11C2E}, {0x11C40, 0x11C40}, {0x11C72, 0x11C8F},
    {0x11D00, 0x11D06}, {0x11D08, 0x11D09}, {0x11D0B, 0x11D30}, {0x11D46, 0x11D46}, {0x11D60, 0x11D65},
    {0x11D67, 0x11D68}, {0x11D6A, 0x11D89}, {0x11D98, 0x11D98}, {0x11EE0, 0x11EF2}, {0x11F02, 0x11F02},
    {0x11F04, 0x11F10}, {0x11F12, 0x11F33}, {0x11FB0, 0x11FB0}, {0x12000, 0x12399}, {0x12400, 0x1246E},
    {0x12480, 0x12543}, {0x12F90, 0x12FF0}, {0x13000, 0x1342F}, {0x13441, 0x13446}, {0x14400, 0x14646},
    {0x16800, 0x16A38}, {0x16A40, 0x16A5E}, {0x16A70, 0x16ABE}, {0x16AD0, 0x16AED}, {0x16B00, 0x16B2F},
    {0x16B40, 0x16B43}, {0x16B63, 0x16B77}, {0x16B7D, 0x16B8F}, {0x16E40, 0x16E7F}, {0x16F00, 0x16F4A},
    {0x16F50, 0x16F50}, {0x16F93, 0x16F9F}, {0x16FE0, 0x16FE1}, {0x16FE3, 0x16FE3}, {0x17000, 0x187F7},
    {0x18800, 0x18CD5}, {0x18D00, 0x18D08}, {0x1AFF0, 0x1AFF3}, {0x1AFF5, 0x1AFFB}, {0x1AFFD, 0x1AFFE},
    {0x1B000, 0x1B122}, {0x1B132, 0x1B132}, {0x1B150, 0x1B152}, {0x1B155, 0x1B155}, {0x1B164, 0x1B167},
    {0x1B170, 0x1B2FB}, {0x1BC00, 0x1BC6A}, {0x1BC70, 0x1BC7C}, {0x1BC80, 0x1BC88}, {0x1BC90, 0x1BC99},
    {0x1D400, 0x1D454}, {0x1D456, 0x1D49C}, {0x1D49E, 0x1D49F}, {0x1D4A2, 0x1D4A2}, {0x1D4A5, 0x1D4A6},
    {0x1D4A9, 0x1D4AC}, {0x1D4AE, 0x1D4B9}, {0x1D4BB, 0x1D4BB}, {0x1D4BD, 0x1D4C3}, {0x1D4C5, 0x1D505},
    {0x1D507, 0x1D50A}, {0x1D50D, 0x1D514}, {0x1D516, 0x1D51C}, {0x1D51E, 0x1D539}, {0x1D53B, 0x1D53E},
    {0x1D540, 0x1D544}, {0x1D546, 0x1D546}, {0x1D54A, 0x1D550}, {0x1D552, 0x1D6A5}, {0x1D6A8, 0x1D7CB},
    {0x1DF00, 0x1DF1E}, {0x1DF25, 0x1DF2A}, {0x1E030, 0x1E06D}, {0x1E100, 0x1E12C}, {0x1E137, 0x1E13D},
    {0x1E14E, 0x1E14E}, {0x1E290, 0x1E2AD}, {0x1E2C0, 0x1E2EB}, {0x1E4D0, 0x1E4EB}, {0x1E7E0, 0x1E7E6},
    {0x1E7E8, 0x1E7EB}, {0x1E7ED, 0x1E7EE}, {0x1E7F0, 0x1E7FE}, {0x1E800, 0x1E8C4}, {0x1E900, 0x1E943},
    {0x1E94B, 0x1E94B}, {0x1EE00, 0x1EE03}, {0x1EE05, 0x1EE1F}, {0x1EE21, 0x1EE22}, {0x1EE24, 0x1EE24},
    {0x1EE27, 0x1EE27}, {0x1EE29, 0x1EE32}, {0x1EE34, 0x1EE37}, {0x1EE39, 0x1EE39}, {0x1EE3B, 0x1EE3B},
    {0x1EE42, 0x1EE42}, {0x1EE47, 0x1EE47}, {0x1EE49, 0x1EE49}, {0x1EE4B, 0x1EE4B}, {0x1EE4D, 0x1EE4F},
    {0x1EE51, 0x1EE52}, {0x1EE54, 0x1EE54}, {0x1EE57, 0x1EE57}, {0x1EE59, 0x1EE59}, {0x1EE5B, 0x1EE5B},
    {0x1EE5D, 0x1EE5D}, {0x1EE5F, 0x1EE5F}, {0x1EE61, 0x1EE62}, {0x1EE64, 0x1EE64}, {0x1EE67, 0x1EE6A},
    {0x1EE6C, 0x1EE72}, {0x1EE74, 0x1EE77}, {0x1EE79, 0x1EE7C}, {0x1EE7E, 0x1EE7E}, {0x1EE80, 0x1EE89},
    {0x1EE8B, 0x1EE9B}, {0x1EEA1, 0x1EEA3}, {0x1EEA5, 0x1EEA9}, {0x1EEAB, 0x1EEBB}, {0x20000, 0x2A6DF},
    {0x2A700, 0x2B739}, {0x2B740, 0x2B81D}, {0x2B820, 0x2CEA1}, {0x2CEB0, 0x2EBE0}, {0x2EBF0, 0x2EE5D},
    {0x2F800, 0x2FA1D}, {0x30000, 0x3134A}, {0x31350, 0x323AF},
};

static const CodeRange UNICODE_SPACE[9] = {
    {0x00085, 0x00085}, {0x000A0, 0x000A0}, {0x01680, 0x01680}, {0x0180E, 0x0180E}, {0x02000, 0x0200A},
    {0x02028, 0x02029}, {0x0202F, 0x0202F}, {0x0205F, 0x0205F}, {0x03000, 0x03000},
};
static bool in_ranges(const CodeRange *r, size_t n, uint32_t c) {
    size_t lo = 0;
    size_t hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < r[mid].lo)
            hi = mid;
        else if (c > r[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}

static bool id_start(uint32_t c) { return in_ranges(ID_START, sizeof ID_START / sizeof *ID_START, c); }

static bool unicode_space(uint32_t c) {
    return in_ranges(UNICODE_SPACE, sizeof UNICODE_SPACE / sizeof *UNICODE_SPACE, c);
}

/* Decodes one UTF-8 character at p. Returns its length, or 0 if it is malformed, overlong, a surrogate, above
 * U+10FFFF or cut off by the end. */
static size_t utf8_decode(const u8 *p, const u8 *end, uint32_t *cp) {
    unsigned c = p[0];
    if (c < 0xC2 || c > 0xF4)
        return 0;
    unsigned n = 2;
    if (c >= 0xF0)
        n = 4;
    else if (c >= 0xE0)
        n = 3;

    /* The second byte's range rules out overlong forms, surrogates and code points above U+10FFFF */
    unsigned lo = 0x80;
    unsigned hi = 0xBF;
    if (c == 0xE0)
        lo = 0xA0;
    else if (c == 0xF0)
        lo = 0x90;
    else if (c == 0xED)
        hi = 0x9F;
    else if (c == 0xF4)
        hi = 0x8F;
    if (end - p < n || p[1] < lo || p[1] > hi)
        return 0;

    uint32_t v = c & (0x7F >> n);
    for (unsigned i = 1; i < n; i++) {
        if ((p[i] & 0xC0) != 0x80)
            return 0;
        v = v << 6 | (p[i] & 0x3F);
    }
    *cp = v;
    return n;
}

/* ---- characters and line splices ---- */
static bool is_hspace(int c) { return c == ' ' || c == '\t' || c == '\f' || c == '\v'; }

static bool is_newline(int c) { return c == '\n' || c == '\r'; }

static bool is_digit(int c) { return c >= '0' && c <= '9'; }

static bool is_ident_ascii(int c) { return is_digit(c) || ((c | 32) >= 'a' && (c | 32) <= 'z') || c == '_'; }

static bool is_raw_delim(int c) { return c > ' ' && c < 0x7F && c != '(' && c != ')' && c != '\\'; }

static int hex_value(int c) {
    if (is_digit(c))
        return c - '0';
    if ((c | 32) >= 'a' && (c | 32) <= 'f')
        return (c | 32) - 'a' + 10;
    return -1;
}

/* The length of the line splice at p (a backslash, optional spaces or tabs and a newline, which vanish), or 0. */
static size_t splice_at(const u8 *p, const u8 *end) {
    if (p >= end || *p != '\\')
        return 0;

    const u8 *q = p + 1;
    while (q < end && is_hspace(*q))
        q++;
    if (q >= end || !is_newline(*q))
        return 0;
    q += q + 1 < end && is_newline(q[1]) && q[1] != *q ? 2 : 1; /* \r\n and \n\r are one newline */
    return (size_t)(q - p);
}

/* The character at p with splices skipped, and in *n the bytes up to and including it. At the end of the input: -1,
 * with *n the bytes of splices skipped. */
static int peek(const u8 *p, const u8 *end, size_t *n) {
    const u8 *q = p;
    size_t s;
    while ((s = splice_at(q, end)))
        q += s;
    *n = (size_t)(q - p) + (q < end);
    return q < end ? *q : -1;
}

/* ---- SIMD ----
 * vec_eq(block, c): the 64-bit mask of bytes equal to c.
 * block_masks(): what a block of code needs, as masks, and the default class of each byte. It reads 65 bytes, since
 * whether the last byte opens a comment or is a %: depends on the next.
 * store_classes(): the default classes, XORed with bit planes. */
typedef struct {
    u64 ident, digit, dot, space, newline;
    u64 special; /* what blocks leave to lex_token: " ' # \ $ NUL, bytes from 0x80, and the starts of // /" "* %: */
} Masks;

#if defined(__AVX2__)
#include <immintrin.h>
typedef struct {
    __m256i a, b;
} Block;

static inline Block vec_load(const u8 *p) {
    Block v = {_mm256_loadu_si256((const __m256i *)p), _mm256_loadu_si256((const __m256i *)(p + 32))};
    return v;
}

static inline u64 mask32(__m256i x) { return (uint32_t)_mm256_movemask_epi8(x); }

static inline __m256i eq(__m256i v, int c) { return _mm256_cmpeq_epi8(v, _mm256_set1_epi8((char)c)); }

static inline u64 vec_eq(Block v, u8 c) { return mask32(eq(v.a, c)) | mask32(eq(v.b, c)) << 32; }

/* The bytes of v found in table, which holds at most one byte per low nibble (-128 where it holds none) */
static inline __m256i lookup_eq(__m256i v, __m256i table) {
    return _mm256_cmpeq_epi8(_mm256_shuffle_epi8(table, v), v);
}

static inline void half_masks(__m256i v, __m256i next, u64 *m, __m256i *def) {
    const __m256i spaces =
        _mm256_setr_epi8(' ', -128, -128, -128, -128, -128, -128, -128, -128, '\t', '\n', '\v', '\f', '\r', -128, -128,
                         ' ', -128, -128, -128, -128, -128, -128, -128, -128, '\t', '\n', '\v', '\f', '\r', -128, -128);
    const __m256i specials =
        _mm256_setr_epi8(0, -128, '"', '#', '$', -128, -128, '\'', -128, -128, -128, -128, '\\', -128, -128, -128, 0,
                         -128, '"', '#', '$', -128, -128, '\'', -128, -128, -128, -128, '\\', -128, -128, -128);
    /* Word characters by nibbles: bit 1 is a digit (row 3), 2 the letters in rows 4 and 6, 4 those in rows 5 and 7,
     * and 8 the _ */
    const __m256i by_low = _mm256_setr_epi8(5, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 2, 2, 2, 2, 10, 5, 7, 7, 7, 7, 7, 7, 7, 7,
                                            7, 6, 2, 2, 2, 2, 10);
    const __m256i by_high = _mm256_setr_epi8(0, 0, 0, 1, 2, 12, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 12, 2, 4,
                                             0, 0, 0, 0, 0, 0, 0, 0);
    __m256i high = _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
    __m256i kind = _mm256_and_si256(_mm256_shuffle_epi8(by_low, v), _mm256_shuffle_epi8(by_high, high));
    __m256i not_word = _mm256_cmpeq_epi8(kind, _mm256_setzero_si256());
    __m256i space = lookup_eq(v, spaces);

    __m256i special = _mm256_or_si256(lookup_eq(v, specials), v); /* v: its sign bit marks bytes from 0x80 */
    __m256i opens = _mm256_and_si256(eq(v, '/'), _mm256_or_si256(eq(next, '/'), eq(next, '*')));
    special = _mm256_or_si256(special, _mm256_or_si256(opens, _mm256_and_si256(eq(v, '%'), eq(next, ':'))));

    m[0] = ~mask32(not_word) & 0xFFFFFFFF;
    m[1] = mask32(_mm256_cmpeq_epi8(kind, _mm256_set1_epi8(1)));
    m[2] = mask32(eq(v, '.'));
    m[3] = mask32(space);
    m[4] = mask32(_mm256_or_si256(eq(v, '\n'), eq(v, '\r')));
    m[5] = mask32(special);
    *def =
        _mm256_andnot_si256(space, _mm256_blendv_epi8(_mm256_set1_epi8(C_IDENT), _mm256_set1_epi8(C_PUNCT), not_word));
}

static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    Block v = vec_load(p);
    Block next = vec_load(p + 1);
    u64 x[6];
    u64 y[6];
    half_masks(v.a, next.a, x, &def->a);
    half_masks(v.b, next.b, y, &def->b);
    m->ident = x[0] | y[0] << 32;
    m->digit = x[1] | y[1] << 32;
    m->dot = x[2] | y[2] << 32;
    m->space = x[3] | y[3] << 32;
    m->newline = x[4] | y[4] << 32;
    m->special = x[5] | y[5] << 32;
}

/* 0xFF in each byte whose bit in m is set */
static inline __m256i bit_bytes(uint32_t m) {
    const __m256i shuffle = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
                                             3, 3, 3, 3, 3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    __m256i x = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)m), shuffle), bits);
    return _mm256_cmpeq_epi8(x, bits);
}

static inline __m256i half_classes(__m256i def, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, uint32_t clear) {
    __m256i x = _mm256_or_si256(_mm256_and_si256(bit_bytes(d0), _mm256_set1_epi8(1)),
                                _mm256_and_si256(bit_bytes(d1), _mm256_set1_epi8(2)));
    x = _mm256_or_si256(x, _mm256_and_si256(bit_bytes(d2), _mm256_set1_epi8(4)));
    if (clear) {
        x = _mm256_or_si256(x, _mm256_and_si256(bit_bytes(d3), _mm256_set1_epi8(8)));
        def = _mm256_andnot_si256(bit_bytes(clear), def);
    }
    return _mm256_xor_si256(def, x);
}

/* The default classes, with the bytes in clear set to 0 first, XORed with four bit planes */
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2, u64 d3, u64 clear) {
    _mm256_storeu_si256((__m256i *)o,
                        half_classes(def.a, (uint32_t)d0, (uint32_t)d1, (uint32_t)d2, (uint32_t)d3, (uint32_t)clear));
    _mm256_storeu_si256((__m256i *)(o + 32),
                        half_classes(def.b, (uint32_t)(d0 >> 32), (uint32_t)(d1 >> 32), (uint32_t)(d2 >> 32),
                                     (uint32_t)(d3 >> 32), (uint32_t)(clear >> 32)));
}

#else
typedef const u8 *Block;

static inline Block vec_load(const u8 *p) { return p; }

static inline u64 vec_eq(Block v, u8 c) {
    u64 m = 0;
    for (int i = 0; i < 64; i++)
        m |= (u64)(v[i] == c) << i;
    return m;
}

static inline void block_masks(const u8 *p, Masks *m, Block *def) {
    memset(m, 0, sizeof *m);
    for (int i = 0; i < 64; i++) {
        int c = p[i];
        int next = p[i + 1];
        u64 bit = 1ULL << i;
        if (is_ident_ascii(c))
            m->ident |= bit;
        if (is_digit(c))
            m->digit |= bit;
        if (c == '.')
            m->dot |= bit;
        if (is_hspace(c) || is_newline(c))
            m->space |= bit;
        if (is_newline(c))
            m->newline |= bit;
        if (c == '"' || c == '\'' || c == '#' || c == '\\' || c == 0 || c == '$' || c >= 0x80)
            m->special |= bit;
        if ((c == '/' && (next == '/' || next == '*')) || (c == '%' && next == ':'))
            m->special |= bit;
    }
    *def = p;
}

/* The default classes, with the bytes in clear set to 0 first, XORed with four bit planes */
static inline void store_classes(u8 *o, Block def, u64 d0, u64 d1, u64 d2, u64 d3, u64 clear) {
    for (int i = 0; i < 64; i++) {
        int c = def[i];
        int base = C_PUNCT;
        if (clear >> i & 1)
            base = 0;
        else if (is_ident_ascii(c))
            base = C_IDENT;
        else if (is_hspace(c) || is_newline(c))
            base = C_SPACE;
        o[i] = (u8)(base ^ ((d0 >> i & 1) | (d1 >> i & 1) << 1 | (d2 >> i & 1) << 2 | (d3 >> i & 1) << 3));
    }
}
#endif

/* The first byte in [p, end) equal to any of a to f, or end. */
static const u8 *find6(const u8 *p, const u8 *end, u8 a, u8 b, u8 c, u8 d, u8 e, u8 f) {
    while (end - p >= 64) {
        Block v = vec_load(p);
        u64 m = vec_eq(v, a) | vec_eq(v, b) | vec_eq(v, c) | vec_eq(v, d) | vec_eq(v, e) | vec_eq(v, f);
        if (m)
            return p + __builtin_ctzll(m);
        p += 64;
    }
    while (p < end && *p != a && *p != b && *p != c && *p != d && *p != e && *p != f)
        p++;
    return p;
}

/* ---- the parts of a token ---- */

/* Reads the UCN whose backslash ends just before p, so p is at u, U or N. Returns its code point, or 0 if it is
 * invalid. *after is where a complete UCN ends, even one with an invalid code point (clang still consumes those), or
 * NULL if it is incomplete. */
static uint32_t read_ucn(const u8 *p, const u8 *end, const u8 **after) {
    size_t n;
    int kind = peek(p, end, &n);
    *after = NULL;
    p += n;

    if (kind == 'N') { /* \N{NAME}: clang looks the name up, this lexer cannot */
        if (peek(p, end, &n) != '{')
            return 0;
        p += n;
        for (size_t len = 0;; len++) {
            int c = peek(p, end, &n);
            p += n;
            if (c == '}') {
                if (len)
                    *after = p;
                return 0;
            }
            if (c <= 0 || is_newline(c))
                return 0;
        }
    }
    if (kind != 'u' && kind != 'U')
        return 0;

    unsigned digits = kind == 'u' ? 4 : 8;
    unsigned count = 0;
    bool delimited = false;
    uint32_t cp = 0;
    while (count != digits || delimited) {
        int c = peek(p, end, &n);
        if (!delimited && count == 0 && c == '{') {
            delimited = true;
            p += n;
            continue;
        }
        if (delimited && c == '}') {
            p += n;
            break;
        }
        int v = hex_value(c);
        if (v < 0 && delimited)
            return 0;
        if (v < 0)
            break;
        if (cp & 0xF0000000u)
            return 0;
        cp = cp << 4 | (uint32_t)v;
        p += n;
        count++;
    }
    if (count == 0 || (delimited && kind == 'U') || (!delimited && count != digits))
        return 0;

    *after = p;
    if (cp < 0xA0 || (cp >= 0xD800 && cp <= 0xDFFF))
        return 0;
    return cp;
}

/* Inside an identifier or number: if the character at p is a UCN or a UTF-8 character, returns its end, unless it is
 * Unicode whitespace or invalid. Otherwise NULL. */
static const u8 *continue_word(const u8 *p, const u8 *end) {
    size_t n;
    int c = peek(p, end, &n);
    uint32_t cp;
    if (c == '\\') {
        const u8 *after;
        cp = read_ucn(p + n, end, &after);
        return cp && !unicode_space(cp) ? after : NULL;
    }
    if (c < 0x80)
        return NULL;
    size_t len = utf8_decode(p + n - 1, end, &cp);
    return len && !unicode_space(cp) ? p + n - 1 + len : NULL;
}

static const u8 *ident_end(const u8 *p, const u8 *end) {
    for (;;) {
        size_t n;
        int c = peek(p, end, &n);
        const u8 *q;
        if (is_ident_ascii(c) || c == '$')
            p += n;
        else if ((q = continue_word(p, end)))
            p = q;
        else
            return p;
    }
}

/* The rest of a number: digits, letters, _ and ., a sign right after e, E, p or P, a ' followed by a letter, digit or
 * _ (C23 digit separators), UCNs and UTF-8. */
static const u8 *number_end(const u8 *p, const u8 *end) {
    for (;;) {
        size_t n;
        size_t m;
        int c = peek(p, end, &n);
        int prev = 0;
        const u8 *q;
        while (is_ident_ascii(c) || c == '.') {
            p += n;
            prev = c;
            c = peek(p, end, &n);
        }
        if ((c == '+' || c == '-') && ((prev | 32) == 'e' || (prev | 32) == 'p'))
            p += n;
        else if (c == '\'' && is_ident_ascii(peek(p + n, end, &m)))
            p += n + m;
        else if ((q = continue_word(p, end)))
            p = q;
        else
            return p;
    }
}

/* A string or character literal whose opening quote ends just before p. Returns its end: past the closing quote, or
 * at the newline or end of input that cuts it off. */
static const u8 *literal_end(const u8 *p, const u8 *end, int quote) {
    size_t n;
    if (quote == '\'' && peek(p, end, &n) == '\'')
        return p + n; /* '' */

    for (;;) {
        p = find6(p, end, (u8)quote, '\\', '\n', '\r', '\n', '\n');
        if (p >= end)
            return end;
        if (*p == quote)
            return p + 1;
        if (is_newline(*p))
            return p;

        size_t s = splice_at(p, end);
        if (s) {
            p += s;
            continue;
        }
        int c = peek(p + 1, end, &n); /* the escaped character */
        if (c == -1)
            return end;
        if (is_newline(c))
            return p + n;
        p += 1 + n;
    }
}

/* A raw string whose opening quote ends just before p: delim( ... )delim" with no splices. With a bad delimiter clang
 * skips to the next quote. */
static const u8 *raw_string_end(const u8 *p, const u8 *end) {
    size_t len = 0;
    while (len < 16 && p + len < end && is_raw_delim(p[len]))
        len++;
    if (p + len >= end || p[len] != '(') {
        const u8 *q = memchr(p, '"', (size_t)(end - p));
        return q ? q + 1 : end;
    }

    for (const u8 *q = p + len + 1; (q = memchr(q, ')', (size_t)(end - q))); q++)
        if ((size_t)(end - q) > len + 1 && !memcmp(q + 1, p, len) && q[len + 1] == '"')
            return q + len + 2;
    return end;
}

/* The end of a line comment whose body starts at p: the newline that ends it, not consumed. */
static const u8 *line_comment_end(const u8 *p, const u8 *end) {
    for (;;) {
        p = find6(p, end, '\n', '\r', '\n', '\n', '\n', '\n');
        if (p >= end)
            return end;

        const u8 *e = p - 1;
        while (is_hspace(*e)) /* stops at the comment's own "//" at the latest */
            e--;
        if (*e != '\\')
            return p;
        size_t n;
        int c = peek(e, end, &n); /* the splice, and the character after it */
        if (c == -1)
            return end;
        if (is_newline(c))
            return e + n - 1;
        p = e + n;
    }
}

/* Is the newline at q the end of "*", backslash, optional spaces and newline (maybe repeated), so that a "/" right
 * after it closes a block comment? Never looks before lo. */
static bool star_splice_before(const u8 *lo, const u8 *q) {
    for (;;) {
        if (q > lo && is_newline(q[-1])) {
            if (q[-1] == *q)
                return false; /* two newlines, not one \r\n */
            q--;
        }
        q--;
        while (q >= lo && (is_hspace(*q) || *q == 0))
            q--;
        if (q < lo || *q != '\\')
            return false;
        if (--q < lo)
            return false;
        if (*q == '*')
            return true;
        if (!is_newline(*q))
            return false;
    }
}

/* The end of a block comment opened at start, whose body starts at p. The first character of the body can't close
 * it, so "/" "*" "/" is still open. Only a "/" after "*" or after a newline can close it. */
static const u8 *block_comment_end(const u8 *start, const u8 *p, const u8 *end) {
    size_t n;
    if (peek(p, end, &n) == -1)
        return end;

    const u8 *i = p + n - 1; /* pairs (i, i + 1) where i + 1 might be the closing slash */
    while (end - i >= 65) {
        Block v = vec_load(i);
        Block w = vec_load(i + 1);
        u64 m = vec_eq(w, '/') & (vec_eq(v, '*') | vec_eq(v, '\n') | vec_eq(v, '\r'));
        for (; m; m &= m - 1) {
            const u8 *s = i + __builtin_ctzll(m) + 1;
            if (s[-1] == '*' || star_splice_before(start, s - 1))
                return s + 1;
        }
        i += 64;
    }
    for (const u8 *s = i + 1; s < end; s++)
        if (*s == '/' && (s[-1] == '*' || (is_newline(s[-1]) && star_splice_before(start, s - 1))))
            return s + 1;
    return end;
}

/* Is the next token after p, past spaces and comments, a "("? */
static bool paren_next(const u8 *p, const u8 *end) {
    for (;;) {
        size_t n;
        size_t m;
        int c = peek(p, end, &n);
        if (is_hspace(c) || is_newline(c) || c == 0) {
            p += n;
            continue;
        }
        if (c == '/') {
            int d = peek(p + n, end, &m);
            if (d == '/') {
                p = line_comment_end(p + n + m, end);
                continue;
            }
            if (d == '*') {
                p = block_comment_end(p, p + n + m, end);
                continue;
            }
        }
        return c == '(';
    }
}

/* If c, whose bytes end just before q, starts a literal (u, u8, U or L before " or ', or any of those or nothing
 * before R"), returns the position just past the opening quote and sets *kind to '"', '\'' or 'R' for a raw string.
 * Otherwise NULL. */
static const u8 *literal_start(int c, const u8 *q, const u8 *end, int *kind) {
    size_t n;
    int d = peek(q, end, &n);
    if (c == 'u' && d == '8') {
        q += n;
        d = peek(q, end, &n);
    }
    bool raw = c == 'R';
    if (!raw && d == 'R') {
        q += n;
        d = peek(q, end, &n);
        raw = true;
    }
    if (d == '"' || (d == '\'' && !raw)) {
        *kind = raw ? 'R' : d;
        return q + n;
    }
    return NULL;
}

/* ---- one token at a time ---- */
typedef struct {
    const u8 *src, *end;
    u8 *cls;
    bool line_start; /* nothing but spaces and comments so far on this line */
    bool directive;  /* inside a # directive */
} Lexer;

/* Classes [a, b). Inside a directive everything but comments and literals is C_PREPROC. */
static void paint(Lexer *L, const u8 *a, const u8 *b, int c) {
    if (L->directive && c != C_COMMENT && c != C_STRING && c != C_CHAR)
        c = C_PREPROC;
    memset(L->cls + (a - L->src), c, (size_t)(b - a));
}

/* Spaces, newlines, NULs and the splices among them. A newline ends a directive. */
static const u8 *lex_space(Lexer *L, const u8 *p) {
    const u8 *start = p;
    for (;;) {
        size_t n;
        int c = peek(p, L->end, &n);
        if (c == -1) {
            p += n;
            break;
        }
        if (!is_hspace(c) && !is_newline(c) && c != 0)
            break;
        if (is_newline(c)) {
            if (L->directive) {
                paint(L, start, p + n - 1, C_PREPROC);
                start = p + n - 1;
                L->directive = false;
            }
            L->line_start = true;
        }
        p += n;
    }
    paint(L, start, p, C_SPACE);
    return p;
}

/* The class of the identifier [p, q): a keyword class, C_FUNCTION or C_IDENT. */
static int ident_class(const Lexer *L, const u8 *p, const u8 *q) {
    u8 w[KEYWORD_MAX + 4] = {0};
    size_t len = 0;
    size_t n;
    for (; p < q && len <= KEYWORD_MAX; p += n)
        w[len++] = (u8)peek(p, L->end, &n);

    int c = p < q ? C_IDENT : keyword_class(w, len);
    if (c == C_IDENT && !L->directive && paren_next(q, L->end))
        return C_FUNCTION;
    return c;
}

/* Lexes the token at p and returns its end. */
static const u8 *lex_token(Lexer *L, const u8 *p) {
    const u8 *end = L->end;
    const u8 *after;
    size_t n;
    size_t n2;
    size_t n3;
    size_t n4;
    int c = peek(p, end, &n);
    int cls = C_PUNCT;
    int kind;
    bool hash = false;
    uint32_t cp;
    if (c == -1 || is_hspace(c) || is_newline(c) || c == 0)
        return lex_space(L, p);

    const u8 *q = p + n;
    int c2 = peek(q, end, &n2);
    if (is_digit(c)) {
        q = number_end(q, end);
        cls = C_CONSTANT;
    } else if (c == '.' && is_digit(c2)) {
        q = number_end(q + n2, end);
        cls = C_CONSTANT;
    } else if (c == '.' && c2 == '.' && peek(q + n2, end, &n3) == '.') {
        q += n2 + n3; /* ... */
    } else if (c == '"' || c == '\'') {
        q = literal_end(q, end, c);
        cls = c == '"' ? C_STRING : C_CHAR;
    } else if (c == '/' && c2 == '/') {
        q = line_comment_end(q + n2, end);
        cls = C_COMMENT;
    } else if (c == '/' && c2 == '*') {
        q = block_comment_end(p, q + n2, end);
        cls = C_COMMENT;
    } else if (c == '#' && c2 == '#') {
        q += n2;
    } else if (c == '#') {
        hash = true;
    } else if (c == '%' && c2 == ':') {
        q += n2;
        if (peek(q, end, &n3) == '%' && peek(q + n3, end, &n4) == ':')
            q += n3 + n4; /* %:%: */
        else
            hash = true;
    } else if ((c == 'u' || c == 'U' || c == 'L' || c == 'R') && (after = literal_start(c, q, end, &kind))) {
        q = kind == 'R' ? raw_string_end(after, end) : literal_end(after, end, kind);
        cls = kind == '\'' ? C_CHAR : C_STRING;
    } else if (is_ident_ascii(c) || c == '$') {
        q = ident_end(q, end);
        cls = ident_class(L, p, q);
    } else if (c == '\\') { /* a UCN: an identifier if it may start one, else an unknown token */
        cp = read_ucn(q, end, &after);
        if (cp && id_start(cp)) {
            q = ident_end(after, end);
            cls = ident_class(L, p, q);
        } else if (after) {
            q = after;
        }
    } else if (c >= 0x80) { /* UTF-8: the same, and a malformed byte is a token by itself */
        size_t len = utf8_decode(q - 1, end, &cp);
        if (len)
            q += len - 1;
        if (len && id_start(cp)) {
            q = ident_end(q, end);
            cls = ident_class(L, p, q);
        }
    }

    if (hash && L->line_start)
        L->directive = true;
    paint(L, p, q, cls);
    if (cls != C_COMMENT)
        L->line_start = false;
    return q;
}

/* ---- directives ----
 * Everything outside comments and literals is C_PREPROC, so only quotes, slashes, backslashes and newlines need a
 * look. A quote or slash right after a word character or dot, or after a backslash that wasn't a splice, might not
 * start a literal or comment (u8"x", 1'000, \N{a'b}): then lex_token takes over from the last token boundary. */

/* Could the character before q, past splices, be part of a word or number? Never looks before lo. */
static bool word_before(const u8 *lo, const u8 *q) {
    while (q > lo && is_newline(q[-1])) {
        const u8 *e = q - 1;
        if (e > lo && is_newline(e[-1]) && e[-1] != *e)
            e--;
        do
            e--;
        while (e >= lo && is_hspace(*e));
        if (e < lo || *e != '\\')
            return false;
        q = e;
    }
    if (q <= lo)
        return false;
    int c = q[-1];
    return is_ident_ascii(c) || c == '$' || c == '.' || c == '+' || c == '-' || c >= 0x80; /* 1e+'5 is one number */
}

/* The rest of a directive from p, which is at a token boundary, up to the newline that ends it. */
static const u8 *directive(Lexer *L, const u8 *p) {
    const u8 *end = L->end;
    const u8 *boundary = p;
    bool unsure = false;
    for (;;) {
        const u8 *q = find6(p, end, '"', '\'', '/', '\\', '\n', '\r');
        paint(L, p, q, C_PREPROC);
        if (q >= end)
            return end;
        size_t n;
        size_t m;
        int c = peek(q, end, &n); /* past splices, which belong to the token after them */
        if (c == -1 || is_newline(c))
            return lex_space(L, q);
        if (*q == '\\' && !splice_at(q, end)) {
            unsure = true;
            paint(L, q, q + 1, C_PREPROC);
            p = q + 1;
            continue;
        }
        if (c != '"' && c != '\'' && c != '/') {
            p = q + n - 1;
            paint(L, q, p, C_PREPROC);
            continue;
        }
        int c2 = peek(q + n, end, &m);
        if (c == '/' && c2 != '/' && c2 != '*') {
            p = q + n;
            paint(L, q, p, C_PREPROC);
            continue;
        }

        if (unsure || word_before(boundary, q)) {
            p = boundary;
            while (p <= q)
                p = lex_token(L, p);
        } else {
            p = lex_token(L, q);
        }
        boundary = p;
        unsure = false;
    }
}

/* ---- blocks of plain code ---- */

/* The bytes right after a backslash run of odd length: the escaped ones, by simdjson's trick. A run that reaches byte
 * 63 escapes a byte of the next block, where no literal of this block reaches. */
static inline u64 escaped_by(u64 backslash) {
    const u64 odd = 0xAAAAAAAAAAAAAAAAULL;
    u64 starts = backslash & ~(backslash << 1);
    u64 from_even = (backslash + (starts & ~odd)) & ~backslash;
    u64 from_odd = (backslash + (starts & odd)) & ~backslash;
    return (from_even & odd) | (from_odd & ~odd);
}

/* Classes the literals and comments that start at the special bytes in todo and end in this block, into the three
 * masks, and returns the specials left. It stops at the first literal or comment that doesn't end here, a literal
 * after a word character (a prefix, or a digit separator), one that might hold a splice, and any other special byte.
 * It is kept out of line: inlined, the compiler computes its masks for every block. */
__attribute__((noinline)) static u64 block_literals(const u8 *b, const Masks *m, u64 todo, u64 *strings, u64 *chars,
                                                    u64 *comments) {
    Block v = vec_load(b);
    Block w = vec_load(b + 1);
    u64 backslash = vec_eq(v, '\\');
    u64 escaped = escaped_by(backslash);
    u64 dquotes = vec_eq(v, '"') & ~escaped;
    u64 squotes = vec_eq(v, '\'') & ~escaped;
    u64 closes = vec_eq(v, '*') & vec_eq(w, '/'); /* each star with a slash after it */
    u64 splices = backslash & m->space >> 1;      /* each backslash before a space or newline */
    while (todo) {
        unsigned q = (unsigned)__builtin_ctzll(todo);
        u64 word = (m->ident | m->dot) & ~(*strings | *chars | *comments);
        u64 *into = comments;
        u64 risky = splices;
        u64 ends;
        if (b[q] == '/' && b[q + 1] == '/') {
            ends = m->newline & ~1ULL << q;
        } else if (b[q] == '/' && b[q + 1] == '*' && q < 62) {
            ends = (closes & ~1ULL << (q + 1)) << 1; /* the slash after a star past the opening one */
            risky = backslash;
        } else if ((b[q] == '"' || b[q] == '\'') && !(q && (word >> (q - 1) & 1))) {
            into = b[q] == '"' ? strings : chars;
            ends = ((b[q] == '"' ? dquotes : squotes) | m->newline) & ~1ULL << q;
        } else {
            break;
        }
        if (!ends)
            break;

        unsigned last = (unsigned)__builtin_ctzll(ends);
        if (is_newline(b[last])) /* the newline that cuts it off is not part of it */
            last--;
        u64 span = (2ULL << last) - (1ULL << q);
        if (risky & span)
            break;
        *into |= span;
        todo &= ~((2ULL << last) - 1);
    }
    return todo;
}

/* One block of code at b: the input, or a padded copy of its last bytes, where real is the same place in the input.
 * Classifies everything before the first byte it can't handle alone and returns how many bytes that is. *special is
 * set when that byte needs lex_token, rather than being the start of a word that might run on. */
static inline unsigned code_block(Lexer *L, const u8 *b, const u8 *real, u8 *o, bool *special) {
    Masks m;
    Block def;
    block_masks(b, &m, &def);

    /* Literals and comments that end in this block get their class here */
    u64 strings = 0;
    u64 chars = 0;
    u64 comments = 0;
    u64 todo = m.special;
    if (todo) {
        int c = b[__builtin_ctzll(todo)];
        if (c == '"' || c == '\'' || c == '/')
            todo = block_literals(b, &m, todo, &strings, &chars, &comments);
    }
    u64 regions = strings | chars | comments;

    /* Numbers: a run of word characters and dots that starts with a digit or a dot and a digit. Adding each start
     * bit carries through its run and clears it. */
    u64 run = (m.ident | m.dot) & ~regions;
    u64 starts = run & ~(run << 1) & (m.digit | (m.dot & m.digit >> 1));
    u64 number = run & ~(run + starts);

    /* Stops: what the literal loop left, a dot and digit inside a word or after dots (x.5, ..5, ...5), and a number
     * ending in an exponent sign (1e+5) */
    u64 stops = todo | (m.dot & m.digit >> 1 & ~number & ~regions);
    for (u64 e = number & ~(number >> 1) & ~m.digit & ~(1ULL << 63); e; e &= e - 1) {
        unsigned i = (unsigned)__builtin_ctzll(e);
        if (((b[i] | 32) == 'e' || (b[i] | 32) == 'p') && (b[i + 1] == '+' || b[i + 1] == '-'))
            stops |= 1ULL << i;
    }

    /* The last word or number waits for the next block when only spaces and comments follow it here: it might go on,
     * or be a function */
    u64 closed = ~(run | m.space | comments);
    u64 suffix = ~0ULL; /* the bytes after the last one that isn't a word character, dot, space or comment */
    if (closed >> 63)
        suffix = 0;
    else if (closed)
        suffix <<= 64 - __builtin_clzll(closed);
    unsigned tail = 64;
    if (run & suffix) {
        u64 below = ~run & ((1ULL << (63 - __builtin_clzll(run & suffix))) - 1);
        tail = below ? 64 - (unsigned)__builtin_clzll(below) : 0;
    }

    unsigned stop = stops ? (unsigned)__builtin_ctzll(stops) : 64;
    unsigned limit = stop < tail ? stop : tail;
    *special = stop < tail;
    if (limit && limit < 64 && (run >> (limit - 1) & 1)) { /* the word or number before the stop waits too */
        u64 before = ~run & ((1ULL << limit) - 1);
        limit = before ? 64 - (unsigned)__builtin_clzll(before) : 0;
    } else if (limit && b[limit] == '\\' && !((m.space | regions) >> (limit - 1) & 1)) {
        limit--; /* a splice can join the byte before it to the one after: "/\<newline>*" */
    }
    if (!limit)
        return 0;

    u64 region = limit == 64 ? ~0ULL : (1ULL << limit) - 1;
    number &= region;
    regions &= region;
    comments &= region;
    u64 word = m.ident & ~number & ~regions & region;
    u64 word_starts = word & ~(word << 1);
    u64 word_ends = word & ~(word >> 1);
    u64 skip = m.space | comments;
    u64 next = (skip + (word_ends << 1)) & ~skip; /* after each word, the first byte that isn't a space or comment */
    /* A special byte, or the byte before one (a slash before a splice), where a comment, splice or NUL may hide */
    u64 unsure = (m.special & ~regions) | m.special >> 1 | (u64)(b[64] == '\\') << 63;
    u64 d0 = (number & m.ident) | (chars & region); /* the default class is C_IDENT, and C_CONSTANT is C_IDENT ^ 1 */
    u64 d1 = regions & ~comments;                   /* C_STRING and C_CHAR, on a cleared default */
    u64 d2 = regions & ~comments;
    while (word_starts) {
        unsigned s = (unsigned)__builtin_ctzll(word_starts);
        unsigned e = (unsigned)__builtin_ctzll(word_ends);
        unsigned a = next ? (unsigned)__builtin_ctzll(next) : 64;
        word_starts &= word_starts - 1;
        word_ends &= word_ends - 1;
        next &= next - 1;

        int c = keyword_class(b + s, e - s + 1);
        if (c == C_IDENT && b[a] == '(')
            c = C_FUNCTION;
        else if (c == C_IDENT && (unsure >> a & 1) && real + a < L->end && paren_next(real + a, L->end))
            c = C_FUNCTION; /* a comment this block didn't take, a splice or a NUL might come first */
        if (c == C_IDENT)
            continue;

        unsigned x = (unsigned)c ^ C_IDENT;
        u64 bits = (2ULL << e) - (1ULL << s);
        d0 |= bits & -(u64)(x & 1);
        d1 |= bits & -(u64)(x >> 1 & 1);
        d2 |= bits & -(u64)(x >> 2 & 1);
    }
    store_classes(o, def, d0, d1, d2, comments, regions);
    for (u64 dots = number & m.dot; dots; dots &= dots - 1) /* a dot's default class is C_PUNCT, not C_IDENT */
        o[__builtin_ctzll(dots)] = C_CONSTANT;

    u64 solid = region & ~m.space & ~comments; /* the bytes of tokens: after the last one, is there a newline? */
    u64 newlines = region & m.newline & ~comments;
    if (solid)
        L->line_start = (newlines >> (63 - __builtin_clzll(solid))) > 1;
    else if (newlines)
        L->line_start = true;
    return limit;
}

void tenxel_classify(const u8 *src, size_t n, u8 *cls) {
    Lexer L = {src, src + n, cls, true, false};
    const u8 *p = src;
    if (n >= 3 && !memcmp(src, "\xEF\xBB\xBF", 3)) { /* clang skips a UTF-8 byte order mark */
        memset(cls, C_SPACE, 3);
        p += 3;
    }

    while (p < L.end) {
        if (L.directive) {
            p = directive(&L, p);
            continue;
        }

        /* The last bytes go through a zero-padded copy, where a NUL past the end stops the block */
        size_t left = (size_t)(L.end - p);
        const u8 *b = p;
        u8 *o = L.cls + (p - L.src);
        u8 in[96];
        u8 out[64];
        if (left < 80) {
            memset(in, 0, sizeof in);
            memcpy(in, p, left);
            b = in;
            o = out;
        }
        bool special;
        unsigned limit = code_block(&L, b, p, o, &special);
        if (o == out)
            memcpy(L.cls + (p - L.src), out, limit);
        p += limit;
        if (p >= L.end || (limit && !special))
            continue;

        /* One token that a block can't take, or, when a whole block is words and spaces, tokens for a while */
        const u8 *until = special ? p : p + 64;
        do
            p = lex_token(&L, p);
        while (p < until && p < L.end && !L.directive);
    }
}

/* ---- command line: the file, colored ---- */
static const char *ANSI[11] = {
    "",         "\x1b[1;35m", "\x1b[36m",     "\x1b[1;34m",     "",         "\x1b[33m",
    "\x1b[32m", "\x1b[32m",   "\x1b[2;3;37m", "\x1b[38;5;208m", "\x1b[90m",
};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: tenxel file.c\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror(argv[1]);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    u8 *src = malloc(n + 1);
    u8 *cls = malloc(n + 1);
    if (fread(src, 1, n, f) != n)
        return 1;
    fclose(f);

    tenxel_classify(src, n, cls);
    int cur = -1;
    for (size_t t = 0; t < n; t++) {
        if (cls[t] != cur) {
            fputs("\x1b[0m", stdout);
            fputs(ANSI[cls[t]], stdout);
            cur = cls[t];
        }
        putchar(src[t]);
    }
    fputs("\x1b[0m", stdout);
    return 0;
}
