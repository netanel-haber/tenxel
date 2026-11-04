#!/usr/bin/env python3
"""The reference for verify.sh: the class of every byte of a C file, according to clang's raw lexer.

    python3 oracle.py FILE                    # one class digit per byte
    python3 oracle.py --check LIB [MB] [SEED] # compare tenxel_classify in LIB with clang on generated inputs

LIB is tenxel.c built as a shared library: cc -O2 -march=native -shared -fPIC -o tenxel.so tenxel.c

Token boundaries and kinds come from `clang -cc1 -std=gnu23 -dump-raw-tokens`. This script adds what
the raw lexer leaves out: keyword classes (the KEYWORDS table in tenxel.c), directives (a #
or %: that is the first token on its line, up to the next newline outside a comment) and functions
(an identifier whose next token, past spaces and comments, is "("). Unterminated literals and
comments, which clang calls unknown tokens, keep the class of what they started as.
"""
import ctypes
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SPACE, KEYWORD, TYPE, FUNCTION, IDENT, CONSTANT, STRING, CHAR, COMMENT, PREPROC, PUNCT = range(11)
HERE = Path(__file__).resolve().parent
NAME = "in.c"
RECORD = re.compile(rb"Loc=<" + re.escape(NAME.encode()) + rb":(\d+):(\d+)>(?:\n|$)")
SPLICE = re.compile(rb"\\[ \t\f\v]*(\r\n|\n\r|\n|\r)")
CLANG = ["clang", "-cc1", "-std=gnu23", "-triple", "x86_64-pc-linux-gnu", "-dump-raw-tokens", NAME]


def keywords():
    body = (HERE / "tenxel.c").read_text().split("static const Keyword KEYWORDS[KEYWORD_SLOTS] = {", 1)[1].split("};", 1)[0]
    names = {"C_KEYWORD": KEYWORD, "C_TYPE": TYPE, "C_CONSTANT": CONSTANT}
    return {w.encode(): names[c] for w, c in re.findall(r'\{"(\w+)",\s*(C_\w+)\}', body)}


KEYWORDS = keywords()


def logical(text):
    return SPLICE.sub(b"", text)


def line_starts(src):
    """Byte offset of each line the way clang numbers them: \\n, \\r and \\r\\n each end one."""
    starts, i, n = [0], 0, len(src)
    while i < n:
        if src[i] in (10, 13):
            if src[i] == 13 and i + 1 < n and src[i + 1] == 10:
                i += 1
            starts.append(i + 1)
        i += 1
    return starts


def token_length(head, kind, src, s):
    """The length of the token at s, from its record "kind 'spelling'<TAB>flags<TAB>". A token with
    splices has the flag [UnClean='raw text']; otherwise the spelling is the raw text."""
    head = head[:-1]
    marker = b" [UnClean='"
    if head.endswith(b"']"):
        k = head.rfind(marker, 0, len(head) - 2)
        while k >= 0:
            raw = head[k + len(marker):-2]
            if src[s:s + len(raw)] == raw:
                return len(raw)
            k = head.rfind(marker, 0, k)
    prefix = kind.encode() + b" '"
    for flags in (b"", b" [StartOfLine]", b" [LeadingSpace]", b" [StartOfLine] [LeadingSpace]"):
        raw = head[len(prefix):len(head) - len(b"'\t" + flags)]
        if head.endswith(b"'\t" + flags) and src[s:s + len(raw)] == raw:
            return len(raw)
    raise RuntimeError(f"cannot measure the last token: {head[:200]!r}")


def tokens(src, workdir):
    """[(kind, start, end)] covering every byte of src."""
    (Path(workdir) / NAME).write_bytes(src)
    out = subprocess.run(CLANG, cwd=workdir, capture_output=True).stderr
    starts, toks, pos, head = line_starts(src), [], 0, b""
    for m in RECORD.finditer(out):
        kind = out[pos:out.index(b" ", pos)].decode()
        start = starts[int(m.group(1)) - 1] + int(m.group(2)) - 1
        # clang counts a column from the last \r or \n, so the \r and the \n of a \r\n both get
        # column 1. A token starts at the \n only after a splice that ended in "\n\r".
        if src[start:start + 2] == b"\r\n" and start > 0 and src[start - 1] == 10:
            j = start - 2
            while j >= 0 and src[j] in b" \t\f\v":
                j -= 1
            start += j >= 0 and src[j] == 92
        toks.append((kind, start))
        head, pos = out[pos:m.start()], m.end()
    if not toks or toks[0][1] > 0:   # a UTF-8 byte order mark, which clang skips
        toks.insert(0, ("bom", 0))
    if len(toks) > 1 or toks[0][0] != "bom":
        # Splices at the very end belong to no token; they count as space.
        end = toks[-1][1] + token_length(head, toks[-1][0], src, toks[-1][1])
        if end < len(src):
            toks.append(("unknown", end))
    toks.append(("eof", len(src)))
    return [(k, s, toks[i + 1][1]) for i, (k, s) in enumerate(toks[:-1])]


def first_newline(src, s, e):
    """Offset of the first newline in src[s:e] that is not part of a splice, or e."""
    i = s
    while i < e:
        if src[i] in (10, 13):
            return i
        m = SPLICE.match(src, i) if src[i] == 92 else None
        i = m.end() if m else i + 1
    return e


def token_class(kind, text):
    if kind == "comment":
        return COMMENT
    if kind.endswith("string_literal"):
        return STRING
    if kind.endswith("char_constant"):
        return CHAR
    if kind == "numeric_constant":
        return CONSTANT
    if kind == "raw_identifier":
        return KEYWORDS.get(logical(text), IDENT)
    if kind in ("unknown", "bom"):
        t = logical(text)
        if kind == "bom" or not t.strip(b" \t\f\v\r\n\0"):
            return SPACE
        if re.match(rb'(u8|u|U|L)?R?"', t):
            return STRING
        if re.match(rb"(u8|u|U|L)?'", t):
            return CHAR
        if t.startswith(b"/*"):
            return COMMENT
    return PUNCT


def classes(src, workdir):
    toks = tokens(src, workdir)
    kinds = [token_class(k, src[s:e]) for k, s, e in toks]
    significant = [i for i, c in enumerate(kinds) if c not in (SPACE, COMMENT)]
    for j, i in enumerate(significant):
        if kinds[i] == IDENT and j + 1 < len(significant) and toks[significant[j + 1]][0] == "l_paren":
            kinds[i] = FUNCTION
    out = bytearray(len(src))
    line_start, directive = True, False
    for (kind, s, e), c in zip(toks, kinds):
        nl = first_newline(src, s, e) if c == SPACE else e
        if directive:
            if nl < e:   # the directive ends at this newline
                out[s:nl] = bytes([PREPROC]) * (nl - s)
                out[nl:e] = bytes([SPACE]) * (e - nl)
                directive, line_start = False, True
            else:
                out[s:e] = bytes([c if c in (COMMENT, STRING, CHAR) else PREPROC]) * (e - s)
            continue
        if kind == "hash" and line_start:
            directive, c = True, PREPROC
        out[s:e] = bytes([c]) * (e - s)
        if nl < e:
            line_start = True
        elif c not in (SPACE, COMMENT):
            line_start = False
    return bytes(out)


# ---- generated inputs ----
WORDS = [w.decode() for w in KEYWORDS] + ["x", "foo", "size_t", "NULL", "intx", "_", "$a", "a$", "L", "u", "U", "R", "u8",
                                          "Lx", "uR", "u8R", "LR", "UR", "in\\\nt", "__builtin_va_argX"]
PIECES = [
    "0", "1", "42", "0x1F", "0x1p-3", "1e+5", "1E-5", "1.5f", ".5", "1.", "1..2", "...", "..", "..5", "...5", "1'000",
    "1'a", "1'_", "1'$", "0x1e+5", "1e+e+5", "1'e+5", "1\\\n2", "1e\\\n+5", "1'\\\n0", ".\\\n5",
    '"s"', '"a\\"b"', '"a\\\\"', '"unterminated', '"a\\\nb"', '"a\\\\\nb"', "'c'", "'\\''", "''", "'", "'ab'", "don't",
    'L"w"', "u8\"s\"", "u'c'", "U'c'", "u8'c'", "L'c'", 'u\\\n8"s"', 'R"(raw)"', 'R"x(a)"b)x"', 'R"(un', 'R"bad delim(x)"',
    'u8R"(z)"', 'LR"-(q)-"', 'R"(a\\\nb)"', 'R"\\\n(x)"',
    "//c", "// c \\\n still", "//\\\n", "/**/", "/* c */", "/*/ x */", "/* a\nb */", "/***/", "/* open", "*/", "/\\\n* c *\\\n/",
    "/\\\n/ c", "/* *\\ \n/", "/* *\\\n\\\n/", "/* *\\\x00\n/",
    "#", "##", "%:", "%:%:", "%:%", "#define X 1", "# include <a.h>", "#if 0", "#x\\\ny", "<:", ":>", "<%", "%>", "%=",
    "(", ")", "{", "}", "[", "]", ";", ",", "->", "+", "-", "*", "/", "=", "<", ">", "!", "~", "?", ":", "&", "|", "^", "%",
    "@", "`", "\\", "\\u00e9", "\\u03b1x", "\\u0041", "\\u00a0", "\\ud800", "\\U0001F600", "\\U00110000", "\\u{e9}",
    "\\u{3b1}", "\\U{e9}", "\\u{}", "\\u12", "\\N{X}", "\\N{}", "\\N{A\nB}", "\\x",
    "é", "α", "→", "😀", "·", "\u00a0", "\u3000", "\u2028", "\u0301", "\u200b", "∂", "\ufeff", "１",
    " ", "  ", "\t", "\n", "\r\n", "\r", "\n\r", "\f", "\v", "\\\n", "\\ \n", "\\\r\n", "\\\t\r", "\x00", "\x1a", "\x7f",
]
BYTES = [b"\xff", b"\x80", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf4\x90\x80\x80", b"\xe2\x86", b"\xc3"]
SEPARATORS = [" ", "", "\n", "(", " (", "\n(", "/**/(", ";", "\\\n", "\t"]


def token_soup(rng, size):
    out = bytearray()
    while len(out) < size:
        r = rng.random()
        if r < 0.03:
            out += rng.choice(BYTES)
        else:
            out += rng.choice(WORDS if r < 0.35 else PIECES).encode()
        out += rng.choice(SEPARATORS).encode()
    return bytes(out)


def mutate(rng, src, n):
    b = bytearray(src)
    for _ in range(n):
        at = rng.randrange(len(b) + 1)
        r = rng.random()
        if r < 0.4:
            b[at:at] = rng.choice(["\\\n", "\\ \n", "\\\r\n"]).encode()
        elif r < 0.7:
            b[at:at] = rng.choice(PIECES).encode()
        elif at < len(b):
            del b[at]
    return bytes(b)


def corpus():
    files = sorted(HERE.glob("*.c")) + sorted(Path("/usr/include").glob("*.h"))
    return [f.read_bytes() for f in files if f.is_file()]


def check_keywords(workdir):
    """Every word in the KEYWORDS table must be a keyword to clang (its preprocessing lexer names their kinds)."""
    (Path(workdir) / NAME).write_bytes(b"\n".join(KEYWORDS) + b"\n")
    out = subprocess.run(CLANG[:-2] + ["-dump-tokens", NAME], cwd=workdir, capture_output=True).stderr
    plain = re.findall(rb"^identifier '(\w+)'", out, re.M)
    if plain:
        print("NOT KEYWORDS to clang:", b" ".join(plain).decode())
        return False
    return True


def check(library, megabytes, seed):
    tenxel = ctypes.CDLL(str(Path(library).resolve()))
    tenxel.tenxel_classify.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
    tenxel.tenxel_classify.restype = None
    rng, real, checked, cases = random.Random(seed), corpus(), 0, 0
    with tempfile.TemporaryDirectory() as d:
        if not check_keywords(d):
            return 1
        while checked < megabytes * 1e6:
            kind = cases % 4
            if kind == 0:
                src = rng.choice(real)
            elif kind == 1:
                src = token_soup(rng, rng.choice([50, 400, 4000, 30000]))
            elif kind == 2:
                base = rng.choice(real)
                at = rng.randrange(len(base))
                src = mutate(rng, base[at:at + rng.choice([200, 3000, 30000])], rng.randrange(1, 40))
            else:
                src = bytes(rng.choice(b"/*\"'\\\n#( .a0_uLR8e+%:\r\t\x00\x80{}") for _ in range(rng.choice([20, 200, 2000])))
            want = classes(src, d)
            out = ctypes.create_string_buffer(len(src))
            tenxel.tenxel_classify(src, len(src), out)
            got = out.raw
            if got != want:
                i = next(i for i in range(len(src)) if i >= len(got) or got[i] != want[i])
                lo, hi = max(0, i - 40), min(len(src), i + 20)
                Path("oracle_fail.c").write_bytes(src)
                print(f"MISMATCH in case {cases} (kind {kind}) at byte {i}; input saved as oracle_fail.c")
                print("  bytes:", src[lo:hi])
                print("  clang:", "".join("0123456789A"[c] for c in want[lo:hi]))
                print("  got:  ", "".join("0123456789A"[c] for c in got[lo:hi]))
                return 1
            checked += len(src)
            cases += 1
    print(f"ok: {Path(library).name} matches clang on {cases} inputs ({checked / 1e6:.1f} MB)")
    return 0


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--check":
        return check(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2, int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    if len(sys.argv) == 2:
        with tempfile.TemporaryDirectory() as d:
            sys.stdout.write("".join("0123456789A"[c] for c in classes(Path(sys.argv[1]).read_bytes(), d)))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
