#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Convert the lexnet neural network into lexnet_ifelse.c and lexnet_fast.c.

    uv run net2c.py                    # reads lexnet_hf/, regenerates the C files here
    uv run net2c.py --src DIR --out DIR

Hash checks, in order:

  Before   SHA-256 of lexnet_weights.bin, lexnet_kernel.h and lexnet.c.
  After    SHA-256 of both generated C files.
  Behavior SHA-256 of the classes and colored output from the original network
           and both generated programs, on stress inputs. Needs a C compiler.

What comes from where:

  From the weights: the 74 words and their order, each word's class, and the perfect
  hash table in lexnet_fast.c. The ANSI table comes from lexnet.c.

  From this script: the C control flow, held below as two templates. It was written by
  reasoning about the gates, so it is not synthesized here. Instead, the script evaluates
  the real weights and refuses to write anything unless they match that logic:
    - every byte predicate against the embedding
    - every reachable forward and backward state pair
    - the word accumulators, on all position and byte pairs (with sampled values)
    - the keyword units, on every threshold interval

Any failed check exits nonzero. --no-pin skips only the pinned hash comparisons, for
experimenting with altered weights. The logic checks still run.
"""
import argparse
import hashlib
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/alpindale/lexnet/resolve"
COMMIT = "c59fe649f8143e81adaff70c32d82da4ca0b6ff3"
INPUT_SHA256 = {
    "lexnet_weights.bin": "0188bedb0765ed5fee5a8763ca09f6dd25faf2eeac5dc775bb06ef306d7cebe2",
    "lexnet_kernel.h": "3eb3b549dde8fa62f6c1809d25eb85d63dcba84e5a7ebe43d405219a0fd07db3",
    "lexnet.c": "568c71ce5152484ac2885bfdba337f9ba812874eb10602de159d8153534db70f",
}
OUTPUT_SHA256 = {
    "lexnet_ifelse.c": "1e7e1441dbe262b90024a0fcab29b51c8d492507a3cfaff597e308f08bdd5ac4",
    "lexnet_fast.c": "af9c20cb779f91353cdc39bd918a37ae3773cbc77ade505f8bf54a834f0ee26e",
}

ALPHA_STATE = "C L S Q P E sf star_in closed Sclose Qclose ls bsP W N d h1 h2".split()
S_W, S_D, S_H1, S_H2 = (ALPHA_STATE.index(n) for n in ("W", "d", "h1", "h2"))
BIT_C, BIT_L, BIT_S, BIT_Q, BIT_P, BIT_W, BIT_N, BIT_KW, BIT_TY, BIT_CONST = range(10)
BIT_OPEN2, BIT_SLASH, BIT_SPACE, BIT_NL, BIT_LPAREN = range(10, 15)


class Fail(Exception):
    """A check failed, so nothing trustworthy can be produced."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def step(msg):
    print(f"  ok  {msg}", flush=True)


# ---- the network: a pure Python port of load() and classify() in lexnet.c ----
class Net:
    def __init__(self, blob):
        self.blob, self.off = blob, 8
        if blob[:8] != b"LEXNET01":
            raise Fail("not a LEXNET01 weight file")
        (self.n_feats, self.n_fstate, self.n_fout, self.n_bstate,
         self.n_classes, self.head_cols) = self._i64(6)
        flat = self._f64(128 * self.n_feats)
        self.emb = [flat[i * self.n_feats:(i + 1) * self.n_feats] for i in range(128)]
        self.fwd, self.bwd = self._cell(), self._cell()
        self.f_state = self._i64(self.n_fstate)
        self.f_out = self._i64(self.n_fout)
        self.b_state = self._i64(self.n_bstate)
        self.h0 = self._f64(self.n_fstate)
        flat = self._f64(self.n_classes * self.head_cols)
        self.head_W = [flat[i * self.head_cols:(i + 1) * self.head_cols] for i in range(self.n_classes)]
        self.head_b = self._f64(self.n_classes)
        if self.off != len(blob):
            raise Fail("trailing bytes in weight file")

    def _i64(self, n):
        v = list(struct.unpack_from(f"<{n}q", self.blob, self.off))
        self.off += 8 * n
        return v

    def _f64(self, n):
        v = list(struct.unpack_from(f"<{n}d", self.blob, self.off))
        self.off += 8 * n
        return v

    def _cell(self):
        layers = []
        for _ in range(self._i64(1)[0]):
            rows, cols, nskip = self._i64(3)
            W = self._f64(rows * cols)
            sparse = [[(c, W[r * cols + c]) for c in range(cols) if W[r * cols + c] != 0] for r in range(rows)]
            layers.append((rows, sparse, self._f64(rows), self._f64(rows), self._i64(nskip)))
        return layers

    @staticmethod
    def run(layers, v):
        for rows, sparse, bias, hi, skip in layers:
            out = []
            for r in range(rows):
                s = bias[r]
                for c, w in sparse[r]:
                    s += w * v[c]
                out.append(0.0 if s < 0 else (hi[r] if s > hi[r] else s))
            out += [v[k] for k in skip]
            v = out
        return v

    def embed(self, byte):
        return self.emb[byte if byte < 128 else 127]

    def forward(self, byte, h):
        v = self.run(self.fwd, self.embed(byte) + h)
        return [v[i] for i in self.f_state], [v[i] for i in self.f_out]

    def backward(self, feats, r):
        v = self.run(self.bwd, feats + r)
        best, bestv = 0, -1e300
        for k in range(self.n_classes):
            s = self.head_b[k]
            for c in range(self.head_cols):
                s += self.head_W[k][c] * v[c]
            if s > bestv:
                best, bestv = k, s
        return [v[i] for i in self.b_state], best


# ---- the logic the C templates implement, written out so it can be checked ----
def is_alpha(c): return (97 <= (c | 32) <= 122 and c < 128) or c == 95
def is_digit(c): return 48 <= c <= 57
def is_ident(c): return is_alpha(c) or is_digit(c)
def is_space(c): return c in (32, 9, 11, 12, 13)


PREDICATES = [
    ("alpha", is_alpha), ("digit", is_digit), ("ident", is_ident),
    ("ident or dot", lambda c: is_ident(c) or c == 46), ("space", is_space),
    ("newline", lambda c: c == 10), ("double quote", lambda c: c == 34),
    ("single quote", lambda c: c == 39), ("backslash", lambda c: c == 92),
    ("slash", lambda c: c == 47), ("star", lambda c: c == 42),
    ("hash", lambda c: c == 35), ("left paren", lambda c: c == 40),
]
CODE, BLOCK_COMMENT, LINE_COMMENT, STRING, CHAR_LIT = range(5)
F_STRING, F_CHAR, F_COMMENT, F_PREPROC, F_WORD, F_NUMBER, F_OPENER = 1, 2, 4, 8, 16, 32, 64
C_SPACE, C_KEYWORD, C_TYPE, C_FUNCTION, C_IDENT, C_CONSTANT = range(6)
C_STRING, C_CHAR, C_COMMENT, C_PREPROC, C_PUNCT = range(6, 11)
FORWARD_START = (CODE, False, False, False, True, False, False, False, False)


def forward_step(state, c):
    mode, last_star, escaped, last_slash, line_start, preproc, last_bs, word, number = state
    newline, flags = c == 10, 0
    if newline and mode != BLOCK_COMMENT:
        mode, escaped = CODE, False
    inside, was_slash, last_slash = mode != CODE, last_slash, False
    if mode == BLOCK_COMMENT:
        flags |= F_COMMENT
        if last_star and c == 47:
            mode = CODE
        last_star = c == 42
    elif mode == LINE_COMMENT:
        flags |= F_COMMENT
    elif mode in (STRING, CHAR_LIT):
        flags |= F_STRING if mode == STRING else F_CHAR
        if escaped:
            escaped = False
        elif c == 92:
            escaped = True
        elif c == (34 if mode == STRING else 39):
            mode = CODE
    elif was_slash and c == 42:
        mode, flags = BLOCK_COMMENT, F_COMMENT | F_OPENER
    elif was_slash and c == 47:
        mode, flags = LINE_COMMENT, F_COMMENT | F_OPENER
    elif c == 34:
        mode, flags = STRING, F_STRING
    elif c == 39:
        mode, flags = CHAR_LIT, F_CHAR
    elif c == 47:
        last_slash = True
    goes_on = preproc and (not newline or last_bs)
    preproc = goes_on or (c == 35 and line_start and not inside)
    last_bs = goes_on and c == 92
    if preproc:
        flags |= F_PREPROC
    line_start = newline or (line_start and is_space(c))
    new_word = new_number = False
    if not inside:
        if (word and is_ident(c)) or (is_alpha(c) and not number):
            new_word = True
        elif (is_ident(c) or c == 46) and (number or is_digit(c)):
            new_number = True
    flags |= (F_WORD if new_word else 0) | (F_NUMBER if new_number else 0)
    return (mode, last_star, escaped, last_slash, line_start, preproc, last_bs, new_word, new_number), flags


def backward_step(state, c, flags, kind):
    paren_ahead, opener, word, word_class = state
    new_class = 0
    if flags & F_WORD:
        new_class = word_class if word else kind if kind else C_FUNCTION if paren_ahead else C_IDENT
    comment = bool(flags & F_COMMENT) or (c == 47 and opener)
    new_state = (c == 40 or (is_space(c) and paren_ahead), bool(flags & F_OPENER), bool(flags & F_WORD), new_class)
    for flag, cls in ((F_STRING, C_STRING), (F_CHAR, C_CHAR)):
        if flags & flag:
            return new_state, cls
    if comment:
        return new_state, C_COMMENT
    if flags & F_PREPROC:
        return new_state, C_PREPROC
    if flags & F_WORD:
        return new_state, new_class
    if flags & F_NUMBER:
        return new_state, C_CONSTANT
    return new_state, C_SPACE if is_space(c) or c == 10 else C_PUNCT


# ---- checks against the real weights ----
def check_predicates(net):
    for col, (name, pred) in enumerate(PREDICATES):
        for byte in range(256):
            if (net.embed(byte)[col] != 0) != bool(pred(byte)):
                raise Fail(f"byte {byte}: embedding column {col} disagrees with '{name}'")
    seen = set()
    for byte in range(128):
        digit = net.emb[byte][13]
        if (digit != 0) != is_ident(byte) or digit != int(digit) or not 0 <= digit <= 63 or digit in seen - {0}:
            raise Fail(f"byte {byte}: bad base-64 digit {digit}")
        seen.add(digit)
    if len(seen - {0}) != 63:
        raise Fail("base-64 digits are not one-to-one on the 63 identifier bytes")
    step("byte predicates match the embedding on all 256 bytes; packing digits are one-to-one")


def check_forward(net):
    start = list(net.h0)
    pair = (tuple(start[:S_H1]), FORWARD_START)
    seen, queue, outputs, edges = {pair}, [pair], set(), 0
    while queue:
        hkey, model = queue.pop()
        h = list(hkey) + [0.0, 0.0]
        for byte in range(256):
            h2, feats = net.forward(byte, h)
            if any(v not in (0.0, 1.0) for v in feats) or any(v not in (0.0, 1.0) for i, v in enumerate(h2[:S_H1]) if i != S_D):
                raise Fail("a logic unit left {0, 1}")
            if h2[S_D] != int(h2[S_D]):
                raise Fail("position counter is not an integer")
            next_model, flags = forward_step(model, byte)
            want = {BIT_S: F_STRING, BIT_Q: F_CHAR, BIT_P: F_PREPROC, BIT_W: F_WORD, BIT_N: F_NUMBER, BIT_OPEN2: F_OPENER}
            for bit, flag in want.items():
                if bool(feats[bit]) != bool(flags & flag):
                    raise Fail(f"forward flag {bit} disagrees on byte {byte}")
            if bool(feats[BIT_C] or feats[BIT_L]) != bool(flags & F_COMMENT):
                raise Fail(f"comment flag disagrees on byte {byte}")
            outputs.add(tuple(int(v) for v in feats))
            edges += 1
            nxt = (tuple(h2[:S_H1]), next_model)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    step(f"forward pass: {len(seen)} reachable state pairs, {edges} edges, no disagreements")
    return outputs


def check_backward(net, outputs):
    inputs = set()
    for f in outputs:                      # the word bits are free: they depend on h1, h2
        for k in (None, BIT_KW, BIT_TY, BIT_CONST):
            g = list(f)
            g[BIT_KW] = g[BIT_TY] = g[BIT_CONST] = 0
            if k is not None:
                g[k] = 1
            inputs.add(tuple(g))
    start = (tuple([0.0] * net.n_bstate), (False, False, False, 0))
    seen, queue, edges = {start}, [start], 0
    while queue:
        r, model = queue.pop()
        for f in inputs:
            r2, cls = net.backward([float(v) for v in f], list(r))
            flags = sum(flag for bit, flag in ((BIT_S, F_STRING), (BIT_Q, F_CHAR), (BIT_P, F_PREPROC), (BIT_W, F_WORD),
                                               (BIT_N, F_NUMBER), (BIT_OPEN2, F_OPENER)) if f[bit])
            if f[BIT_C] or f[BIT_L]:
                flags |= F_COMMENT
            c = 47 if f[BIT_SLASH] else 32 if f[BIT_SPACE] else 10 if f[BIT_NL] else 40 if f[BIT_LPAREN] else 127
            kind = C_KEYWORD if f[BIT_KW] else C_TYPE if f[BIT_TY] else C_CONSTANT if f[BIT_CONST] else 0
            model2, want = backward_step(model, c, flags, kind)
            if cls != want:
                raise Fail(f"backward class: network {cls}, logic {want}, input {f}")
            if (r2[0] != 0) != model2[2] or (r2[2] != 0) != model2[0] or (r2[1] != 0) != model2[1]:
                raise Fail(f"backward state disagrees on input {f}")
            edges += 1
            nxt = (tuple(r2), model2)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    step(f"backward pass: {len(inputs)} inputs x {len(seen)} right-states, {edges} edges, no disagreements")


def word_flags(net, a, b):
    """The word bits for accumulators (a, b). With d = 0 an identifier byte holds them."""
    h = [0.0] * net.n_fstate
    h[S_W], h[S_H1], h[S_H2] = 1.0, float(a), float(b)
    h2, feats = net.forward(ord("q"), h)
    if h2[S_H1] != a or h2[S_H2] != b:
        raise Fail("accumulators did not hold")
    return feats[BIT_KW], feats[BIT_TY], feats[BIT_CONST]


def word_class(net, bit):
    feats = [0.0] * net.n_fout
    feats[BIT_W] = feats[bit] = 1.0
    return net.backward(feats, [0.0] * net.n_bstate)[1]


def extract_words(net):
    """Read the word list out of the thresholds. Returns [(word, class, h1, h2)]."""
    rows, sparse, bias, _, _ = net.fwd[3]
    limits = ([], [])
    for r in range(rows):
        for col, w in sparse[r]:
            if col in (0, 1) and w > 0:
                limits[col].append(int(1 - bias[r]))
    alphabet = {int(net.emb[c][13]): chr(c) for c in range(128) if net.emb[c][13]}

    def decode(v):
        s = ""
        while v:
            s, v = alphabet[v & 63] + s, v >> 6
        return s

    classes = {BIT_KW: word_class(net, BIT_KW), BIT_TY: word_class(net, BIT_TY), BIT_CONST: word_class(net, BIT_CONST)}
    words = []
    for a in sorted(set(limits[0])):
        for b in [0] + sorted(set(limits[1])):
            fired = word_flags(net, a, b)
            for bit, on in zip((BIT_KW, BIT_TY, BIT_CONST), fired):
                if on:
                    words.append((decode(a) + decode(b), classes[bit], a, b))
    words.sort(key=lambda w: (w[3], w[2]))
    if len({w[0] for w in words}) != len(words) or not words:
        raise Fail("word table is empty or has duplicates")
    step(f"read {len(words)} words from the thresholds: "
         + ", ".join(f"{sum(w[1] == c for w in words)} class {c}" for c in sorted({w[1] for w in words})))
    return words


def check_accumulators(net):
    ident = [c for c in range(128) if is_ident(c)]
    rng, checks = random.Random(1), 0
    for W in (0, 1):
        for d in range(15):
            consumed = 15 - d if d else 14
            m1 = 64 ** min(consumed, 8)
            m2 = 64 ** min(max(consumed - 8, 0), 6)
            for c in ident:
                for it in range(12):
                    a, b = ((0, 0), (m1 - 1, m2 - 1), (m1 // 2, m2 // 2))[it] if it < 3 else (rng.randrange(m1), rng.randrange(m2))
                    if not W:                  # a word start must discard whatever was there
                        a, b = rng.getrandbits(48), rng.getrandbits(36)
                    h = [0.0] * net.n_fstate
                    h[S_W], h[S_D], h[S_H1], h[S_H2] = float(W), float(d), float(a), float(b)
                    h2, _ = net.forward(c, h)
                    digit = int(net.emb[c][13])
                    if not W:
                        want = digit, 0, 14
                    elif d >= 8:
                        want = a * 64 + digit, b, d - 1
                    elif d >= 2:
                        want = a, b * 64 + digit, d - 1
                    else:
                        want = a, b, max(d - 1, 0)
                    if (h2[S_H1], h2[S_H2], h2[S_D]) != tuple(float(v) for v in want):
                        raise Fail(f"accumulators disagree: W={W} d={d} byte={c}")
                    if h2[S_W] != (1.0 if W or not is_digit(c) else 0.0):
                        raise Fail(f"word flag disagrees: W={W} d={d} byte={c}")
                    checks += 1
    step(f"accumulators: {checks} checks on all (W, d, byte), with sampled h values, no disagreements")


def check_keywords(net, words):
    want = {(w[2], w[3]): w[1] for w in words}
    near = lambda vals, extra: sorted({v + k for v in vals for k in range(-2, 3) if v + k >= 0} | set(extra))
    A = near({w[2] for w in words}, list(range(8)) + [2 ** 48 - 1, 2 ** 48])
    B = near({w[3] for w in words}, list(range(8)) + [2 ** 36 - 1, 2 ** 36])
    bits = (BIT_KW, BIT_TY, BIT_CONST)
    classes = [word_class(net, b) for b in bits]
    for a in A:
        for b in B:
            fired = [classes[i] for i, on in enumerate(word_flags(net, a, b)) if on]
            expect = [want[a, b]] if (a, b) in want else []
            if fired != expect:
                raise Fail(f"keyword units: h1={a} h2={b} fired {fired}, expected {expect}")
    step(f"keyword units: {len(A)} x {len(B)} points on every threshold interval, no disagreements")


# ---- rendering ----
# Words are listed by (h2, h1), the network's own packed values. This one pair was ordered by
# hand in the original file. Order has no effect on behavior.
LAYOUT_PAIRS = [("_Thread_local", "thread_local")]


def render_word_tests(words):
    lines = []
    for cls, name in ((C_KEYWORD, "C_KEYWORD"), (C_TYPE, "C_TYPE"), (C_CONSTANT, "C_CONSTANT")):
        names = [w[0] for w in words if w[1] == cls]
        for first, second in LAYOUT_PAIRS:
            if first in names and second in names:
                names.remove(first)
                names.insert(names.index(second), first)
        items = [f'W("{name}")' for name in names]
        line = "    if ("
        for i, item in enumerate(items):
            tail = " ||" if i + 1 < len(items) else ")"
            piece = item + tail
            if len(line) + len(piece) > 99 and not line.endswith("("):
                lines.append(line.rstrip())
                line = "        "
            line += piece + " "
        lines.append(line.rstrip())
        lines.append(f"        return {name};")
    return "\n".join(lines)


def render_word_table(words, bits=8):
    mask, k2 = (1 << 64) - 1, 0x9E3779B97F4A7C15
    keys = [struct.unpack("<QQ", w[0].encode()[:14].ljust(16, b"\0")) for w in words]
    mixed = [(a + b * k2) & mask for a, b in keys]
    rng, shift, tries = random.Random(1), 64 - bits, 0
    while True:
        tries += 1
        m = rng.getrandbits(64) | 1
        seen = set()
        for v in mixed:
            h = (v * m & mask) >> shift
            if h in seen:
                break
            seen.add(h)
        else:
            break
    out = [f"/* perfect hash: {tries} tries */", f"#define HASH_MUL 0x{m:016X}ULL", f"#define HASH_BITS {bits}",
           "static const WordEntry WORDS[1 << HASH_BITS] = {"]
    for h, i in sorted(((mixed[i] * m & mask) >> shift, i) for i in range(len(words))):
        a, b = keys[i]
        out.append(f"    [{h}] = {{0x{a:016X}ULL, 0x{b | words[i][1] << 56:016X}ULL}}, /* {words[i][0]} */")
    out.append("};")
    step(f"found a perfect hash for {len(words)} words after {tries} tries")
    return "\n".join(out)


def render(template, **parts):
    for key, value in parts.items():
        template = template.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise Fail("unfilled template slot")
    return template


# ---- behavior: build all three programs and compare output hashes ----
def stress_inputs(src, generated, words):
    rng = random.Random(1)
    hot = b"/*\"'\\\n#( .a0_"
    near = []
    for w, *_ in words:
        near += [w, w + "x", "x" + w, w[:-1], w + "_" * 16, w + "(", w + " (", "1" + w, "." + w, '"' + w, "/*" + w + "*/"]
    tokens = "\n ;(".encode()
    real = b"".join((src / n).read_bytes() for n in ("lexnet.c", "lexnet_kernel.h")) + b"".join(generated.values())
    soup = bytearray()
    while len(soup) < 300_000:
        soup += rng.choice(near).encode() + bytes([rng.choice(tokens)])
    return {
        "empty": b"",
        "real C": real,
        "word soup": bytes(soup),
        "hot bytes": bytes(rng.choices(hot, k=400_000)),
        "all bytes": rng.randbytes(200_000),
        "no final newline": b'#define X "a\\\n/* open',
    }


def check_behavior(src, generated, words):
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise Fail("no C compiler found (set CC, or pass --no-build to skip this check)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in ("lexnet.c", "lexnet_kernel.h"):
            shutil.copy(src / name, tmp / name)
        for name, data in generated.items():
            (tmp / name).write_bytes(data)
        bins = {"network": "lexnet.c", "ifelse": "lexnet_ifelse.c", "fast": "lexnet_fast.c"}
        for out, source in bins.items():
            subprocess.run([cc, "-O3", "-march=native", "-w", "-o", str(tmp / out), str(tmp / source), "-lm"], check=True)
        weights = str(src.resolve() / "lexnet_weights.bin")
        rows, failed = [], False
        for label, data in stress_inputs(src, generated, words).items():
            path = tmp / "input"
            path.write_bytes(data)
            for mode, args in (("classes", ["--classes"]), ("colored", [])):
                hashes = {}
                for name in bins:
                    cmd = [str(tmp / name)] + (["-w", weights] if name == "network" else []) + args + [str(path)]
                    hashes[name] = sha256(subprocess.run(cmd, check=True, capture_output=True).stdout)
                same = len(set(hashes.values())) == 1
                failed |= not same
                rows.append((label, mode, len(data), hashes["network"], same))
        return rows, failed


def fetch(src):
    pointer = src / "lexnet_weights.bin"
    if pointer.exists() and pointer.read_bytes().startswith(b"version https://git-lfs"):
        raise Fail(f"{pointer} is a Git LFS pointer, not the weights. Run: git lfs install && git submodule update --init --force")
    src.mkdir(parents=True, exist_ok=True)
    for name in INPUT_SHA256:
        path = src / name
        if not path.exists():
            print(f"  ..  downloading {name} at {COMMIT[:7]}", flush=True)
            with urllib.request.urlopen(f"{REPO}/{COMMIT}/{name}", timeout=60) as r:
                path.write_bytes(r.read())


def main():
    ap = argparse.ArgumentParser(description="Convert the lexnet network into C, checking hashes before and after.")
    ap.add_argument("--src", type=Path, default=Path("lexnet_hf"), help="folder with the original files (default: lexnet_hf)")
    ap.add_argument("--out", type=Path, default=Path("."), help="folder for the generated C (default: this folder)")
    ap.add_argument("--no-pin", action="store_true", help="skip pinned hash comparisons; logic checks still run")
    ap.add_argument("--no-build", action="store_true", help="skip compiling and comparing program output")
    args = ap.parse_args()
    began = time.time()
    try:
        print("Before: input hashes")
        fetch(args.src)
        for name, want in INPUT_SHA256.items():
            got = sha256((args.src / name).read_bytes())
            if got != want and not args.no_pin:
                raise Fail(f"{name}: expected {want}, got {got}")
            print(f"  {'ok' if got == want else '!!'}  {got}  {name}", flush=True)

        print("Conversion: reading the logic out of the real weights")
        net = Net((args.src / "lexnet_weights.bin").read_bytes())
        check_predicates(net)
        outputs = check_forward(net)
        check_backward(net, outputs)
        words = extract_words(net)
        check_accumulators(net)
        check_keywords(net, words)
        ansi = re.search(r"static const char \*ANSI\[11\] = \{.*?\n\};", (args.src / "lexnet.c").read_text(), re.S)
        if not ansi:
            raise Fail("ANSI table not found in lexnet.c")
        parts = {"ANSI": ansi.group(0), "N_WORDS": str(len(words))}
        generated = {
            "lexnet_ifelse.c": render(IFELSE_TEMPLATE, WORD_TESTS=render_word_tests(words), **parts).encode(),
            "lexnet_fast.c": render(FAST_TEMPLATE, WORD_TABLE=render_word_table(words), **parts).encode(),
        }

        args.out.mkdir(parents=True, exist_ok=True)
        print("After: generated hashes")
        bad = []
        for name, data in generated.items():
            (args.out / name).write_bytes(data)
            got, want = sha256(data), OUTPUT_SHA256[name]
            if got != want and not args.no_pin:
                bad.append(name)
            print(f"  {'ok' if got == want else '!!'}  {got}  {args.out / name}", flush=True)
        if bad:
            raise Fail(f"generated output differs from the pinned hash: {', '.join(bad)}")

        if args.no_build:
            print("Behavior: skipped (--no-build)")
        else:
            print("Behavior: network vs ifelse vs fast, SHA-256 of program output")
            rows, failed = check_behavior(args.src, generated, words)
            for label, mode, size, digest, same in rows:
                print(f"  {'ok' if same else '!!'}  {digest}  {label}, {mode} ({size} bytes)", flush=True)
            if failed:
                raise Fail("program output differs from the network")
    except (Fail, subprocess.CalledProcessError, OSError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(f"Done in {time.time() - began:.1f}s. Wrote {args.out / 'lexnet_ifelse.c'} and {args.out / 'lexnet_fast.c'}.")
    return 0


IFELSE_TEMPLATE = r'''/* lexnet_ifelse.c - the lexnet "neural network", written as the program it always was.
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
@@WORD_TESTS@@
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
@@ANSI@@

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
'''

FAST_TEMPLATE = r'''/* lexnet_fast.c - the lexnet network, compiled all the way down.
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

/* ---- perfect hash over the @@N_WORDS@@ words ----
 * A word is at most 14 significant bytes. .a holds bytes 0-7, .b holds bytes 8-13 with the
 * class in its top byte. Anything past 14 is ignored, which keeps the network's quirk:
 * "_Static_assertXYZ" counts as a keyword. */
typedef struct { u64 a, b; } WordEntry;
@@WORD_TABLE@@

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
@@ANSI@@

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
'''

if __name__ == "__main__":
    sys.exit(main())
