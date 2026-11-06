#!/usr/bin/env python3
"""The blocks' keyword table in tenxel.c, WORD_MUL and WORDS, made from its KEYWORDS table.

    python3 words.py            # prints the table
    python3 words.py --write    # puts it in tenxel.c, in place of the one there

A key is a keyword's first 8 bytes read as a little-endian number, zero-padded if it is shorter. Each key sits at
the slot it hashes to, (key * WORD_MUL) >> 55, and the multiplier is the first one a seeded search finds that puts
no two keys in one slot. A slot's meta is len << 4 | longer << 3 | class ^ C_IDENT: the length and class of the
keyword of up to 8 bytes with that key, if there is one, and whether a longer keyword starts with those 8 bytes.
exhaust.c checks the table against KEYWORDS.
"""
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tenxel.c"
CLASS_X = {"C_KEYWORD": 1 ^ 4, "C_TYPE": 2 ^ 4, "C_CONSTANT": 5 ^ 4}  # each class XOR C_IDENT
START = "#define WORD_MUL "
END = "\n};\n"


def keywords(src):
    body = src.split("static const Keyword KEYWORDS[KEYWORD_SLOTS] = {", 1)[1].split("};", 1)[0]
    return re.findall(r'\{"(\w+)",\s*(C_\w+)\}', body)


def entries(kws):
    """key -> [length of the keyword of up to 8 bytes with it (or 0), its class XOR C_IDENT, longer, the 8 bytes]"""
    out = {}
    for word, cls in kws:
        key = int.from_bytes(word.encode()[:8].ljust(8, b"\0"), "little")
        e = out.setdefault(key, [0, 0, 0, word[:8]])
        if len(word) <= 8:
            e[0], e[1] = len(word), CLASS_X[cls]
        else:
            e[2] = 1
    return out


def multiplier(keys):
    rng = random.Random(7)
    mask = (1 << 64) - 1
    while True:
        mul = rng.getrandbits(64) | 1
        slots = [((k * mul) & mask) >> 55 for k in keys]
        if len(set(slots)) == len(slots):
            return mul, slots


def table(src):
    es = entries(keywords(src))
    keys = list(es)
    mul, slots = multiplier(keys)
    code, notes = [], []
    for s, k in sorted(zip(slots, keys)):
        length, x, longer, name = es[k]
        code.append(f"    [{s}] = {{0x{k:X}, {length << 4 | longer << 3 | x}}},")
        notes.append(f"/* {name}{'...' if longer else ''} */")
    width = max(map(len, code)) + 1  # the column clang-format aligns trailing comments to
    lines = [f"{START}0x{mul:016X}ULL", "static const WordSlot WORDS[512] = {"]
    lines += [c.ljust(width) + n for c, n in zip(code, notes)]
    return "\n".join(lines + ["};"]) + "\n"


def main():
    src = SOURCE.read_text()
    new = table(src)
    if sys.argv[1:] != ["--write"]:
        print(new, end="")
        return
    start = src.index(START)
    end = src.index(END, start) + len(END)
    SOURCE.write_text(src[:start] + new + src[end:])


if __name__ == "__main__":
    main()
