"""Records the glyph data CarRacing's score is drawn with (DR-27): pygame's default font at size 42.

For each of '0'-'9' and '-': SDL_ttf's glyph box (left, top, width, rows from TTF_GlyphMetrics), its 8-bit coverage
(read from a one-glyph shaded render), and its hinted advance in 26.6 (whole pixels: FreeType's autohinter rounds
them); for each pair of them, the kerning HarfBuzz adds, from the font's kern table scaled as hb_font_t::em_scale_x
does; and the font's ascent, height and line skip. Gym.NET's ScoreFont lays these out as SDL_ttf 2.24 does.

Usage: python record_score_font.py OUTPUT.json   (needs the parity stack's pygame-ce 2.5.8)

The data is not committed until the maintainer decides it may ship (DR-27's licence check): the font is GNU FreeFont's
FreeSans Bold of 2002, whose embedded licence is the GPL. The parity suite records it live instead.
"""
import base64
import json
import os
import struct
import sys

import numpy as np
import pygame

CHARS = "0123456789-"
SIZE = 42


def _tables(data):
    """The font file's table directory.

    Args:
        data: The TrueType file's bytes.

    Returns:
        dict: Each table's tag mapped to its (offset, length) in the file.
    """
    count = struct.unpack(">H", data[4:6])[0]
    tables = {}
    for i in range(count):
        tag, _check, off, length = struct.unpack(">4sIII", data[12 + 16 * i: 28 + 16 * i])
        tables[tag.decode("latin-1")] = (off, length)
    return tables


def _glyph_ids(data, tables):
    """The glyph id of each recorded character, from the Windows Unicode cmap (format 4).

    Args:
        data: The TrueType file's bytes.
        tables: The table directory from _tables.

    Returns:
        dict: Each of CHARS mapped to its glyph id (0 when the font lacks it).

    Raises:
        RuntimeError: The font has no format-4 Unicode cmap subtable.
    """
    cmap = tables["cmap"][0]
    count = struct.unpack(">H", data[cmap + 2: cmap + 4])[0]
    for i in range(count):
        pid, _eid, off = struct.unpack(">HHI", data[cmap + 4 + 8 * i: cmap + 12 + 8 * i])
        sub = cmap + off
        if pid != 3 or struct.unpack(">H", data[sub: sub + 2])[0] != 4:
            continue
        segx2 = struct.unpack(">H", data[sub + 6: sub + 8])[0]
        segs = segx2 // 2
        ends = struct.unpack(f">{segs}H", data[sub + 14: sub + 14 + segx2])
        starts = struct.unpack(f">{segs}H", data[sub + 16 + segx2: sub + 16 + 2 * segx2])
        deltas = struct.unpack(f">{segs}h", data[sub + 16 + 2 * segx2: sub + 16 + 3 * segx2])
        ranges_base = sub + 16 + 3 * segx2
        ranges = struct.unpack(f">{segs}H", data[ranges_base: ranges_base + segx2])
        ids = {}
        for ch in CHARS:
            c = ord(ch)
            for s in range(segs):
                if starts[s] <= c <= ends[s]:
                    if ranges[s] == 0:
                        ids[ch] = (c + deltas[s]) & 0xFFFF
                    else:
                        addr = ranges_base + 2 * s + ranges[s] + 2 * (c - starts[s])
                        g = struct.unpack(">H", data[addr: addr + 2])[0]
                        ids[ch] = (g + deltas[s]) & 0xFFFF if g else 0
                    break
        return ids
    raise RuntimeError("no format-4 Unicode cmap")


def _kern_units(data, tables):
    """The kern table's horizontal format-0 pairs.

    Args:
        data: The TrueType file's bytes.
        tables: The table directory from _tables.

    Returns:
        dict: Each (left glyph id, right glyph id) mapped to its kerning in font units; empty without a kern table.
    """
    pairs = {}
    if "kern" not in tables:
        return pairs
    k = tables["kern"][0]
    _version, ntab = struct.unpack(">HH", data[k: k + 4])
    p = k + 4
    for _ in range(ntab):
        _sver, slen, coverage = struct.unpack(">HHH", data[p: p + 6])
        if coverage >> 8 == 0:
            npairs = struct.unpack(">H", data[p + 6: p + 8])[0]
            for j in range(npairs):
                left, right, value = struct.unpack(">HHh", data[p + 14 + 6 * j: p + 20 + 6 * j])
                pairs[(left, right)] = value
        p += slen
    return pairs


def record():
    """Records the score font's data from the running pygame.

    Returns:
        dict: JSON-ready: font, size, ascent, height and lineskip; glyphs (each character's left, top, width, rows,
        26.6 advance and base64 coverage); kerning (each kerned pair of characters mapped to its 26.6 adjustment).
    """
    pygame.font.init()
    path = os.path.join(os.path.dirname(pygame.__file__), pygame.font.get_default_font())
    font = pygame.font.Font(path, SIZE)
    data = open(path, "rb").read()
    tables = _tables(data)
    head = tables["head"][0]
    upem = struct.unpack(">H", data[head + 18: head + 20])[0]
    ascent = font.get_ascent()
    glyphs = {}
    for ch in CHARS:
        minx, maxx, miny, maxy, advance = font.metrics(ch)[0]
        width, rows = maxx - minx, maxy - miny
        surf = font.render(ch, True, (255, 255, 255), (0, 0, 0))
        levels = np.array(pygame.surfarray.pixels2d(surf)).T  # rows of palette indices = coverage
        x0 = max(minx, 0)  # a negative bearing shifts the line right by -minx (xstart)
        y0 = ascent - maxy
        box = np.ascontiguousarray(levels[y0: y0 + rows, x0: x0 + width], dtype=np.uint8)
        glyphs[ch] = {"left": minx, "top": maxy, "width": width, "rows": rows, "advance": advance * 64,
                      "coverage": base64.b64encode(box.tobytes()).decode("ascii")}
    # HarfBuzz scales the kern table's units by its em multiplier: x_scale is 26.6 per em at this size.
    x_scale = SIZE * 64
    mult = (x_scale << 16) // upem
    ids = _glyph_ids(data, tables)
    units = _kern_units(data, tables)
    kerning = {}
    for a in CHARS:
        for b in CHARS:
            v = units.get((ids[a], ids[b]), 0)
            if v:
                kerning[a + b] = (v * mult + 32768) >> 16
    return {"font": os.path.basename(path), "size": SIZE, "ascent": ascent, "height": font.get_height(),
            "lineskip": font.get_linesize(), "glyphs": glyphs, "kerning": kerning}


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as out:
        json.dump(record(), out, indent=1)
