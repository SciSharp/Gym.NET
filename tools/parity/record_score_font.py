"""Records the glyph data CarRacing's score is drawn with (DR-27, DR-48): pygame's default font at size 42.

The score is Python's f"{reward:04.0f}" of a float, which can show 15 characters: the digits, '-', and the letters of
"nan" and "inf" (any NaN shows as "0nan", the infinities as "0inf" and "-inf"). For each of them: SDL_ttf's glyph box
(left, top, width, rows from TTF_GlyphMetrics), its 8-bit coverage (read from a one-glyph shaded render), and its hinted
advance in 26.6 (whole pixels: FreeType rounds them); for each pair of them, the kerning HarfBuzz adds, from the font's
kern table scaled as hb_font_t::em_scale_x does; the sequences of them HarfBuzz replaces by a ligature (the font's
'liga' feature turns "fi" into one glyph), which Gym.NET's ScoreFont refuses, since no score contains them; the font's
ascent, height and line skip; and the font file's name and SHA-256. Gym.NET's ScoreFont lays these out as SDL_ttf 2.24
does.

The data ships embedded in Gym.NET.Environments as src/Gym.Environments/Assets/fonts/freesansbold-42.json (the
maintainer's decision of 2026-11-11, DR-48). The parity cell
CarRacingLiveParityTests.GymNet_ScoreFont_EmbeddedDataIsTheLiveRecording compares it with a live recording, so a
pygame-ce re-pin that changes the font fails there. Re-record with the parity stack's pygame-ce 2.5.8:

    python tools/parity/record_score_font.py src/Gym.Environments/Assets/fonts/freesansbold-42.json
"""
import base64
import hashlib
import json
import os
import struct
import sys

import numpy as np
import pygame

# Every character f"{x:04.0f}" can produce for a float: the digits, the sign, and the letters of "nan" and "inf".
CHARS = "0123456789-afin"
SIZE = 42

# The GSUB features HarfBuzz applies to horizontal text by default (its common and horizontal features). The font's other
# features, 'init' and 'medi', belong to the Arabic shaper and never apply to these characters.
DEFAULT_GSUB_FEATURES = {"ccmp", "locl", "rlig", "calt", "clig", "liga", "rclt"}


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


def _coverage(data, offset):
    """An OpenType coverage table's glyphs, in coverage-index order.

    Args:
        data: The TrueType file's bytes.
        offset: The coverage table's offset in the file.

    Returns:
        list: The glyph ids; the i-th one is covered with index i.

    Raises:
        RuntimeError: The table has a format other than 1 (a glyph array) or 2 (glyph ranges).
    """
    fmt, count = struct.unpack(">HH", data[offset: offset + 4])
    if fmt == 1:
        return list(struct.unpack(f">{count}H", data[offset + 4: offset + 4 + 2 * count]))
    if fmt != 2:
        raise RuntimeError(f"coverage format {fmt}")
    glyphs = []
    for r in range(count):
        # Ranges come in coverage-index order, so their glyphs line up with the indices when concatenated.
        start, end, _first_index = struct.unpack(">HHH", data[offset + 4 + 6 * r: offset + 10 + 6 * r])
        glyphs.extend(range(start, end + 1))
    return glyphs


def _ligatures(data, tables, ids):
    """The sequences of recorded characters that the font's default GSUB features replace by one ligature glyph.

    Only the FeatureList is read, not the ScriptList's per-script choice: pygame's font has one script ('latn'), whose
    default language system lists every feature.

    Args:
        data: The TrueType file's bytes.
        tables: The table directory from _tables.
        ids: Each recorded character's glyph id, from _glyph_ids.

    Returns:
        list: The sequences (strings of two or more recorded characters), sorted; empty without a GSUB table.

    Raises:
        RuntimeError: A default feature uses a lookup other than a ligature substitution, which the layout does not
            model, or a coverage table of an unknown format.
    """
    if "GSUB" not in tables:
        return []
    chars = {g: ch for ch, g in ids.items() if g}
    base = tables["GSUB"][0]

    def u16(offset):
        return struct.unpack(">H", data[offset: offset + 2])[0]

    feature_list = base + u16(base + 6)
    lookup_list = base + u16(base + 8)
    lookups = set()
    for i in range(u16(feature_list)):
        record = feature_list + 2 + 6 * i
        if data[record: record + 4].decode("latin-1") in DEFAULT_GSUB_FEATURES:
            feature = feature_list + u16(record + 4)
            lookups.update(u16(feature + 4 + 2 * j) for j in range(u16(feature + 2)))
    found = set()
    for index in sorted(lookups):
        lookup = lookup_list + u16(lookup_list + 2 + 2 * index)
        kind = u16(lookup)
        for k in range(u16(lookup + 4)):
            sub = lookup + u16(lookup + 6 + 2 * k)
            sub_kind = kind
            if kind == 7:
                # An extension lookup: the real subtable and its type sit behind a 32-bit offset.
                sub_kind = u16(sub + 2)
                sub += struct.unpack(">I", data[sub + 4: sub + 8])[0]
            if sub_kind != 4:
                raise RuntimeError(f"GSUB lookup {index} (type {sub_kind}) of a default feature is not modelled")
            for position, first in enumerate(_coverage(data, sub + u16(sub + 2))):
                if first not in chars:
                    continue
                ligature_set = sub + u16(sub + 6 + 2 * position)
                for m in range(u16(ligature_set)):
                    ligature = ligature_set + u16(ligature_set + 2 + 2 * m)
                    # ligatureGlyph, componentCount, then the components after the first (the covered glyph).
                    components = [u16(ligature + 4 + 2 * c) for c in range(u16(ligature + 2) - 1)]
                    if all(c in chars for c in components):
                        found.add(chars[first] + "".join(chars[c] for c in components))
    return sorted(found)


def record():
    """Records the score font's data from the running pygame.

    Returns:
        dict: JSON-ready: font, sha256, size, ascent, height and lineskip; glyphs (each character's left, top, width,
        rows, 26.6 advance and base64 coverage); kerning (each kerned pair of characters mapped to its 26.6
        adjustment); ligatures (the character sequences the font's default GSUB features substitute).

    Raises:
        RuntimeError: The font lacks a format-4 Unicode cmap, or its GSUB uses what _ligatures does not model.
    """
    pygame.font.init()
    path = os.path.join(os.path.dirname(pygame.__file__), pygame.font.get_default_font())
    font = pygame.font.Font(path, SIZE)
    with open(path, "rb") as f:
        data = f.read()
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
        y0 = max(ascent - maxy, 0)  # a glyph above the ascent shifts the line down (ystart)
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
    return {"font": os.path.basename(path), "sha256": hashlib.sha256(data).hexdigest(), "size": SIZE, "ascent": ascent,
            "height": font.get_height(), "lineskip": font.get_linesize(), "glyphs": glyphs, "kerning": kerning,
            "ligatures": _ligatures(data, tables, ids)}


if __name__ == "__main__":
    # LF endings and a final newline, so the committed file reads the same on every OS.
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as out:
        json.dump(record(), out, indent=1)
        out.write("\n")
