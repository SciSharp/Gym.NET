"""Converts MuJoCo's bitmap font source (render/classic/font/normal150.inc) into the JSON Gym.NET embeds (DR-70).

MuJoCo draws its overlay text (mjr_overlay, the menu of Gymnasium's MuJoCo window) with glBitmap from a font compiled into
the library: per printable character (32 to 126) its code, width and height in pixels, then its 1-bit rows from the
bottom, each row padded to whole bytes, the most significant bit leftmost; the list ends with the byte 123. Gymnasium's
window uses the context's normal font at its font scale, mjFONTSCALE_150, so normal150.inc is the one Gym.NET needs.

The JSON keeps MuJoCo's own byte layout, base64-encoded, beside the source's identity:

    {"source": ..., "tag": "4.0.0", "file": "normal150.inc", "sha256": <of the .inc>, "height": 28, "data": <base64>}

and is embedded as src/Gym.Environments/Assets/mujoco/fonts/normal150.json, read by Gym.NET's MujocoOverlay. MuJoCo is
Apache-2.0 (Copyright 2021 DeepMind Technologies Limited); the Assets README records the attribution.

Usage:
    python -I tools/parity/mujoco/convert_font.py <normal150.inc> <output.json>

Get the .inc from https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/font/normal150.inc into a
folder of its own first.

Returns (exit code): 0 after writing; raises on a source that is not a MuJoCo bitmap font.
"""
import base64
import hashlib
import json
import re
import sys


def parse(text: str) -> bytes:
    """The C array's bytes.

    Args:
        text: the .inc file's text: one `const unsigned char name[] = { ... };` array.
    Returns:
        The bytes, in order.
    Raises:
        ValueError: when the text holds no such array, or a value is not a byte.
    """
    m = re.search(r"\{(.*)\}", text, re.S)
    if not m:
        raise ValueError("no C array in the font source")
    values = [int(v) for v in re.findall(r"\d+", m.group(1))]
    if any(v > 255 for v in values):
        raise ValueError("a value beyond a byte")
    return bytes(values)


def check(data: bytes) -> int:
    """Walks the font as MuJoCo's makeFont does and returns its height.

    Args:
        data: the font's bytes.
    Returns:
        The characters' common height in pixels.
    Raises:
        ValueError: when the layout is not MuJoCo's (wrong codes, heights or termination).
    """
    adr = 0
    height = None
    for code in range(32, 127):
        if data[adr] != code:
            raise ValueError(f"expected character {code} at {adr}, found {data[adr]}")
        width, h = data[adr + 1], data[adr + 2]
        if height is None:
            height = h
        elif h != height:
            raise ValueError(f"character {code} is {h} high, not {height}")
        adr += 3 + ((width - 1) // 8 + 1) * h
    if data[adr] != 123:
        raise ValueError(f"the font does not end with 123 at {adr}")
    return height


def main() -> int:
    """Converts the font.

    Returns:
        0.
    Raises:
        ValueError: when the source is not a MuJoCo bitmap font.
    """
    source, output = sys.argv[1], sys.argv[2]
    with open(source, "rb") as f:
        raw = f.read()
    data = parse(raw.decode("ascii"))
    height = check(data)
    doc = {
        "source": "https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/font/normal150.inc",
        "tag": "4.0.0",
        "file": "normal150.inc",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "license": "Apache-2.0, Copyright 2021 DeepMind Technologies Limited",
        "height": height,
        "data": base64.b64encode(data).decode("ascii"),
    }
    with open(output, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"{len(data)} bytes, height {height}, sha256 {doc['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
