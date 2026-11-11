"""The MJCF surface Gymnasium's bundled models use, which bounds the compiler subset P9 ports
(docs/plans/MUJOCO_ASSESSMENT.md): every element with the attributes it is given anywhere, the files that use it, and the
texture kinds (type/builtin).

Usage (any Python 3):
    python tools/parity/mujoco/mjcf_census.py [ASSETS_DIR]   (default: refs/Gymnasium's mujoco assets)
"""
import glob
import os
import sys
import xml.etree.ElementTree as ElementTree
from collections import defaultdict

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def main(argv):
    """Prints the census.

    Args:
        argv: The command line without the program name: an optional assets directory.

    Returns:
        int: 0.

    Raises:
        xml.etree.ElementTree.ParseError: A model is not XML.
    """
    assets = argv[0] if argv else os.path.join(REPO, "refs", "Gymnasium", "gymnasium", "envs", "mujoco", "assets")
    attrs = defaultdict(set)
    files = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(assets, "*.xml"))):
        for el in ElementTree.parse(path).getroot().iter():
            attrs[el.tag].update(el.attrib)
            files[el.tag].add(os.path.basename(path))
            if el.tag == "texture":
                attrs["texture kinds"].add(f"{el.get('type', '2d')}/{el.get('builtin', 'file')}")
    for tag in sorted(attrs):
        print(f"{tag:14} ({len(files.get(tag, ())):2} files) {', '.join(sorted(attrs[tag]))}")
    pairs = sum(len(v) for k, v in attrs.items() if k != "texture kinds")
    print(f"\n{len(attrs) - 1} elements, {pairs} element-attribute pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
