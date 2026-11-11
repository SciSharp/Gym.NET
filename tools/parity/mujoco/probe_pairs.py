"""Every geom pair Gymnasium's bundled MJCF models let collide, and the MuJoCo collider each pair takes: the full collision
reach for the P9 assessment (docs/plans/MUJOCO_ASSESSMENT.md), not only the pairs random actions happen to hit.

A pair counts when MuJoCo's filters pass it: different weld bodies, contype/conaffinity bits that match, and not parent and
child (unless the model disables that filter). The collider is MuJoCo 3.15.0's table (engine_collision_driver.c,
mjCOLLISIONFUNC) for the two types.

Usage (a Python with the mujoco to measure):
    python tools/parity/mujoco/probe_pairs.py [ASSETS_DIR]   (default: refs/Gymnasium's mujoco assets)
"""
import glob
import os
import sys
from collections import Counter, defaultdict

import mujoco

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
FUNC = {  # (lower type, higher type) -> collider, from mjCOLLISIONFUNC in 3.15.0
    ("plane", "sphere"): "mjc_PlaneSphere", ("plane", "capsule"): "mjc_PlaneCapsule", ("plane", "cylinder"): "mjc_PlaneCylinder",
    ("plane", "box"): "mjc_PlaneBox", ("sphere", "sphere"): "mjc_SphereSphere", ("sphere", "capsule"): "mjc_SphereCapsule",
    ("sphere", "cylinder"): "mjc_SphereCylinder", ("sphere", "box"): "mjc_SphereBox", ("capsule", "capsule"): "mjc_CapsuleCapsule",
    ("capsule", "cylinder"): "mjc_Convex", ("capsule", "box"): "mjc_CapsuleBox", ("cylinder", "cylinder"): "mjc_Convex",
    ("cylinder", "box"): "mjc_Convex", ("box", "box"): "mjc_BoxBox",
}
ORDER = ["plane", "hfield", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh", "sdf"]
GEOM = {int(v): k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}


def collidable_pairs(m):
    """The geom-type pairs a compiled model's filters let collide.

    Args:
        m: The mujoco.MjModel.

    Returns:
        Counter: (lower type, higher type) mapped to the number of geom pairs of those types.
    """
    weld, parent = m.body_weldid, m.body_parentid
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    pairs = Counter()
    for i in range(m.ngeom):
        for j in range(i + 1, m.ngeom):
            wi, wj = weld[m.geom_bodyid[i]], weld[m.geom_bodyid[j]]
            if wi == wj:
                continue
            if not ((m.geom_contype[i] & m.geom_conaffinity[j]) or (m.geom_contype[j] & m.geom_conaffinity[i])):
                continue
            if filterparent and wi and wj and (weld[parent[wi]] == wj or weld[parent[wj]] == wi):
                continue
            a, b = sorted((GEOM[int(m.geom_type[i])], GEOM[int(m.geom_type[j])]), key=ORDER.index)
            pairs[(a, b)] += 1
    return pairs


def main(argv):
    """Prints each model's collidable pairs with their colliders, then which models reach each collider.

    Args:
        argv: The command line without the program name: an optional assets directory.

    Returns:
        int: 0.

    Raises:
        ValueError: mujoco cannot compile a model (its own error).
    """
    assets = argv[0] if argv else os.path.join(REPO, "refs", "Gymnasium", "gymnasium", "envs", "mujoco", "assets")
    reach = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(assets, "*.xml"))):
        pairs = collidable_pairs(mujoco.MjModel.from_xml_path(path))
        name = os.path.basename(path)
        print(f"{name:28} " + ", ".join(f"{a}-{b}:{n} ({FUNC.get((a, b), '?')})" for (a, b), n in sorted(pairs.items())))
        for p in pairs:
            reach[FUNC.get(p, "?")].add(name)
    print("\ncolliders reached:")
    for f, files in sorted(reach.items()):
        print(f"  {f:20} {', '.join(sorted(files))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
