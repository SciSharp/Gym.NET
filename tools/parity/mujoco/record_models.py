"""Records Gymnasium's 14 bundled MuJoCo models as compiled by the pinned MuJoCo (DR-63, DR-64): each XML in
refs/Gymnasium/gymnasium/envs/mujoco/assets is compiled once with mujoco.MjModel.from_xml_path and written as
src/Gym.Environments/Assets/mujoco/<model>.json, with its textures as PNG files under Assets/mujoco/textures/ (named
by the SHA-256 of their pixels, so a texture several models share is stored once).

Gym.NET never compiles MJCF: the envs read these files (the Gym.MuJoCo loader), so the numbers are MuJoCo's own,
bit for bit, as CarRacing's score glyphs are pygame's own (DR-48). Re-record after a re-pin of MuJoCo or Gymnasium;
the files change only where the compiler's output changed.

What a model file holds (format 1):
    format, mujoco, gymnasium     the format version, the MuJoCo version that compiled it, Gymnasium's commit.
    source, source_sha256         the XML's file name and the SHA-256 of its bytes.
    sizes                         every n* count the loader or Gymnasium reads (nq, nv, nu, na, nbody, ...).
    opt, stat, vis                mjOption's fields, mjStatistic's, and mjVisual's six groups (global_, quality,
                                  headlight, map, scale, rgba), each value a number or a list.
    names                         per element kind (body, jnt, geom, site, cam, light, actuator, tendon, mat, tex),
                                  the names in index order ("" for an unnamed element).
    textures                      per texture: name, type, width, height, nchannel, the PNG's resource path and the
                                  SHA-256 of its raw bytes (tex_data's slice), so a decoded PNG can be checked
                                  against MuJoCo's own pixels.
    arrays                        the model's arrays, flattened (row-major), by MuJoCo's name: the physics fields the
                                  core reads or validates, and every geom_*, site_*, mat_*, tex_* (but tex_data),
                                  light_* and cam_* array the renderer may read.

Doubles are written by Python's repr, which round-trips exactly; .NET's parser reads them back to the same bits.

Usage (a Python with numpy and the pinned mujoco; -I keeps the script's folder off sys.path):
    python -I tools/parity/mujoco/record_models.py [--check]

Args (command line):
    --check     compare with the committed files instead of writing them; exit 1 on any difference.
Returns (exit code): 0 when every file was written (or, with --check, matched), 1 otherwise.
Raises:
    FileNotFoundError: refs/Gymnasium's assets folder is missing (submodule not checked out).
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import zlib

import mujoco
import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SOURCE = os.path.join(REPO, "refs", "Gymnasium", "gymnasium", "envs", "mujoco", "assets")
OUT = os.path.join(REPO, "src", "Gym.Environments", "Assets", "mujoco")
FORMAT = 1

# the counts the loader and Gymnasium read; any missing from a MuJoCo version is skipped
SIZES = ["nq", "nv", "nu", "na", "nbody", "njnt", "ngeom", "nsite", "ncam", "nlight", "nmat", "ntex", "nmesh", "nhfield",
         "nskin", "ntendon", "nwrap", "nbvh", "nbvhstatic", "nM", "nC", "nD", "nB", "nJten", "ntree", "nexclude", "npair",
         "neq", "nflex", "nplugin", "nmocap", "nsensor", "nkey", "nuser_body", "nuser_jnt", "nuser_geom", "nuser_site",
         "nuser_cam", "nuser_tendon", "nuser_actuator", "nuser_sensor", "nnames", "nnumeric", "ntext", "ntuple"]

# the physics arrays the core reads or the loader validates (oracle.py's list), in MuJoCo's names
PHYSICS = ["body_parentid", "body_rootid", "body_weldid", "body_jntnum", "body_jntadr", "body_dofnum", "body_dofadr",
           "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_subtreemass", "body_inertia",
           "body_invweight0", "body_sameframe", "body_simple", "body_gravcomp", "body_geomadr", "body_geomnum",
           "body_mocapid", "body_contype", "body_conaffinity", "body_treeid",
           "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_limited", "jnt_pos", "jnt_axis",
           "jnt_stiffness", "jnt_stiffnesspoly", "jnt_range", "jnt_margin", "jnt_solref", "jnt_solimp",
           "jnt_actuatorid", "jnt_actfrclimited", "jnt_group",
           "dof_bodyid", "dof_jntid", "dof_parentid", "dof_treeid", "dof_Madr", "dof_simplenum", "dof_damping",
           "dof_dampingpoly", "dof_armature", "dof_frictionloss", "dof_M0", "dof_invweight0", "dof_solref",
           "dof_solimp",
           "geom_fluid", "geom_adhesion",
           "actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trnid",
           "actuator_ctrllimited", "actuator_forcelimited", "actuator_ctrlrange", "actuator_forcerange",
           "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_damping", "actuator_dampingpoly",
           "actuator_armature", "actuator_actadr", "actuator_actnum", "actuator_ctrladr", "actuator_outadr",
           "actuator_delay", "actuator_ctrlnum", "actuator_group",
           "qpos0", "qpos_spring", "M_rownnz", "M_rowadr", "M_colind", "mapM2M", "mapM2D", "D_rownnz", "D_rowadr",
           "D_colind", "D_diag", "B_rownnz", "B_rowadr", "B_colind", "tree_bodyadr", "tree_bodynum", "tree_dofadr",
           "tree_dofnum",
           "tendon_adr", "tendon_num", "tendon_limited", "tendon_actfrclimited", "tendon_range", "tendon_stiffness",
           "tendon_stiffnesspoly", "tendon_damping", "tendon_dampingpoly", "tendon_armature", "tendon_frictionloss",
           "tendon_lengthspring", "tendon_margin", "tendon_actuatorid",
           "wrap_type", "wrap_objid", "wrap_prm", "ten_J_rownnz", "ten_J_rowadr", "ten_J_colind"]

# every array of these element kinds is kept for the renderer (tex_data goes to PNG instead)
RENDER_PREFIXES = ["geom_", "site_", "mat_", "tex_", "light_", "cam_"]

NAME_KINDS = {"body": ("nbody", "body"), "jnt": ("njnt", "joint"), "geom": ("ngeom", "geom"), "site": ("nsite", "site"),
              "cam": ("ncam", "camera"), "light": ("nlight", "light"), "actuator": ("nu", "actuator"),
              "tendon": ("ntendon", "tendon"), "mat": ("nmat", "material"), "tex": ("ntex", "texture")}


def plain(value):
    """Turns a MuJoCo or NumPy value into JSON's types.

    Args:
        value: a scalar or an array of MuJoCo's.
    Returns:
        A Python int, float or bool, or a flat list of them (row-major).
    """
    a = np.asarray(value)
    if a.ndim == 0:
        return a.item()
    return a.ravel().tolist()


def struct_fields(obj):
    """Reads a MuJoCo struct's public numeric fields (mjOption, mjStatistic, an mjVisual group).

    Args:
        obj: the struct.
    Returns:
        {field: number or list}, sorted by name; methods and non-numeric fields are left out.
    """
    out = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        value = getattr(obj, name)
        if callable(value):
            continue
        try:
            a = np.asarray(value)
        except Exception:  # a field that is not array-like (none expected) is skipped rather than guessed
            continue
        if a.dtype.kind in "biuf":
            out[name] = plain(a)
    return out


def png(width, height, channels, data):
    """Encodes raw 8-bit pixels as a PNG, deterministically (no timestamps, fixed zlib level).

    Args:
        width: the image's width.
        height: its height.
        channels: 3 (RGB) or 4 (RGBA).
        data: width*height*channels bytes, row-major.
    Returns:
        The PNG file's bytes.
    Raises:
        ValueError: a channel count other than 3 or 4, or a wrong data length.
    """
    if channels not in (3, 4):
        raise ValueError(f"PNG with {channels} channels")
    if len(data) != width * height * channels:
        raise ValueError("pixel data length")
    stride = width * channels
    # each row is prefixed with filter type 0 (none), so the encoding depends only on the pixels and zlib
    raw = b"".join(b"\x00" + bytes(data[r * stride:(r + 1) * stride]) for r in range(height))

    def chunk(kind, body):
        """One PNG chunk: length, type, body and the CRC of type and body."""
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2 if channels == 3 else 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def gymnasium_commit():
    """Reads refs/Gymnasium's HEAD.

    Returns:
        The commit hash.
    Raises:
        subprocess.CalledProcessError: git failed (not a checkout).
    """
    return subprocess.run(["git", "-C", os.path.join(REPO, "refs", "Gymnasium"), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def record(xml_name, commit):
    """Compiles one XML and builds its file contents.

    Args:
        xml_name: the file name in Gymnasium's assets folder.
        commit: Gymnasium's commit, recorded in the file.
    Returns:
        (model JSON text, {texture file name: PNG bytes}).
    """
    path = os.path.join(SOURCE, xml_name)
    with open(path, "rb") as f:
        source = f.read()
    m = mujoco.MjModel.from_xml_path(path)
    doc = {"format": FORMAT, "mujoco": mujoco.__version__, "gymnasium": commit, "source": xml_name,
           "source_sha256": hashlib.sha256(source).hexdigest()}
    doc["sizes"] = {k: int(getattr(m, k)) for k in SIZES if hasattr(m, k)}
    doc["opt"] = struct_fields(m.opt)
    doc["stat"] = struct_fields(m.stat)
    doc["vis"] = {g: struct_fields(getattr(m.vis, g)) for g in ["global_", "quality", "headlight", "map", "scale", "rgba"]}
    names = {}
    for kind, (count, accessor) in NAME_KINDS.items():
        n = int(getattr(m, count))
        names[kind] = [getattr(m, accessor)(i).name for i in range(n)]
    doc["names"] = names
    textures, files = [], {}
    for t in range(m.ntex):
        w, h, c = int(m.tex_width[t]), int(m.tex_height[t]), int(m.tex_nchannel[t])
        adr = int(m.tex_adr[t])
        pixels = bytes(np.asarray(m.tex_data[adr:adr + w * h * c], dtype=np.uint8))
        digest = hashlib.sha256(pixels).hexdigest()
        file_name = f"{digest[:16]}.png"
        files[file_name] = png(w, h, c, pixels)
        textures.append({"name": names["tex"][t], "type": int(m.tex_type[t]), "width": w, "height": h, "nchannel": c,
                         "png": f"textures/{file_name}", "sha256": digest})
    doc["textures"] = textures
    arrays = {}
    for name in PHYSICS:
        if hasattr(m, name):
            arrays[name] = plain(getattr(m, name))
    for name in sorted(dir(m)):
        if name == "tex_data" or not any(name.startswith(p) for p in RENDER_PREFIXES) or name in arrays:
            continue
        value = getattr(m, name)
        if callable(value):
            continue
        a = np.asarray(value)
        if a.dtype.kind in "biuf":
            arrays[name] = plain(a)
    doc["arrays"] = arrays
    # one array per line keeps the files reviewable in diffs; the order is the dict's (sizes first, arrays last)
    text = json.dumps(doc, indent=0, separators=(",", ":")) + "\n"
    return text, files


def main():
    """Records (or checks) every model.

    Returns:
        The exit code.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    commit = gymnasium_commit()
    xmls = sorted(f for f in os.listdir(SOURCE) if f.endswith(".xml"))
    bad = 0
    os.makedirs(os.path.join(OUT, "textures"), exist_ok=True)
    for xml_name in xmls:
        text, files = record(xml_name, commit)
        targets = {os.path.join(OUT, xml_name[:-4] + ".json"): text.encode("utf-8")}
        for file_name, data in files.items():
            targets[os.path.join(OUT, "textures", file_name)] = data
        for target, data in targets.items():
            old = open(target, "rb").read() if os.path.exists(target) else None
            if args.check:
                if old != data:
                    print(f"differs: {os.path.relpath(target, REPO)}")
                    bad += 1
            elif old != data:
                with open(target, "wb") as f:
                    f.write(data)
                print(f"wrote {os.path.relpath(target, REPO)} ({len(data):,} bytes)")
    print(f"{len(xmls)} models, mujoco {mujoco.__version__}, gymnasium {commit[:8]}{': ' + str(bad) + ' differ' if bad else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
