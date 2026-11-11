"""Are two Windows PE files the same build apart from their Authenticode signatures (docs/plans/MUJOCO_ASSESSMENT.md §9)?
Strips each file's certificate table, zeroes its checksum and certificate-directory entry, compares the rest, and
reports each file's link timestamp and linker version. Used to compare the mujoco.dll of DeepMind's signed release with
the one inside the `mujoco` Python wheel.

Usage (any Python 3):
    python tools/parity/mujoco/pe_compare.py RELEASE.dll WHEEL.dll
"""
import hashlib
import struct
import sys


def normalized(path):
    """A PE file's bytes without its signature, checksum and certificate-directory entry.

    Args:
        path: The PE file.

    Returns:
        tuple: (the normalized bytes, link timestamp, certificate table size, linker version as (major, minor)).

    Raises:
        OSError: The file cannot be read.
        struct.error: The file is too short to be a PE file.
    """
    with open(path, "rb") as f:
        data = bytearray(f.read())
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    timestamp = struct.unpack_from("<I", data, pe + 8)[0]
    optional = pe + 24
    magic = struct.unpack_from("<H", data, optional)[0]
    linker = (data[optional + 2], data[optional + 3])
    directories = optional + (112 if magic == 0x20B else 96)
    # Data directory 4 is the certificate table; unlike the others, its address is a file offset.
    cert_offset, cert_size = struct.unpack_from("<II", data, directories + 8 * 4)
    struct.pack_into("<I", data, optional + 64, 0)
    struct.pack_into("<II", data, directories + 8 * 4, 0, 0)
    if cert_size:
        del data[cert_offset:cert_offset + cert_size]
    return bytes(data), timestamp, cert_size, linker


def main(argv):
    """Prints both files' facts and whether they are the same build.

    Args:
        argv: The command line without the program name: two PE files.

    Returns:
        int: 0 when the builds are identical, 1 when they differ, 2 for a malformed command line.

    Raises:
        OSError: A file cannot be read.
        struct.error: A file is not a PE file.
    """
    if len(argv) != 2:
        print(__doc__)
        return 2
    results = [normalized(p) for p in argv]
    for path, (blob, timestamp, cert, linker) in zip(argv, results):
        print(f"{path}\n  {len(blob)} bytes unsigned, certificate {cert} bytes, link timestamp {timestamp}, "
              f"linker {linker[0]}.{linker[1]}, sha256 {hashlib.sha256(blob).hexdigest()}")
    same = results[0][0] == results[1][0]
    print("identical without signatures:", same)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
