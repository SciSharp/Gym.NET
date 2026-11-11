"""Summarizes a dotnet-coverage cobertura report of the checker's run (MUJOCO_ASSESSMENT.md §13.4): per source file, the
lines executed out of the lines the compiler marked as sequence points, then every unexecuted run of lines with its
method and first source line. The runs it lists are what branch_search.py probes.

A line counts once per file even when several classes (nested types, closures, records) list it: executed when any of
them hit it. A run of unexecuted lines continues across lines the compiler did not mark (blank lines, braces, comments).

The report comes from a Debug build of the checker over the trace set, for example:
    dotnet tool install dotnet-coverage --tool-path <tools>
    dotnet build tools/parity/mujoco/halfcheetah_poc -c Debug
    <tools>/dotnet-coverage collect -f cobertura -o <report.xml> -- \\
        tools/parity/mujoco/halfcheetah_poc/bin/Debug/net10.0/HalfCheetahPoc.exe <trace folders...>

Args (positional): the cobertura XML; the POC folder (to print the source lines); optionally the file names to list
(default: every file).
Returns (stdout): the per-file totals, then the unexecuted runs of the listed files.
Raises:
    FileNotFoundError: the report is missing.
    xml.etree.ElementTree.ParseError: the report is not XML.
"""
import collections
import os
import sys
import xml.etree.ElementTree as ET


def load(report):
    """Reads the line hits of a cobertura report.

    Args:
        report: the XML file.
    Returns:
        (hits, method_of): {file name: {line: hits}} with the largest hit count per line, and {file name: {line: "Class.Method"}}.
    """
    root = ET.parse(report).getroot()
    hits = collections.defaultdict(dict)
    method_of = collections.defaultdict(dict)
    for cls in root.iter("class"):
        name = os.path.basename(cls.get("filename"))
        for method in cls.iter("method"):
            for line in method.iter("line"):
                method_of[name][int(line.get("number"))] = f"{cls.get('name').split('.')[-1]}.{method.get('name')}"
        for line in cls.iter("line"):
            n = int(line.get("number"))
            hits[name][n] = max(hits[name].get(n, 0), int(line.get("hits")))
    return hits, method_of


def unexecuted_runs(lines):
    """Groups a file's unexecuted lines into runs.

    Args:
        lines: {line: hits} of one file.
    Returns:
        A list of (first line, last line) runs, in order; a run continues across lines missing from the report.
    """
    missed = sorted(n for n, h in lines.items() if h == 0)
    runs = []
    if not missed:
        return runs
    start = prev = missed[0]
    for n in missed[1:] + [None]:
        if n is not None and all(k not in lines for k in range(prev + 1, n)):
            prev = n
            continue
        runs.append((start, prev))
        if n is not None:
            start = prev = n
    return runs


def main():
    """Prints the per-file totals, then the unexecuted runs of the requested files."""
    report, folder = sys.argv[1], sys.argv[2]
    wanted = set(sys.argv[3:])
    hits, method_of = load(report)
    for name in sorted(hits):
        done = sum(1 for h in hits[name].values() if h > 0)
        print(f"{name}: {done} of {len(hits[name])} lines executed")
    for name in sorted(hits):
        if wanted and name not in wanted:
            continue
        runs = unexecuted_runs(hits[name])
        if not runs:
            continue
        path = os.path.join(folder, name)
        source = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
        print(f"\n== {name}: {sum(1 for h in hits[name].values() if h == 0)} lines never executed")
        for a, b in runs:
            text = source[a - 1].strip() if a - 1 < len(source) else ""
            count = sum(1 for k in range(a, b + 1) if hits[name].get(k) == 0)
            print(f"  {a}-{b} ({count}) {method_of[name].get(a, '?')}: {text[:110]}")


if __name__ == "__main__":
    main()
