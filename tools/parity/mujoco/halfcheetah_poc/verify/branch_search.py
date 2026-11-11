"""Builds the branch search of MUJOCO_ASSESSMENT.md §13.4: a copy of the POC with a probe at the first line of every run
of lines the coverage report shows unexecuted, markers at the entry of a few rarely executed paths, and conditions no
single line shows (the island warm start's reset of a dof that would change), plus a Main that plays seeded episodes of
one env through the core alone (no MuJoCo, about a million physics steps per 6 to 50 seconds) and reports, per probe, how
often and in which seeds and physics steps it fired. Seeds it finds are then recorded with oracle.py and checked, so a
branch the search reaches is proven, not just run (traces.txt's targeted lines).

Insertion rules, per run start: a member signature ending in "{" gets the probe after it; a line after one that does
not end a statement or a block (an if/for header, the last line of a multi-line condition, else, a case label) is an
embedded statement and is wrapped as "{ probe; statement }"; any other line gets the probe on a line before it. Lines
the rules cannot handle (else branches, switch arms, expression-bodied members, multi-line embedded statements) are
listed and skipped. A probe must change no bit: build the copy as the checker ("checker" below) and run it over the
trace set before trusting a search. (A first version put probes before statements embedded under multi-line conditions,
which made those statements unconditional; the checker run is what catches that.)

Usage:
    python branch_search.py <cobertura.xml> <poc folder> <out folder> [checker]
    dotnet build <out folder> -c Release
    <out folder>/bin/Release/net10.0/HalfCheetahPoc.exe <trace folder> <first seed> <end seed>
The trace folder supplies the model (model.json) and the run (run.json: the id, flip, termination, info keys); the
seeds are played as oracle.py plays them. "checker" builds the probed copy as the checker instead (Program.Main).

Args (positional): the cobertura XML (cs_coverage.py's input), the POC folder, the output folder (replaced), optionally
"checker".
Returns (stdout): the skipped lines and the number of probes.
Raises:
    AssertionError: a marker's or a custom probe's anchor is not found exactly once in the source.
"""
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cs_coverage  # noqa: E402  (the same reading of the report)

FILES = ["CheetahCore.cs", "Kernels.cs"]

# executed paths worth locating too: (file, anchor that ends a line, label); the probe goes on the line after the anchor
MARKERS = [
    ("Kernels.cs", "private static int Polytope3(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {", "marker Polytope3"),
    ("Kernels.cs", "private static int Polytope4(CcdPolytope pt, CcdStatus status, CcdObject obj1, CcdObject obj2) {", "marker Polytope4"),
    ("Kernels.cs", "private static int Epa(CcdStatus status, CcdPolytope pt, CcdObject obj1, CcdObject obj2) {", "marker Epa"),
    ("Kernels.cs", "private static int GjkIntersect(CcdStatus status, CcdObject obj1, CcdObject obj2) {", "marker GjkIntersect"),
]

# conditions no single line shows: (file, anchor, code inserted after the anchor, labels the code uses); the code only
# reads state
CUSTOM = [
    ("CheetahCore.cs",
     "        if (costWarm > costSmooth)\n            Array.Copy(qacc_smooth, qacc, nv);\n",
     "        if (costWarm <= costSmooth && nidof < nv) {\n"
     "            Probe.Hit(\"custom: warm start kept while some dofs are in no island\");\n"
     "            for (int q = nidof; q < nv; q++) {\n"
     "                int dq = mapIdof2dof[q];\n"
     "                if (BitConverter.DoubleToInt64Bits(qacc[dq]) != BitConverter.DoubleToInt64Bits(qacc_smooth[dq])) {\n"
     "                    Probe.Hit(\"custom: warm start kept and a free dof's warm start differs from qacc_smooth\");\n"
     "                    break;\n"
     "                }\n"
     "            }\n"
     "        }\n",
     ["custom: warm start kept while some dofs are in no island",
      "custom: warm start kept and a free dof's warm start differs from qacc_smooth"]),
    # two runs the insertion rules cannot probe (statements under an unbraced header), probed by their conditions
    ("Kernels.cs",
     "            outward[k] = verts[v1 * VertexSize + k] - pt.Center[k];\n",
     "        if (Dot3(projection, outward) < 0)\n"
     "            Probe.Hit(\"custom: AttachFace turns a face's projection outward\");\n",
     ["custom: AttachFace turns a face's projection outward"]),
    ("CheetahCore.cs",
     "        else if (root2 < root1)\n            treeParent[root1] = root2;\n",
     "        if (root2 < root1)\n"
     "            Probe.Hit(\"custom: the union-find links the first root under the second\");\n",
     ["custom: the union-find links the first root under the second"]),
]

PROBE = '''using System;
using System.Collections.Generic;
using System.Linq;

namespace MujocoPoc;

/// <summary>The search's probes: per label, the hit count, the episodes and the first (seed, physics step) pairs.</summary>
internal static class Probe {
    /// <summary>The episode's seed, set by the search before each episode.</summary>
    public static int Seed;

    /// <summary>The episode's physics step, set by the search before each step.</summary>
    public static int Step;

    private static readonly Dictionary<string, (long Count, HashSet<int> Seeds, List<string> First)> hits = new();

    /// <summary>Records one hit of a probe.</summary>
    /// <param name="label">The probe's label (file, lines and method, or a marker's name).</param>
    public static void Hit(string label) {
        if (!hits.TryGetValue(label, out var h))
            h = (0, new HashSet<int>(), new List<string>());
        if (h.First.Count < 6 && !h.Seeds.Contains(Seed))
            h.First.Add($"{Seed}@{Step}");
        h.Seeds.Add(Seed);
        hits[label] = (h.Count + 1, h.Seeds, h.First);
    }

    /// <summary>Prints every probe that fired, then the ones that never did.</summary>
    /// <param name="all">Every probe's label.</param>
    /// <param name="physics">The physics steps played.</param>
    /// <param name="episodes">The episodes played.</param>
    public static void Report(IEnumerable<string> all, long physics, int episodes) {
        Console.WriteLine($"searched {episodes} episodes, {physics} physics steps");
        foreach (var (label, h) in hits.OrderBy(p => p.Key))
            Console.WriteLine($"HIT {label}: {h.Count} times in {h.Seeds.Count} episodes; first {string.Join(" ", h.First)}");
        foreach (string label in all.Where(l => !hits.ContainsKey(l)))
            Console.WriteLine($"never {label}");
    }
}
'''

SEARCH = '''using System;
using System.IO;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace MujocoPoc;

/// <summary>Plays seeded episodes through the core alone, as oracle.py's Gymnasium runs play them, with the probes armed.</summary>
internal static class Search {
    /// <summary>Plays the seeds of one env and prints the probes' report.</summary>
    /// <param name="args">The trace folder (model.json, run.json), the first seed, the end seed (exclusive).</param>
    /// <returns>0.</returns>
    /// <exception cref="InvalidOperationException">The run's id has no env rules.</exception>
    private static int Main(string[] args) {
        string dir = args[0];
        int from = int.Parse(args[1]), to = int.Parse(args[2]);
        var model = CheetahModel.Load(Path.Combine(dir, "model.json"));
        var run = RunInfo.Load(Path.Combine(dir, "run.json"));
        EnvRules rules = EnvRules.TryFor(model, run) ?? throw new InvalidOperationException("no rules for " + run.Env);
        long physics = 0;
        for (int seed = from; seed < to; seed++) {
            Probe.Seed = seed;
            var core = new CheetahCore(model);
            rules.Reset(core, seed, run.Flip);
            // the action space's own stream, as Program.SampleActions draws it, one action per env step
            var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
            using NDArray low = np.array(rules.ActionLow);
            using NDArray high = np.array(rules.ActionHigh);
            for (int t = 0; t < run.MaxEpisodeSteps; t++) {
                using NDArray drawn = rng.uniform(low, high, new Shape(rules.ActionLow.Length));
                double[] action = drawn.ToArray<double>().Select(v => (double)(float)v).ToArray();
                rules.Before(core, action);
                Array.Copy(action, core.ctrl, model.nu);
                for (int k = 0; k < model.frame_skip; k++) {
                    Probe.Step = t * model.frame_skip + k;
                    core.Step();
                    physics++;
                }
                core.RnePostConstraint();
                (double[] _, double[] row) = rules.After(core, action, t, run.InfoKeys);
                // row: reward, terminated, truncated, info...
                if (row[1] != 0 || row[2] != 0)
                    break;
            }
        }
        Probe.Report(ProbeLabels.All, physics, to - from);
        return 0;
    }
}
'''


def insert_probes(path, runs, name):
    """Inserts a probe at the start of each unexecuted run of one file, by the rules in the module's docstring.

    Args:
        path: the copied source file, rewritten in place.
        runs: the file's unexecuted runs, [(first line, last line, method)].
        name: the file name, for the labels.
    Returns:
        The labels of the probes inserted.
    """
    lines = open(path, encoding="utf-8").read().split("\n")
    edits, labels = [], []
    for start, end, method in runs:
        i = start - 1
        s = lines[i].strip()
        label = f"{name}:{start}-{end} {method}"
        j = i - 1
        while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith("//")):
            j -= 1
        prev = lines[j].strip()
        # a line is in a block only after a line that ends a statement or opens/closes a block; after anything else it
        # is an embedded statement, and a probe put before it would become the header's body
        embedded = not (prev.endswith(";") or prev.endswith("{") or prev.endswith("}"))
        if re.match(r"^(private|public|internal|protected)\b", s):
            if not s.endswith("{"):
                print(f"skipped (expression-bodied member): {label}: {s}")
                continue
            edits.append((i, "after", label))
        elif s.startswith("else") or "=>" in s.split("(")[0]:
            print(f"skipped (else or switch arm): {label}: {s}")
            continue
        elif embedded:
            if not s.endswith(";"):
                print(f"skipped (multi-line embedded statement): {label}: {s}")
                continue
            edits.append((i, "wrap", label))
        else:
            edits.append((i, "before", label))
        labels.append(label)
    # from the bottom up, so earlier indices stay valid
    for i, mode, label in sorted(edits, reverse=True):
        indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
        call = f'Probe.Hit("{label}");'
        if mode == "after":
            lines.insert(i + 1, indent + "    " + call)
        elif mode == "wrap":
            lines[i] = indent + "{ " + call + " " + lines[i].strip() + " }"
        else:
            lines.insert(i, indent + call)
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    return labels


def insert_after(path, anchor, code):
    """Inserts code after a unique anchor.

    Args:
        path: the source file, rewritten in place.
        anchor: the exact text to find once.
        code: the text inserted right after it.
    Raises:
        AssertionError: the anchor is not found exactly once.
    """
    src = open(path, encoding="utf-8").read()
    assert src.count(anchor) == 1, (path, anchor, src.count(anchor))
    open(path, "w", encoding="utf-8").write(src.replace(anchor, anchor + code))


def main():
    """Copies the project, inserts every probe, writes Probe.cs, Search.cs and the label list, and sets the entry point."""
    report, poc, out = sys.argv[1], sys.argv[2], sys.argv[3]
    checker = len(sys.argv) > 4 and sys.argv[4] == "checker"
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for f in os.listdir(poc):
        if f.endswith(".cs") or f.endswith(".csproj"):
            shutil.copy(os.path.join(poc, f), out)
    hits, method_of = cs_coverage.load(report)
    labels = []
    for name in FILES:
        runs = [(a, b, method_of[name].get(a, "?")) for a, b in cs_coverage.unexecuted_runs(hits[name])]
        labels += insert_probes(os.path.join(out, name), runs, name)
    for name, anchor, label in MARKERS:
        insert_after(os.path.join(out, name), anchor, f'\n        Probe.Hit("{label}");')
        labels.append(label)
    for name, anchor, code, custom_labels in CUSTOM:
        insert_after(os.path.join(out, name), anchor, code)
        labels += custom_labels
    with open(os.path.join(out, "Probe.cs"), "w", encoding="utf-8") as fh:
        fh.write(PROBE)
    with open(os.path.join(out, "Search.cs"), "w", encoding="utf-8") as fh:
        fh.write(SEARCH)
    with open(os.path.join(out, "ProbeLabels.cs"), "w", encoding="utf-8") as fh:
        body = ",\n        ".join('"' + l.replace('"', '\\"') + '"' for l in labels)
        fh.write("namespace MujocoPoc;\n\n/// <summary>Every probe's label, so the report can list the ones that never fired.</summary>\n"
                 "internal static class ProbeLabels {\n    /// <summary>The labels.</summary>\n    public static readonly string[] All = [\n        "
                 + body + "\n    ];\n}\n")
    # two Mains: the project names its entry point (CS7022 warns that the other is ignored)
    startup = "MujocoPoc.Program" if checker else "MujocoPoc.Search"
    proj = os.path.join(out, "HalfCheetahPoc.csproj")
    src = open(proj, encoding="utf-8").read()
    src = src.replace("<RootNamespace>MujocoPoc</RootNamespace>",
                      f"<RootNamespace>MujocoPoc</RootNamespace>\n    <StartupObject>{startup}</StartupObject>\n    <NoWarn>CS7022</NoWarn>")
    open(proj, "w", encoding="utf-8").write(src)
    print(f"{len(labels)} probes; entry point {startup}")


if __name__ == "__main__":
    main()
