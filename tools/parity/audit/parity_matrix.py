"""Gym.NET's parity audit from its own test results: every test of both suites, by env and by criterion.

Reads the TRX files of a unit run (tests/Gym.Tests) and a parity run (tests/Gym.Tests.Parity), both run WITHOUT the
OpenBugs filter so the open gaps are counted too, classifies each result (each data row counts) by the env family and
registered id it exercises and by the criterion it checks, and writes the audit's tables as Markdown. The defect ids of
open tests come from the [OpenBugs(...)] attributes in the test sources, since TRX keeps only the category.

Usage:
    dotnet test tests/Gym.Tests/Gym.Tests.csproj -c Release -f net10.0-windows \
        --filter "FullyQualifiedName!~AvaloniaEnv&FullyQualifiedName!~WinFormEnv" \
        --logger "trx;LogFileName=unit.trx" --results-directory OUT
    dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -c Release -f net10.0 \
        --logger "trx;LogFileName=parity.trx" --results-directory OUT
    python tools/parity/audit/parity_matrix.py OUT/unit.trx OUT/parity.trx [--json OUT/audit.json] > tables.md

The rules below are the audit's definitions: change them and the tables change with them. Anything they cannot place is
listed under "Unclassified" in the output, never dropped.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ElementTree
from collections import defaultdict

NS = {"t": "http://microsoft.com/schemas/VisualStudio/TeamTest/2010"}
REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# Each env family (one Gym.NET class), its registered ids in registration order, and Gymnasium's env package.
FAMILIES = [
    ("CartPole", ["CartPole-v0", "CartPole-v1"], "classic_control"),
    ("MountainCar", ["MountainCar-v0", "MountainCarContinuous-v0"], "classic_control"),
    ("Pendulum", ["Pendulum-v1"], "classic_control"),
    ("Acrobot", ["Acrobot-v1"], "classic_control"),
    ("LunarLander", ["LunarLander-v3", "LunarLanderContinuous-v3"], "box2d"),
    ("BipedalWalker", ["BipedalWalker-v3", "BipedalWalkerHardcore-v3"], "box2d"),
    ("CarRacing", ["CarRacing-v3"], "box2d"),
    ("FrozenLake", ["FrozenLake-v1", "FrozenLake8x8-v1"], "toy_text"),
    ("CliffWalking", ["CliffWalking-v1", "CliffWalkingSlippery-v1"], "toy_text"),
    ("Taxi", ["Taxi-v4"], "toy_text"),
    ("Blackjack", ["Blackjack-v1"], "toy_text"),
]
FAMILY_OF_ID = {i: name for name, ids, _ in FAMILIES for i in ids}
ID_PATTERN = re.compile(r"\b(" + "|".join(sorted((re.escape(i) for i in FAMILY_OF_ID), key=len, reverse=True)) + r")\b")
# A family named in a class, method or data row ("Cartpole" in the 0.2-era unit class names too).
FAMILY_PATTERN = re.compile(r"(CartPole|MountainCar|Pendulum|Acrobot|LunarLander|BipedalWalker|CarRacing|FrozenLake|CliffWalking|Taxi|Blackjack)",
                            re.IGNORECASE)
# MuJoCo's envs (P9, postponed), bounded by non-letters so "_Ant" and "AntEnvV4" match and "Constant" does not.
MUJOCO_PATTERN = re.compile(r"(?<![A-Za-z])(HumanoidStandup|Humanoid|InvertedDoublePendulum|InvertedPendulum|HalfCheetah|Hopper|Pusher|Reacher|"
                            r"Swimmer|Walker2d|Ant)(?![a-z])(-v\d+)?")

# The own-suite criteria (a family's file under Tests/Envs), first match wins, matched against the method name.
OWN_SUITE_RULES = [
    ("harness", r"^Reference_"),
    ("frames", r"Frame|Render|ScoreFont|ScoreBox"),
    ("text", r"Ansi"),
    ("api", r"Space|Metadata|Repr|Desc|ConstructorChecks|Class_Refuses"),
    ("states", r"Inject|Malformed"),
    # Taxi's encode/decode/action_mask are the env's own functions, not actions it is given.
    ("dynamics", r"EncodeDecode"),
    ("actions", r"Action|StepBeforeReset|StepAfterTermination|TorqueDtypes|DecodeBeyond|FloatMove|OrderErrors"),
    ("reset", r"Reset|Track|World|Terrain|DomainRandomize"),
    ("make", r"MakeStack"),
    ("dynamics", r"."),
]

# Cells whose display name holds no id: the criterion and the ids their bodies run (read from the test sources).
OVERRIDES = {
    ("FrozenLakeLiveParityTests", "GymNet_MakeStack_MatchesGymnasiumThroughTruncation"): ("make", ["FrozenLake-v1", "FrozenLake8x8-v1"]),
    ("CliffWalkingLiveParityTests", "GymNet_MakeStack_MatchesGymnasium"): ("make", ["CliffWalking-v1", "CliffWalkingSlippery-v1"]),
    ("CoreLiveParityTests", "MadeStack_MatchesGymnasiumsMake_ThroughTruncation"): ("make", ["CartPole-v1"]),
    ("RegistrationLiveParityTests", "Make_CartPole_StackAndSpecs_MatchGymnasium"): ("registration", ["CartPole-v1"]),
    ("RegistrationLiveParityTests", "Spec_MountainCarContinuousKwargs_CarryContinuous_AsDr32Accepts"): ("registration", ["MountainCarContinuous-v0"]),
    ("RegistrationLiveParityTests", "Spec_TaxiRewardThreshold_IsFloat64"): ("registration", ["Taxi-v4"]),
}
# A family's id for an id-level cell that names neither an id nor a variant: the id its suite runs by default.
PRIMARY_ID = {"CartPole": "CartPole-v1", "MountainCar": "MountainCar-v0", "LunarLander": "LunarLander-v3",
              "BipedalWalker": "BipedalWalker-v3", "FrozenLake": "FrozenLake-v1", "CliffWalking": "CliffWalking-v1"}
# A variant named in a method ("...MountainCarContinuous...") picks its id.
VARIANT_STEMS = [("MountainCarContinuous", "MountainCarContinuous-v0"), ("LunarLanderContinuous", "LunarLanderContinuous-v3"),
                 ("BipedalWalkerHardcore", "BipedalWalkerHardcore-v3"), ("FrozenLake8x8", "FrozenLake8x8-v1"),
                 ("CliffWalkingSlippery", "CliffWalkingSlippery-v1"), ("CartPoleV0", "CartPole-v0")]
# Facts about the envs, for the cells no test can fill: the families with an ansi mode, and those whose state is a
# Box2D world built by reset (nothing to inject).
ANSI_FAMILIES = {"FrozenLake", "CliffWalking", "Taxi"}
BOX2D_FAMILIES = {"LunarLander", "BipedalWalker", "CarRacing"}

# The columns of the audit's two env tables, in order, with their headings.
ENV_CRITERIA = [
    ("reset", "Reset"),
    ("dynamics", "Dynamics"),
    ("actions", "Actions & errors"),
    ("states", "Injected states"),
    ("api", "Spaces, metadata, repr"),
    ("frames", "`rgb_array` frames"),
    ("text", "`ansi` text"),
]
ID_CRITERIA = [
    ("make", "`make` stack"),
    ("registration", "Registration"),
    ("bridge", "`check_env` / `env_match`"),
    ("golden", "Golden"),
    ("vector_sync", "Sync vector"),
    ("vector_async", "Async vector"),
    ("human", "`human`"),
    ("rgb_array_list", "`rgb_array_list`"),
]
WIDE_CRITERIA = [
    ("wrappers", "Wrapper cells naming it"),
    ("unit_gymnasium", "Gymnasium's own tests (C#)"),
    ("unit_gymnet", "Gym.NET's unit tests"),
]
# Results per golden case: the live cell that verifies and records it, its replay without Python, its codec round trip.
GOLDEN_RESULTS_PER_CASE = 3


def load_open_bugs():
    """The defect ids of every [OpenBugs(...)] test method, read from the test sources.

    Returns:
        dict: (class name, method name) mapped to the list of ids; a class-level attribute is keyed by (class, None).

    Raises:
        OSError: A test source cannot be read.
    """
    found = {}
    attribute = re.compile(r"\[OpenBugs\(([^)]*)\)\]")
    method = re.compile(r"public\s+(?:async\s+\w+\s+|void\s+|Task\s+)(\w+)\s*\(")
    # A declaration, not the word in a comment. Only a class marked [TestClass] owns the methods that follow, so a nested
    # helper class does not take them over; Gymnasium's mirrored classes are named TestXxx, the others XxxTests.
    klass = re.compile(r"^\s*(?:(?:public|internal|private|protected|sealed|static|abstract|partial)\s+)*class\s+(\w+)\b")
    for root, _dirs, files in os.walk(os.path.join(REPO, "tests")):
        for name in files:
            if not name.endswith(".cs"):
                continue
            with open(os.path.join(root, name), encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
            current_class = None
            pending = None
            test_class_next = False
            for line in lines:
                if line.lstrip().startswith("//"):
                    continue
                if "[TestClass" in line:
                    test_class_next = True
                c = klass.search(line)
                if c:
                    if not test_class_next:
                        # A helper class: its methods are not tests, and an attribute pending above it is not its own.
                        continue
                    test_class_next = False
                    if pending is not None:
                        found[(c.group(1), None)] = pending
                        pending = None
                    current_class = c.group(1)
                a = attribute.search(line)
                if a:
                    pending = [s.strip().strip('"') for s in a.group(1).split(",") if s.strip()]
                    continue
                m = method.search(line)
                if m and pending is not None and current_class:
                    found[(current_class, m.group(1))] = pending
                    pending = None
    return found


def read_trx(path, suite):
    """Every result of a TRX file, data rows as results of their own.

    Args:
        path: The TRX file.
        suite: "unit" or "parity", recorded on each result.

    Returns:
        list: dicts with suite, class (full name), short (class name), method, name (the display name, data row
        included), outcome (Passed, Failed, NotExecuted, Inconclusive, ...), and open (whether it is an OpenBugs test).

    Raises:
        xml.etree.ElementTree.ParseError: The file is not TRX XML.
    """
    tree = ElementTree.parse(path)
    definitions = {}
    for unit in tree.getroot().iterfind(".//t:TestDefinitions/t:UnitTest", NS):
        test_method = unit.find("t:TestMethod", NS)
        categories = [item.get("TestCategory") for item in unit.iterfind(".//t:TestCategoryItem", NS)]
        definitions[unit.get("id")] = (test_method.get("className"), test_method.get("name"), categories)
    results = []

    def add(element, definition):
        """Records one result.

        Args:
            element: The UnitTestResult element (a whole test's, or one data row's).
            definition: Its test's (class name, method name, categories).
        """
        class_name, method_name, categories = definition
        results.append({
            "suite": suite, "class": class_name, "short": class_name.split(".")[-1], "method": method_name,
            "name": element.get("testName"), "outcome": element.get("outcome"), "open": "OpenBugs" in categories,
        })

    for result in tree.getroot().iterfind(".//t:Results/t:UnitTestResult", NS):
        definition = definitions[result.get("testId")]
        inner = result.findall("t:InnerResults/t:UnitTestResult", NS)
        if inner:
            # A data-driven test reported as one parent with a result per row: count the rows.
            for row in inner:
                add(row, definition)
        else:
            add(result, definition)
    return results


def families_in(text):
    """The env families a text names, in FAMILIES order.

    Args:
        text: A class, method or display name.

    Returns:
        list: Family names (empty when none is named).
    """
    named = {m.group(1).lower() for m in FAMILY_PATTERN.finditer(text)}
    return [name for name, _ids, _pkg in FAMILIES if name.lower() in named]


def classify(r):
    """Places one result: the families and ids it exercises, and the criterion it checks.

    Args:
        r: A result from read_trx.

    Returns:
        tuple: (families, ids, criterion); families empty for infrastructure, whose criterion then names the area.
    """
    cls, short, method, name = r["class"], r["short"], r["method"], r["name"]
    override = OVERRIDES.get((short, method))
    if override:
        criterion, ids = override
        return [FAMILY_OF_ID[ids[0]]], ids, criterion
    ids = sorted(set(ID_PATTERN.findall(name)), key=lambda i: list(FAMILY_OF_ID).index(i))
    fams = []
    for i in ids:
        if FAMILY_OF_ID[i] not in fams:
            fams.append(FAMILY_OF_ID[i])
    # MuJoCo's names contain classic ones ("InvertedPendulum"): take them out before looking for a family.
    mujoco = MUJOCO_PATTERN.search(name) or MUJOCO_PATTERN.search(method)
    for f in families_in(MUJOCO_PATTERN.sub("", short + " " + method + " " + name)):
        if f not in fams:
            fams.append(f)

    if r["suite"] == "unit":
        if mujoco and not fams:
            return [], [], "MuJoCo (P9, postponed)"
        if not fams:
            return [], [], "unit: " + unit_area(cls)
        return fams, ids, "unit_gymnasium" if ".Gymnasium." in cls else "unit_gymnet"

    # Parity suite: the cross-cutting classes first, then the families' own suites.
    if short == "EnvCoverageLiveParityTests":
        if method.startswith("SyncVector"):
            return fams, ids, "vector_sync"
        if method.startswith("AsyncVector"):
            return fams, ids, "vector_async"
        if method.startswith("HumanMode"):
            return fams, ids, "human"
        if method.startswith("RgbArrayList"):
            return fams, ids, "rgb_array_list"
        return fams, ids or ["CartPole-v0"], "make"
    if short in ("GoldenLiveTests", "GoldenReplayTests"):
        if not fams:
            return [], [], "golden harness"
        return fams, ids, "golden"
    if short == "GymnasiumToolsLiveParityTests":
        if not fams:
            return [], [], "reverse bridge (no env)"
        return fams, ids or [family_ids(fams[0])[-1]], "bridge"
    if short == "RegistrationLiveParityTests":
        if not fams:
            return [], [], "registry"
        return fams, ids, "registration"
    if short == "VectorEnvLiveParityTests":
        return (fams, ids, "vector_sync") if fams else ([], [], "vector API")
    if short == "AsyncVectorEnvLiveParityTests":
        return (fams, ids, "vector_async") if fams else ([], [], "vector API")
    if ".Wrappers." in cls:
        return (fams, ids, "wrappers") if fams else ([], [], "wrappers (no env)")
    if short == "PlayLiveParityTests":
        # play's scenarios run CartPole and Gymnasium's own test env mostly without naming them: one area, not per env.
        return [], [], "play"
    if short == "CoreLiveParityTests":
        return (fams, ids, "make") if fams else ([], [], "core API")
    if ".Envs." in cls and short.endswith("LiveParityTests") and fams:
        for criterion, pattern in OWN_SUITE_RULES:
            if re.search(pattern, method):
                if criterion == "harness":
                    return [], [], "harness"
                return fams[:1], ids, criterion
    return [], [], "parity: " + parity_area(cls, short)


def default_ids(fams, ids, method, criterion):
    """The ids an id-level cell is counted for when its display name names none.

    Args:
        fams: The families the cell exercises.
        ids: The ids its display name names (returned unchanged when not empty).
        method: Its method name, which may name a variant ("...MountainCarContinuous...").
        criterion: The criterion it was placed under; only the id-level ones get ids.

    Returns:
        list: The ids: a variant the method names, else the family's only or primary id.
    """
    if ids or not fams or criterion not in dict(ID_CRITERIA):
        return ids
    for stem, variant in VARIANT_STEMS:
        if stem.lower() in method.lower():
            return [variant]
    family = fams[0]
    own = family_ids(family)
    return [own[0]] if len(own) == 1 else [PRIMARY_ID[family]]


def family_ids(family):
    """A family's registered ids.

    Args:
        family: The family's name.

    Returns:
        list: Its ids in registration order.

    Raises:
        StopIteration: The name is not one of FAMILIES'.
    """
    return next(ids for name, ids, _pkg in FAMILIES if name == family)


def unit_area(cls):
    """The infrastructure area of a unit test that names no env.

    Args:
        cls: The test's full class name.

    Returns:
        str: The area, from the namespace (spaces, wrappers, vector, utils, ...).
    """
    parts = cls.split(".")
    if "Gymnasium" in parts:
        i = parts.index("Gymnasium")
        return "Gymnasium's tests: " + (parts[i + 1] if i + 2 < len(parts) else parts[-1])
    return parts[2] if len(parts) > 3 else parts[-1]


def parity_area(cls, short):
    """The infrastructure area of a parity cell that names no env.

    Args:
        cls: The cell's full class name.
        short: Its class name.

    Returns:
        str: The area (the namespace under Tests, or the class).
    """
    parts = cls.split(".")
    if "Tests" in parts:
        i = len(parts) - 1 - parts[::-1].index("Tests")
        if i + 1 < len(parts) - 1:
            return parts[i + 1] + " (" + short + ")"
    return short


def tally(results, open_bugs):
    """Counts each (family, id, criterion) and each infrastructure area, and lists the open and failing tests.

    Args:
        results: The results of both suites.
        open_bugs: The defect ids per test, from load_open_bugs.

    Returns:
        dict: "family" maps (family, criterion) and "id" maps (id, criterion) to counts of passed, open, failed and
        skipped; "infra" maps an area to the same; "open" lists the open tests with their defects; "failed" lists gated
        tests that did not pass; "unclassified" lists results no rule placed.
    """
    family = defaultdict(lambda: defaultdict(int))
    by_id = defaultdict(lambda: defaultdict(int))
    infra = defaultdict(lambda: defaultdict(int))
    opened, failed, unclassified = [], [], []
    for r in results:
        fams, ids, criterion = classify(r)
        ids = default_ids(fams, ids, r["method"], criterion)
        outcome = r["outcome"]
        if r["open"]:
            # An open test is expected to fail; one that passes is a fixed gap whose attribute should go.
            key = "open" if outcome != "Passed" else "open_but_passing"
            defects = open_bugs.get((r["short"], r["method"])) or open_bugs.get((r["short"], None)) or ["?"]
            opened.append({"test": f"{r['short']}.{r['name']}", "defects": defects, "outcome": outcome, "families": fams})
        elif outcome == "Passed":
            key = "passed"
        elif outcome in ("NotExecuted", "Inconclusive"):
            key = "skipped"
        else:
            key = "failed"
            failed.append(f"{r['short']}.{r['name']} ({outcome})")
        if not fams:
            infra[criterion][key] += 1
            if criterion.startswith("parity: ") and "Envs" in criterion:
                unclassified.append(f"{r['short']}.{r['name']}")
            continue
        for f in fams:
            family[(f, criterion)][key] += 1
        for i in ids:
            by_id[(i, criterion)][key] += 1
        if not ids and criterion in dict(ID_CRITERIA):
            unclassified.append(f"{r['short']}.{r['name']} (no id for {criterion})")
    return {"family": family, "id": by_id, "infra": infra, "open": opened, "failed": failed, "unclassified": unclassified}


def cell(counts):
    """One table cell: the passing count, then open and failing counts when there are any.

    Args:
        counts: A dict of passed, open, open_but_passing, failed and skipped counts.

    Returns:
        str: "—" when nothing ran, else for example "✓ 12", "✓ 12 · open 1", "✗ 2 failed".
    """
    if not counts or sum(counts.values()) == 0:
        return "—"
    parts = []
    if counts.get("passed"):
        parts.append(f"✓ {counts['passed']}")
    if counts.get("open"):
        parts.append(f"open {counts['open']}")
    if counts.get("open_but_passing"):
        parts.append(f"fixed-open {counts['open_but_passing']}")
    if counts.get("skipped"):
        parts.append(f"skip {counts['skipped']}")
    if counts.get("failed"):
        parts.append(f"✗ {counts['failed']} failed")
    return " · ".join(parts)


def id_cell(counts, criterion):
    """A cell of the per-id table: golden results shown as cases when every result of each case passed.

    Args:
        counts: The cell's counts, or None.
        criterion: One of ID_CRITERIA's keys.

    Returns:
        str: For example "✓ 2 cases" for goldens, else what cell() gives.
    """
    if criterion == "golden" and counts and set(counts) <= {"passed"} and counts["passed"] % GOLDEN_RESULTS_PER_CASE == 0:
        cases = counts["passed"] // GOLDEN_RESULTS_PER_CASE
        return f"✓ {cases} case{'s' if cases != 1 else ''}"
    return cell(counts)


def env_cell(t, family, criterion):
    """A cell of the env table, saying why when no dedicated cell exists.

    Args:
        t: The tally from tally().
        family: The env family.
        criterion: One of ENV_CRITERIA's keys.

    Returns:
        str: The counts, or "in every episode cell" for a reset (each episode cell starts with a seeded reset compared
        whole), "n/a" for a mode or state the env does not have, else "—".
    """
    counts = t["family"].get((family, criterion))
    if counts and sum(counts.values()):
        return cell(counts)
    if criterion == "reset" and t["family"].get((family, "dynamics")):
        return "in every episode cell"
    if criterion == "text" and family not in ANSI_FAMILIES:
        return "n/a (no `ansi` mode)"
    if criterion == "states" and family in BOX2D_FAMILIES:
        return "n/a (a Box2D world)"
    return "—"


def render(t):
    """The audit's tables as Markdown.

    Args:
        t: The tally from tally().

    Returns:
        str: The tables, the open tests, any failure and the unclassified results.
    """
    out = []
    out.append("### Per env family: the env itself (its own parity suite)\n")
    out.append("| Env | Ids | " + " | ".join(h for _c, h in ENV_CRITERIA) + " |")
    out.append("|---|---|" + "---|" * len(ENV_CRITERIA))
    for name, ids, pkg in FAMILIES:
        out.append(f"| {name} ({pkg}) | {', '.join(ids)} | " + " | ".join(env_cell(t, name, c) for c, _h in ENV_CRITERIA) + " |")
    out.append("\n### Per registered id: integration layers\n")
    out.append("| Id | " + " | ".join(h for _c, h in ID_CRITERIA) + " |")
    out.append("|---|" + "---|" * len(ID_CRITERIA))
    for name, ids, _pkg in FAMILIES:
        for i in ids:
            out.append(f"| {i} | " + " | ".join(id_cell(t['id'].get((i, c)), c) for c, _h in ID_CRITERIA) + " |")
    out.append("\n### Per env family: wider coverage\n")
    out.append("| Env | " + " | ".join(h for _c, h in WIDE_CRITERIA) + " |")
    out.append("|---|" + "---|" * len(WIDE_CRITERIA))
    for name, _ids, _pkg in FAMILIES:
        out.append(f"| {name} | " + " | ".join(cell(t['family'].get((name, c))) for c, _h in WIDE_CRITERIA) + " |")
    out.append("\n### Infrastructure the envs rest on (tests naming no env)\n")
    out.append("| Area | Results |")
    out.append("|---|---|")
    for area in sorted(t["infra"]):
        out.append(f"| {area} | {cell(t['infra'][area])} |")
    out.append("\n### Open tests ([OpenBugs]: the expectation is Gymnasium's, Gym.NET does not meet it yet)\n")
    for o in sorted(t["open"], key=lambda o: o["test"]):
        out.append(f"- `{o['test']}`: {', '.join(o['defects'])} ({o['outcome']})")
    out.append("\n### Gated tests that did not pass\n")
    if t["failed"]:
        out.extend(f"- {f}" for f in t["failed"])
    else:
        out.append("None.")
    out.append("\n### Unclassified\n")
    if t["unclassified"]:
        out.extend(f"- {u}" for u in t["unclassified"])
    else:
        out.append("None.")
    return "\n".join(out) + "\n"


def main(argv):
    """Reads the TRX files and prints the audit's tables.

    Args:
        argv: The command line without the program name.

    Returns:
        int: 0, or 1 when a gated test failed.

    Raises:
        SystemExit: The command line is malformed (argparse).
        OSError: A TRX file or a test source cannot be read, or the JSON cannot be written.
        xml.etree.ElementTree.ParseError: A TRX file is not XML.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("unit_trx")
    parser.add_argument("parity_trx")
    parser.add_argument("--json", help="also write the raw tally here")
    args = parser.parse_args(argv)
    # The tables hold ✓ and ·, and a redirected stdout on Windows is cp1252 (python -I ignores PYTHONIOENCODING).
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    results = read_trx(args.unit_trx, "unit") + read_trx(args.parity_trx, "parity")
    t = tally(results, load_open_bugs())
    sys.stdout.write(f"{len(results)} results: {sum(1 for r in results if r['suite'] == 'unit')} unit, "
                     f"{sum(1 for r in results if r['suite'] == 'parity')} parity.\n\n")
    sys.stdout.write(render(t))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "family": {f"{k[0]}|{k[1]}": v for k, v in t["family"].items()},
                "id": {f"{k[0]}|{k[1]}": v for k, v in t["id"].items()},
                "infra": t["infra"], "open": t["open"], "failed": t["failed"], "unclassified": t["unclassified"],
            }, f, indent=1)
    return 1 if t["failed"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
