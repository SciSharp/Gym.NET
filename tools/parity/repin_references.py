"""Move every Gymnasium permalink in Gym.NET from one pinned commit to another, line numbers included.

Gym.NET cites the Gymnasium source a member is based on as a pinned GitHub permalink
(https://github.com/Farama-Foundation/Gymnasium/blob/<commit>/<path>#L<start>[-L<end>]), port plan section 3.1 rule 2.
Re-pinning refs/Gymnasium (port plan section 14.10) leaves those links at the old commit, where they still resolve but
no longer describe the oracle the suite runs. This script moves each one to the new commit, carrying its line anchor
through `git diff -U0 <old> <new> -- <path>`, so a link keeps pointing at the same code even where lines moved.

A link is moved only when that is provably safe:
- its file exists at the new commit;
- no line it cites was changed or removed, and nothing was inserted inside a cited range;
- the cited text at the new commit equals the text at the old commit, line for line (a cross-check of the mapping).
Every other link is refused: left at the old commit and listed with the reason, to be reviewed and moved by hand,
because a changed target means the Gym.NET member it documents may need the change too. Links without a line anchor
move whenever the file exists; their file may have changed, which the report lists for review as well.

Usage, from the repository root, after checking out the new commit in refs/Gymnasium and before updating
tools/parity/manifest.json (whose gymnasium.commit is the default old commit):

    python tools/parity/repin_references.py                 # dry run: the report only
    python tools/parity/repin_references.py --write         # rewrite the safe links in place

Exit codes: 0 every link moved (or would move), 1 at least one link was refused, 2 bad arguments or repository state.
Only files tracked by Git are scanned, never refs/ (the submodules themselves); bytes outside the links, the BOM and the
line endings included, are left untouched.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

GYMNASIUM_REPO = "https://github.com/Farama-Foundation/Gymnasium/blob/"
HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def git(repo, *args):
    """Run git in a repository and return its standard output as text.

    Args:
        repo: The repository's working directory.
        *args: The git arguments, e.g. ("diff", "-U0", old, new, "--", path).

    Returns:
        The command's standard output, decoded as UTF-8 (Gymnasium's sources are UTF-8).

    Raises:
        subprocess.CalledProcessError: git exited non-zero (an unknown commit or path, a broken checkout).
    """
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          encoding="utf-8", errors="strict").stdout


def file_lines(gymnasium, commit, path, cache):
    """The lines of a Gymnasium file at a commit, or None when the file does not exist there.

    Args:
        gymnasium: The refs/Gymnasium working directory.
        commit: The full commit id.
        path: The file's path inside the Gymnasium repository.
        cache: A dict memoizing (commit, path) -> lines, since many links cite the same few files.

    Returns:
        The file's lines without their line endings (index 0 is line 1), or None for a path the commit lacks.
    """
    key = (commit, path)
    if key not in cache:
        exists = subprocess.run(["git", "-C", str(gymnasium), "cat-file", "-e", f"{commit}:{path}"],
                                capture_output=True).returncode == 0
        # splitlines() drops "\r\n" and "\n" alike, so a cross-check never fails on line endings alone.
        cache[key] = git(gymnasium, "show", f"{commit}:{path}").splitlines() if exists else None
    return cache[key]


def hunks(gymnasium, old, new, path, cache):
    """The zero-context hunks of a file's diff between two commits, as (old_start, old_count, new_start, new_count).

    Args:
        gymnasium: The refs/Gymnasium working directory.
        old: The commit the links cite today.
        new: The commit they move to.
        path: The file's path inside the Gymnasium repository.
        cache: A dict memoizing path -> hunks.

    Returns:
        The hunks in file order. A count of 0 on the old side is a pure insertion after line old_start; a count of 0
        on the new side is a pure deletion. An empty list means the file is unchanged.
    """
    if path not in cache:
        found = []
        for line in git(gymnasium, "diff", "-U0", "--no-color", "--no-ext-diff", old, new, "--", path).splitlines():
            match = HUNK.match(line)
            if match:
                # A count git omits is 1 (the unified diff format's shorthand for a one-line hunk).
                old_start, old_count, new_start, new_count = match.groups()
                found.append((int(old_start), 1 if old_count is None else int(old_count),
                              int(new_start), 1 if new_count is None else int(new_count)))
        cache[path] = found
    return cache[path]


def map_range(file_hunks, start, end):
    """Carry a cited line range from the old commit to the new one, or explain why it cannot be carried.

    Args:
        file_hunks: The file's hunks from hunks().
        start: The first cited line at the old commit (1-based).
        end: The last cited line at the old commit; equal to start for a single-line anchor.

    Returns:
        (new_start, new_end, None) when every cited line survived unchanged and nothing was inserted between them, or
        (None, None, reason) otherwise.
    """
    delta_start = delta_end = 0
    for old_start, old_count, _, new_count in file_hunks:
        if old_count == 0:
            # A pure insertion sits after old line old_start: it shifts the lines below it, and splits a range that
            # spans it, which changes what the range shows.
            if start <= old_start < end:
                return None, None, f"lines were inserted inside the cited range, after line {old_start}"
            if old_start < start:
                delta_start += new_count
            if old_start < end:
                delta_end += new_count
            continue
        removed_last = old_start + old_count - 1
        if old_start <= end and removed_last >= start:
            return None, None, f"cited lines {max(start, old_start)}-{min(end, removed_last)} changed"
        if removed_last < start:
            delta_start += new_count - old_count
        if removed_last < end:
            delta_end += new_count - old_count
    return start + delta_start, end + delta_end, None


def scan_files(root):
    """The tracked files of the Gym.NET repository that may hold links, refs/ excluded.

    Args:
        root: The Gym.NET repository root.

    Returns:
        Paths relative to root, in Git's order.
    """
    tracked = git(root, "ls-files", "-z").split("\0")
    return [p for p in tracked if p and not p.startswith("refs/")]


def main(argv):
    """Parse the arguments, re-pin or report every link, and return the exit code.

    Args:
        argv: The command line after the program name.

    Returns:
        0 when every link moved or would move, 1 when any was refused, 2 for bad arguments or repository state.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    gymnasium = root / "refs" / "Gymnasium"
    manifest = json.loads((root / "tools" / "parity" / "manifest.json").read_text(encoding="utf-8"))

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", default=manifest["stack"]["gymnasium.commit"],
                        help="the full commit the links cite today (default: tools/parity/manifest.json's gymnasium.commit)")
    parser.add_argument("--new", default=None,
                        help="the full commit to move them to (default: refs/Gymnasium's checked-out HEAD)")
    parser.add_argument("--write", action="store_true", help="rewrite the files; without it, only report")
    args = parser.parse_args(argv)

    try:
        new = args.new or git(gymnasium, "rev-parse", "HEAD").strip()
        old = git(gymnasium, "rev-parse", "--verify", f"{args.old}^{{commit}}").strip()
        new = git(gymnasium, "rev-parse", "--verify", f"{new}^{{commit}}").strip()
    except subprocess.CalledProcessError as failure:
        print(f"error: {failure.stderr.strip() or failure}", file=sys.stderr)
        return 2
    if old == new:
        print(f"error: old and new are the same commit ({old}); check out the new commit in refs/Gymnasium first",
              file=sys.stderr)
        return 2

    # The link's path stops at the characters that end a URL in Markdown, XML docs and Python strings alike.
    link = re.compile(re.escape(GYMNASIUM_REPO + old) + r"/(?P<path>[^\s\"'<>)#\]]+)(?:#L(?P<start>\d+)(?:-L(?P<end>\d+))?)?")
    lines_cache, hunk_cache = {}, {}
    moved = refused = whole_changed = 0
    report = []

    for relative in scan_files(root):
        path = root / relative
        try:
            text = path.read_bytes().decode("utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # a binary file, or a tracked path deleted in the working tree: nothing to re-pin
        if old not in text:
            continue

        def rewrite(match):
            """Replace one link with its re-pinned form, or keep it and record why it was refused.

            Args:
                match: The link's regex match.

            Returns:
                The text that replaces the link: the re-pinned link, or the original one when refused.
            """
            nonlocal moved, refused, whole_changed
            gym_path, start, end = match.group("path"), match.group("start"), match.group("end")
            where = f"{relative}:{text.count(chr(10), 0, match.start()) + 1}"
            new_lines = file_lines(gymnasium, new, gym_path, lines_cache)
            if new_lines is None:
                refused += 1
                report.append(f"REFUSED {where}: {gym_path} does not exist at {new[:8]}")
                return match.group(0)
            if start is None:
                if hunks(gymnasium, old, new, gym_path, hunk_cache):
                    whole_changed += 1
                    report.append(f"REVIEW  {where}: whole-file link to {gym_path}, which changed")
                moved += 1
                return f"{GYMNASIUM_REPO}{new}/{gym_path}"
            first, last = int(start), int(end) if end else int(start)
            new_first, new_last, reason = map_range(hunks(gymnasium, old, new, gym_path, hunk_cache), first, last)
            if reason is not None:
                refused += 1
                report.append(f"REFUSED {where}: {gym_path}#L{first}{f'-L{last}' if end else ''}: {reason}")
                return match.group(0)
            old_lines = file_lines(gymnasium, old, gym_path, lines_cache)
            # The cross-check: a mapping bug must never move a link to different code silently.
            if old_lines[first - 1:last] != new_lines[new_first - 1:new_last]:
                refused += 1
                report.append(f"REFUSED {where}: {gym_path}#L{first}: the mapped text differs (a mapping bug?)")
                return match.group(0)
            moved += 1
            if (new_first, new_last) != (first, last):
                report.append(f"moved   {where}: {gym_path}#L{first}{f'-L{last}' if end else ''} -> "
                              f"#L{new_first}{f'-L{new_last}' if end else ''}")
            anchor = f"#L{new_first}" + (f"-L{new_last}" if end else "")
            return f"{GYMNASIUM_REPO}{new}/{gym_path}{anchor}"

        updated = link.sub(rewrite, text)
        if args.write and updated != text:
            # Encode back exactly: the decoded text kept its BOM and line endings, so only the links change.
            path.write_bytes(updated.encode("utf-8"))

    print("\n".join(report))
    print(f"\n{old[:8]} -> {new[:8]}: {moved} moved, {refused} refused, {whole_changed} whole-file links to changed "
          f"files{'' if args.write else ' (dry run: nothing written)'}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
