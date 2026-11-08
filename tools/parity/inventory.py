"""List Gymnasium's public names by AST, the input of Gym.NET's API inventory test (port plan section 14.6).

Scans every module of the pinned refs/Gymnasium/gymnasium package (files and packages whose name starts with an
underscore are private, __init__.py excepted) without importing anything, so it needs no NumPy, pygame or Box2D:

- a module that declares __all__ exports exactly those names;
- a module without __all__ exports its top-level classes and functions whose names do not start with an underscore
  (module-level constants of such modules, e.g. an env's FPS or SCALE, are implementation details and are left out).

Each exported name is resolved through the module's `from ... import ...` statements to the module that defines it, so
`gymnasium.Env` and `gymnasium.core.Env` are one entry (module gymnasium.core, exported as both). Names that are
modules themselves (`gymnasium.spaces` in gymnasium.__all__) are not entries.

Output (stdout, or --write to tools/parity/gymnasium_inventory.json): a JSON object with the Gymnasium commit the scan
ran on and the entries sorted by module and name:

    {"gymnasium_commit": "...", "entries": [{"module": "gymnasium.core", "name": "Env", "kind": "class",
     "exports": ["gymnasium.Env", "gymnasium.core.Env"]}, ...]}

kind is "class", "function" or "value" (an assignment, e.g. gymnasium.envs.registration.registry). Regenerate it after
every re-pin; tests/Gym.Tests/Gymnasium/InventoryTests.cs refuses an inventory of another commit than the manifest pins.

Usage, from the repository root:

    python tools/parity/inventory.py            # print
    python tools/parity/inventory.py --write    # update tools/parity/gymnasium_inventory.json
"""
import argparse
import ast
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "refs" / "Gymnasium" / "gymnasium"
OUTPUT = ROOT / "tools" / "parity" / "gymnasium_inventory.json"


def module_name(path):
    """The dotted module name of a source file of the package.

    Args:
        path: A .py file under refs/Gymnasium/gymnasium.

    Returns:
        For example "gymnasium.spaces.box", or "gymnasium.spaces" for its __init__.py.
    """
    parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def is_public_file(path):
    """Whether a file is a public module: no path component below the package starts with an underscore, except __init__.

    Args:
        path: A .py file under the package.

    Returns:
        True for public modules and packages.
    """
    for part in path.relative_to(PACKAGE).parts:
        name = part[:-3] if part.endswith(".py") else part
        if name.startswith("_") and name != "__init__":
            return False
    return True


class Module:
    """One parsed module: what it defines at top level, what it imports by name, and its __all__ if it declares one."""

    def __init__(self, name, path, is_package):
        """Parses a module.

        Args:
            name: The dotted module name.
            path: The source file.
            is_package: Whether the file is a package's __init__.py, which decides how relative imports resolve.

        Raises:
            SyntaxError: The file is not valid Python for this interpreter.
        """
        self.name = name
        self.is_package = is_package
        self.defined = {}
        self.imported = {}
        self.all = None
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            self._visit(node)

    def _visit(self, node):
        """Records one top-level statement: a definition, a by-name import, or __all__.

        Args:
            node: The statement.
        """
        if isinstance(node, ast.ClassDef):
            self.defined[node.name] = "class"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self.defined[node.name] = "function"
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        self.all = [element.value for element in node.value.elts
                                    if isinstance(element, ast.Constant) and isinstance(element.value, str)]
                    else:
                        # A later definition of the same name (a class after a placeholder) keeps the stronger kind.
                        self.defined.setdefault(target.id, "value")
        elif isinstance(node, ast.ImportFrom):
            source = self._absolute(node.module, node.level)
            for alias in node.names:
                self.imported[alias.asname or alias.name] = (source, alias.name)
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` and version guards still define or import names at module level.
            for child in node.body + node.orelse:
                self._visit(child)
        elif isinstance(node, ast.Try):
            for child in node.body + [h for handler in node.handlers for h in handler.body] + node.orelse + node.finalbody:
                self._visit(child)

    def _absolute(self, module, level):
        """Resolves a possibly relative import's module to an absolute dotted name.

        Args:
            module: The module part of `from X import y` (None for `from . import y`).
            level: The number of leading dots.

        Returns:
            The absolute module name.
        """
        if level == 0:
            return module
        base = self.name.split(".")
        # A package's __init__ is the package itself, so one dot means the package; a module's one dot means its parent.
        drop = level - 1 if self.is_package else level
        base = base[: len(base) - drop]
        return ".".join(base + ([module] if module else []))

    def exports(self):
        """The names this module exports.

        Returns:
            __all__ when declared, otherwise the public top-level classes and functions.
        """
        if self.all is not None:
            return list(self.all)
        return [n for n, kind in self.defined.items() if kind in ("class", "function") and not n.startswith("_")]


def resolve(modules, module, name, seen=None):
    """Follows imports from a module to the module that defines a name.

    Args:
        modules: Every parsed module by name.
        module: Where the name is looked up.
        name: The name.
        seen: The (module, name) pairs already followed, which stops import cycles.

    Returns:
        (defining module, name, kind), kind "module" when the name is a module, or "unknown" when it leaves the package.
    """
    seen = seen or set()
    if (module, name) in seen:
        return module, name, "unknown"
    seen.add((module, name))
    parsed = modules.get(module)
    if parsed is None:
        return module, name, "unknown"
    if name in parsed.defined:
        return module, name, parsed.defined[name]
    if name in parsed.imported:
        source, original = parsed.imported[name]
        if f"{source}.{original}" in modules:
            return f"{source}.{original}", original, "module"
        return resolve(modules, source, original, seen)
    if f"{module}.{name}" in modules:
        return f"{module}.{name}", name, "module"
    return module, name, "unknown"


def gymnasium_commit():
    """The commit refs/Gymnasium has checked out.

    Returns:
        The full commit id.

    Raises:
        subprocess.CalledProcessError: git cannot read the submodule.
    """
    return subprocess.run(["git", "-C", str(PACKAGE.parent), "rev-parse", "HEAD"], check=True, capture_output=True,
                          encoding="utf-8").stdout.strip()


def inventory():
    """Scans the package.

    Returns:
        The inventory object described in the module docstring.
    """
    modules = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if is_public_file(path):
            modules[module_name(path)] = Module(module_name(path), path, path.name == "__init__.py")

    entries = {}
    for name in sorted(modules):
        for exported in modules[name].exports():
            source, original, kind = resolve(modules, name, exported)
            if kind == "module":
                continue
            entry = entries.setdefault((source, original), {"module": source, "name": original, "kind": kind, "exports": set()})
            entry["exports"].add(f"{name}.{exported}")

    ordered = [dict(e, exports=sorted(e["exports"])) for _, e in sorted(entries.items())]
    return {"gymnasium_commit": gymnasium_commit(), "entries": ordered}


def main(argv):
    """Prints the inventory, or writes it with --write.

    Args:
        argv: The command line after the program name.

    Returns:
        The exit code, 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.relative_to(ROOT)} instead of printing")
    args = parser.parse_args(argv)
    # One entry per line keeps the committed file diffable when a re-pin adds or removes names.
    data = inventory()
    lines = ["{", f'  "gymnasium_commit": {json.dumps(data["gymnasium_commit"])},', '  "entries": [']
    lines += [f"    {json.dumps(entry, separators=(', ', ': '))}" + ("," if i < len(data["entries"]) - 1 else "")
              for i, entry in enumerate(data["entries"])]
    lines += ["  ]", "}"]
    text = "\n".join(lines) + "\n"
    if args.write:
        OUTPUT.write_bytes(text.encode("utf-8"))
        print(f"wrote {len(data['entries'])} entries to {OUTPUT.relative_to(ROOT)}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
