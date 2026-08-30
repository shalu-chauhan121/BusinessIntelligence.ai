"""
Static guard against the retired 4-stage pipeline creeping back in.

A9 deleted `engines/{act,contest,hypotheses,investigate,llm_hypotheses,
signals}.py`, `services/pipeline.py` and `personas/reframe.py`. Nothing left
in the tree should import them -- not because a stray import would break
today (it would, immediately, with `ModuleNotFoundError`), but because a
partial revert or a merge conflict could silently resurrect one of these
files without resurrecting its callers' understanding of why it was removed.
This walks the source tree with `ast` rather than `grep` so a multi-line or
aliased import cannot slip past it.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import List, Tuple

BACKEND = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND / "app"
SCRIPTS_ROOT = BACKEND.parent / "scripts"

# Dotted module paths that must not exist and must not be imported from
# anywhere still in the tree.
RETIRED_MODULES = {
    "app.engines.act",
    "app.engines.contest",
    "app.engines.hypotheses",
    "app.engines.investigate",
    "app.engines.llm_hypotheses",
    "app.engines.signals",
    "app.services.pipeline",
    "app.personas.reframe",
}
# The relative forms the same modules are imported under from within `app/`
# itself (`from ..engines.act import act`, `from .pipeline import run_full`,
# `from ...personas.reframe import reframe_for`, etc). Matched on the
# `(level, module)` pair `ast.ImportFrom` records, resolved against each
# file's own package.
RETIRED_LEAVES = {"act", "contest", "hypotheses", "investigate",
                  "llm_hypotheses", "signals", "pipeline", "reframe"}
RETIRED_PARENT_PACKAGES = {"engines", "services", "personas"}


def _package_of(path: Path) -> List[str]:
    """
    The dotted package a module file lives in, e.g. `app.api` for
    `app/api/routes_analysis.py`, resolved from its position under `app/`.

    `scripts/` sits beside `backend/`, not inside `app/`, and holds no
    `__init__.py` -- it is never a relative-import target, so its files get a
    flat, one-element package rather than a path relative to `APP_ROOT`.
    """
    if SCRIPTS_ROOT.exists() and path.is_relative_to(SCRIPTS_ROOT):
        return ["scripts"]
    rel = path.relative_to(APP_ROOT.parent)
    parts = list(rel.with_suffix("").parts)
    return parts[:-1]


def _resolve_relative(level: int, module: str, package: List[str]) -> str:
    """Reimplements the part of Python's import resolution this test needs:
    `level` dots walk up `package` before `module` is appended."""
    base = package[:len(package) - (level - 1)] if level > 1 else package
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _iter_source_files() -> List[Path]:
    files = list(APP_ROOT.rglob("*.py"))
    if SCRIPTS_ROOT.exists():
        files += list(SCRIPTS_ROOT.rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


def _find_violations() -> List[Tuple[str, int, str]]:
    violations = []
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:                      # pragma: no cover - defensive
            violations.append((str(path), exc.lineno or 0, f"file does not parse: {exc}"))
            continue
        package = _package_of(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in RETIRED_MODULES:
                        violations.append((str(path), node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0:
                    resolved = node.module or ""
                else:
                    resolved = _resolve_relative(node.level, node.module or "", package)
                if resolved in RETIRED_MODULES:
                    violations.append((str(path), node.lineno, resolved))
                    continue
                # `from ..engines import act` / `from . import pipeline` --
                # the retired leaf arrives as an imported NAME, not inside
                # `module`, so check the aliases too.
                leaf = resolved.rsplit(".", 1)[-1] if resolved else ""
                if leaf in RETIRED_PARENT_PACKAGES or resolved in (
                    "app.engines", "app.services", "app.personas"):
                    for alias in node.names:
                        if alias.name in RETIRED_LEAVES:
                            violations.append(
                                (str(path), node.lineno, f"{resolved}.{alias.name}"))
    return violations


class TestRetiredPipelineStaysDeleted(unittest.TestCase):
    def test_no_source_file_imports_a_retired_module(self):
        violations = _find_violations()
        self.assertEqual(violations, [], (
            "the following files import a module retired at A9:\n" +
            "\n".join(f"  {f}:{ln}  {name}" for f, ln, name in violations)
        ))

    def test_the_retired_modules_are_actually_gone(self):
        import importlib
        for dotted in sorted(RETIRED_MODULES):
            with self.assertRaises(ModuleNotFoundError, msg=f"{dotted} still importable"):
                importlib.import_module(dotted)


if __name__ == "__main__":
    unittest.main()
