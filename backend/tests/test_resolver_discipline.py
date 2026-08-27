"""
Static guard against the bug class fixed in this batch.

Six call sites (`contest.temporal_check` x2, `contest.consistency_check`,
`act.monitoring_threshold`, `drivers.member_persistence` x2) had `resolver`
sitting in scope as a parameter and simply did not pass it to `compute` /
`weekly_frame` / `member_change_table`. Each is individually easy to fix and
easy to reintroduce -- the resolver is threaded by hand through roughly forty
call sites across the engines, so omitting it silently compiles, silently
runs, and silently returns NaN or a wrong number only for contract-only or
contract-overridden KPIs. The retail demo dataset cannot expose this at all,
because its KPIs are simultaneously seed-registry keys and raw columns.

This test does not re-check today's known sites; it walks every module under
`app/` with Python's own `ast` module and flags *any* call, anywhere, to one
of the resolver-taking functions in `app.engines.metrics` / `app.engines.analysis`
that does not supply enough positional arguments to reach `resolver`, and does
not supply it by keyword either. New code that repeats this mistake fails the
build before it fails a user's investigation.

Scope note: the batch plan named `app/engines/` and `app/agent/`; this walks
the whole `app/` package, since call sites already exist in `app/kpi/` and
`app/api/` and the discipline is worth having everywhere, not only where the
bug happened to live today.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from app.engines import analysis, metrics

APP_ROOT = Path(__file__).resolve().parents[1] / "app"

# name -> the real function object, so the required argument position is read
# from the live signature rather than hand-copied and left to drift.
GUARDED_FUNCTIONS = {
    "compute": metrics.compute,
    "metric_spec": metrics.metric_spec,
    "metric_label": metrics.metric_label,
    "metric_unit": metrics.metric_unit,
    "higher_is_better": metrics.higher_is_better,
    "metric_components": metrics.metric_components,
    "metric_description": metrics.metric_description,
    "weekly_frame": analysis.weekly_frame,
    "member_change_table": analysis.member_change_table,
}


def _resolver_positions() -> Dict[str, int]:
    """The zero-based position of the `resolver` parameter in each guarded
    function's real signature. Fails loudly if a signature ever drops the
    parameter, rather than silently checking nothing."""
    positions = {}
    for name, fn in GUARDED_FUNCTIONS.items():
        params = list(inspect.signature(fn).parameters)
        assert "resolver" in params, (
            f"{name} no longer declares a 'resolver' parameter -- this "
            "discipline test has nothing left to check for it and must be updated.")
        positions[name] = params.index("resolver")
    return positions


def _find_violations(app_root: Path) -> List[Tuple[str, int, str, str]]:
    """
    (file, line, function name, source snippet) for every call site that
    imports one of the guarded names directly and calls it without enough
    positional arguments or a `resolver=` keyword to supply the resolver.

    Deliberately narrow to prevent false positives: only bare-name calls
    (`compute(...)`, not `some_object.compute(...)`) in files that imported
    the name via `from ... import <name>` are checked. A `CompiledKpi.compute`
    or `MetricSpec`-bound method call is a different, already-bound function
    and is correctly out of scope.
    """
    resolver_index = _resolver_positions()
    guarded_names = set(GUARDED_FUNCTIONS)
    violations: List[Tuple[str, int, str, str]] = []

    for path in app_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        src = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        imported_as: Dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in guarded_names:
                        imported_as[alias.asname or alias.name] = alias.name
        if not imported_as:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            local_name = node.func.id
            real_name = imported_as.get(local_name)
            if real_name is None:
                continue

            need_idx = resolver_index[real_name]
            n_positional = len(node.args)
            has_star_arg = any(isinstance(a, ast.Starred) for a in node.args)
            has_resolver_kw = any(kw.arg == "resolver" for kw in node.keywords)
            has_kwargs_splat = any(kw.arg is None for kw in node.keywords)  # **kwargs

            satisfied = (
                n_positional > need_idx
                or has_resolver_kw
                or has_star_arg           # *args could contain it -- do not flag
                or has_kwargs_splat       # **kwargs could contain it -- do not flag
            )
            if not satisfied:
                snippet = ast.get_source_segment(src, node) or ""
                violations.append((str(path.relative_to(app_root.parent)),
                                   node.lineno, real_name, snippet))
    return violations


class TestResolverDiscipline(unittest.TestCase):
    def test_no_call_site_omits_the_resolver_argument(self):
        violations = _find_violations(APP_ROOT)
        if violations:
            report = "\n".join(
                f"  {file}:{line}  {fn}(...)  ->  {snippet}"
                for file, line, fn, snippet in violations)
            self.fail(
                "The following call(s) have `resolver` in scope or available "
                "and do not pass it, which silently falls back to the seed "
                "KPI registry and returns wrong or NaN values for any "
                "contract-only KPI:\n" + report)

    def test_detector_actually_flags_a_planted_violation(self):
        """
        The detector itself must not be a tautological no-op. Prove it flags
        a call that mirrors the exact shape of the bug this batch fixed:
        `resolver` sitting in scope, unpassed.
        """
        sample = (
            "from app.engines.metrics import compute\n"
            "from app.engines.analysis import weekly_frame\n"
            "\n"
            "def broken(df, key, resolver):\n"
            "    a = compute(df, key)\n"          # missing resolver -- violation
            "    b = weekly_frame(df, key)\n"      # missing resolver -- violation
            "    return a, b\n"
            "\n"
            "def fixed_positional(df, key, resolver):\n"
            "    return compute(df, key, resolver)\n"
            "\n"
            "def fixed_keyword(df, key, resolver):\n"
            "    return weekly_frame(df, key, resolver=resolver)\n"
        )
        tmp_dir = Path(__file__).resolve().parent / "_discipline_selftest_scratch"
        tmp_dir.mkdir(exist_ok=True)
        sample_file = tmp_dir / "sample_module.py"
        try:
            sample_file.write_text(sample, encoding="utf-8")
            violations = _find_violations(tmp_dir)
            flagged_functions = sorted(v[2] for v in violations)
            self.assertEqual(flagged_functions, ["compute", "weekly_frame"],
                             "the detector must flag exactly the two planted "
                             "violations and must not flag the two fixed calls")
        finally:
            sample_file.unlink(missing_ok=True)
            tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
