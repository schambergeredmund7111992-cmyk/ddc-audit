"""Make ``from eval.metrics import ...`` resolve to the packaged metrics module.

The author's regeneration scripts were written against the research repository,
where the metric lives at ``eval/metrics.py``. The competition package ships
that implementation as ``ddc_audit.metrics`` (numerically identical, see
``tests/test_metrics.py::test_numerically_identical_to_reference_implementation``).
This shim installs a synthetic ``eval`` package in ``sys.modules`` so the
scripts import unchanged.

If a real ``eval`` package is importable (the research repository is on the
path) the real one wins and nothing is shadowed.
"""
from __future__ import annotations

import importlib
import sys
import types


def install() -> str:
    try:
        importlib.import_module("eval.metrics")
        return "research repository (eval.metrics)"
    except Exception:  # noqa: BLE001 - any failure means: use the packaged copy
        pass

    from ddc_audit import metrics as packaged

    eval_module = sys.modules.get("eval")
    if eval_module is None:
        eval_module = types.ModuleType("eval")
        eval_module.__path__ = []  # mark as a package
        sys.modules["eval"] = eval_module
    eval_module.metrics = packaged
    sys.modules["eval.metrics"] = packaged
    return "packaged ddc_audit.metrics"


def ensure_package_on_path() -> None:
    """Put the sibling ``ddc_audit`` source tree on ``sys.path`` when needed."""
    from pathlib import Path

    here = Path(__file__).resolve()
    for base in here.parents:
        for candidate in (base / "src", base / "code" / "ddc_audit" / "src"):
            if (candidate / "ddc_audit" / "metrics.py").is_file():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return
