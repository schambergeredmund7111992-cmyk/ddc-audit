"""Regeneration pipeline: recompute the paper's numbers from the shipped
regeneration inputs and reconcile them against the frozen expected values.

Two renames are papered over on import so the author's modules run unmodified
inside the competition package:

* ``from eval.metrics import ...`` resolves to the packaged, numerically
  identical ``ddc_audit.metrics`` (see ``_metrics_compat``);
* ``from scripts.regenerate.<module> import ...`` -- the path the research
  repository used -- resolves to this package, which the package ships as
  ``scripts_regenerate``.
"""
import sys

from scripts_regenerate._metrics_compat import ensure_package_on_path, install

ensure_package_on_path()
METRIC_SOURCE = install()

# Alias the research-repository module path onto this package.
sys.modules.setdefault("scripts", sys.modules[__name__])
sys.modules.setdefault("scripts.regenerate", sys.modules[__name__])
