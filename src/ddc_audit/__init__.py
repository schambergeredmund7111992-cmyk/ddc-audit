"""ddc-audit: pre-submission drug-discrimination audit for single-cell
perturbation prediction models.

The drug-discrimination control (DDC) asks whether a held-out drug's predicted
response matches its own measured response better than the measured responses
of other held-out drugs in the same cell line. A collapsed model that emits a
drug-invariant response template scores near the chance value of 0.5 even when
its per-pair correlations look acceptable.

Method reference: Zhao & Chen, "Beyond per-pair correlation: a
drug-discrimination control exposes prediction collapse in single-cell
multi-drug perturbation models" (preprint). Core scoring functions are adapted
from the paper's reference implementation (github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark,
eval/metrics.py) and are numerically identical to it.
"""

__version__ = "0.2.1"
