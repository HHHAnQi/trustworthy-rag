"""
src/conformal.py  —  Split Conformal Prediction (abstention / risk control)

Theory (one paragraph)
-----------------------
Given n calibration nonconformity scores s₁…sₙ and a target miscoverage
rate α ∈ (0,1), the conformal threshold is:

    q̂ = the ⌈(n+1)(1-α)⌉/n quantile of {s₁,…,sₙ}   (using method='higher')

At test time: accept (answer) if s_test ≤ q̂, else abstain.

Guarantee: P(error on answered queries) ≤ α, in expectation over the
draw of the calibration set (Vovk et al. 2005; Angelopoulos & Bates 2021).

method='higher'
---------------
np.quantile with method='higher' rounds UP to the next observed value.
This is the conservative choice that ensures coverage ≥ 1-α rather than
just ≈ 1-α.  ConU uses the same convention (np.quantile, method='higher').
"""

from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Pure function used by calibrate.py ────────────────────────────────────────

def compute_qhat(scores: List[float], alpha: float) -> float:
    """
    Compute the conformal threshold q̂.

    Parameters
    ----------
    scores : nonconformity scores from calibration set  (s = 1 - confidence)
    alpha  : target miscoverage rate, e.g. 0.1

    Returns
    -------
    float  q̂ threshold
    """
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    level = math.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    # method='higher': conservative, matches ConU implementation
    qhat = float(np.quantile(scores, level, method="higher"))
    return qhat


# ── Inference-time predictor ───────────────────────────────────────────────────

class ConformalPredictor:
    """
    Loads pre-computed q̂ table (from calibrate.py) and makes accept/abstain
    decisions at inference time.
    """

    def __init__(self, config: dict):
        """
        config keys
        -----------
        qhat_path      str    path to JSON file written by calibrate.py
        default_alpha  float  fallback α if caller does not specify  (default 0.1)
        """
        self.default_alpha: float        = config.get("default_alpha", 0.1)
        self.qhat_table: Dict[str, float] = {}

        path = config.get("qhat_path")
        if path:
            p = Path(path)
            if p.exists():
                with open(p) as f:
                    self.qhat_table = json.load(f)
                logger.info("Loaded q̂ table: %s", self.qhat_table)
            else:
                logger.warning(
                    "qhat_path %s not found. Run scripts/calibrate.py first.", p
                )

    def predict(self, nonconformity_score: float, alpha: Optional[float] = None) -> dict:
        """
        Return accept/abstain decision.

        Parameters
        ----------
        nonconformity_score : s = 1 - confidence  (from Uncertainty.compute_nonconformity)
        alpha               : miscoverage level; falls back to default_alpha

        Returns dict with keys: accepted, threshold, alpha, score
        """
        a = alpha if alpha is not None else self.default_alpha
        threshold = self._get_threshold(a)
        return {
            "accepted":  bool(nonconformity_score <= threshold),
            "threshold": round(threshold, 4),
            "alpha":     a,
            "score":     round(nonconformity_score, 4),
        }

    def available_alphas(self) -> List[float]:
        return [float(k) for k in self.qhat_table]

    def _get_threshold(self, alpha: float) -> float:
        key = str(round(alpha, 4))
        if key in self.qhat_table:
            return float(self.qhat_table[key])
        # Try without rounding (e.g. "0.1" stored as "0.1")
        key2 = str(alpha)
        if key2 in self.qhat_table:
            return float(self.qhat_table[key2])
        # Fall back to nearest calibrated alpha
        if self.qhat_table:
            available = [(float(k), v) for k, v in self.qhat_table.items()]
            nearest_k, nearest_v = min(available, key=lambda x: abs(x[0] - alpha))
            logger.warning("α=%.4f not in table; using nearest α=%.4f", alpha, nearest_k)
            return float(nearest_v)
        # No calibration at all
        logger.error("No q̂ data. Defaulting to 0.5. Run calibrate.py.")
        return 0.5
