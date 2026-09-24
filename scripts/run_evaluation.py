"""Regenerate data/bundled/evaluation_results.json from held-out test matches.

Usage:
    python scripts/run_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from football_intel.data.adapter import load_bundled_matches  # noqa: E402
from football_intel.evaluation.run import run_full_evaluation  # noqa: E402

if __name__ == "__main__":
    matches = load_bundled_matches()
    results = run_full_evaluation(matches)
    out = Path(__file__).resolve().parents[1] / "data" / "bundled" / "evaluation_results.json"
    out.write_text(__import__("json").dumps(results, indent=2, default=str), encoding="utf-8")
    for scenario in ("prematch", "halftime"):
        print(f"\n== {scenario.upper()} ==")
        for model, m in results["scenario"][scenario].items():
            print(
                f"  {model:12} n={m['n']:5d} logloss={m['log_loss']:.4f} "
                f"brier={m['brier']:.4f} cal|err|={m['calibration_abs_error']:.4f} "
                f"({m['calibration_n_bins']} bins)"
            )
    print(f"\nWrote {out}")