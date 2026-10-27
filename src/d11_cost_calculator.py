"""
Task D11 — cost & sensor-lifespan calculator.

Uses the MEASURED physical-probe usage reduction (from Phase 7) to ESTIMATE
cost and lifespan impact. All assumptions are stated explicitly and labelled as
estimates — we never present these as measured facts.

Model of probe wear: electrode life is driven mainly by cumulative immersion /
measurement activations. If usage drops by factor (1 - u), calendar lifespan
scales by roughly 1/u under a usage-proportional wear assumption (stated as a
first-order approximation, not a guarantee).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"


def main():
    conf = json.load(open(METRICS / "phase7_confidence.json"))
    reduction_pct = conf["reduction_pct"]            # measured, e.g. 75.6
    usage_fraction = conf["physical_fraction"]        # measured, e.g. 0.244

    # ---- stated assumptions (editable) ----
    A = {
        "probe_price_usd": 45.0,          # typical replacement pH probe
        "baseline_life_months": 6.0,      # manufacturer: ~6 months continuous
        "labor_per_replacement_usd": 15.0,
        "horizon_years": 3.0,             # planning horizon for the estimate
    }

    # Baseline (always-physical): replacements over the horizon
    months = A["horizon_years"] * 12
    base_replacements = months / A["baseline_life_months"]
    base_cost = base_replacements * (A["probe_price_usd"] +
                                     A["labor_per_replacement_usd"])

    # Proposed: under usage-proportional wear, life extends by ~1/usage_fraction
    # (first-order approximation). Clip the multiplier for realism.
    life_multiplier = min(1.0 / max(usage_fraction, 1e-6), 8.0)
    prop_life_months = A["baseline_life_months"] * life_multiplier
    prop_replacements = months / prop_life_months
    prop_cost = prop_replacements * (A["probe_price_usd"] +
                                     A["labor_per_replacement_usd"])

    savings = base_cost - prop_cost

    out = {
        "measured": {
            "physical_usage_fraction": round(usage_fraction, 4),
            "usage_reduction_pct": round(reduction_pct, 1),
        },
        "assumptions": A,
        "estimate": {
            "baseline_life_months": A["baseline_life_months"],
            "estimated_proposed_life_months": round(prop_life_months, 1),
            "estimated_life_multiplier": round(life_multiplier, 2),
            "baseline_replacements_over_horizon": round(base_replacements, 2),
            "proposed_replacements_over_horizon": round(prop_replacements, 2),
            "baseline_cost_usd": round(base_cost, 2),
            "proposed_cost_usd": round(prop_cost, 2),
            "estimated_savings_usd": round(savings, 2),
            "estimated_savings_pct": round(100 * savings / base_cost, 1),
        },
        "disclaimer": (
            "ESTIMATE ONLY. Based on the measured usage reduction plus a "
            "first-order usage-proportional wear assumption. Actual probe life "
            "depends on chemistry, temperature, calibration and handling; not "
            "directly measured in this project."),
    }
    with open(METRICS / "cost_lifespan_estimate.json", "w") as f:
        json.dump(out, f, indent=2)

    e = out["estimate"]
    print("Cost & sensor-lifespan ESTIMATE (assumptions stated)")
    print(f"  measured usage reduction : {reduction_pct:.1f}% "
          f"(probe used {usage_fraction*100:.1f}% of the time)")
    print(f"  baseline probe life      : {A['baseline_life_months']} months")
    print(f"  est. proposed probe life : {e['estimated_proposed_life_months']} "
          f"months (~{e['estimated_life_multiplier']}x)")
    print(f"  over {A['horizon_years']} yrs: baseline ${e['baseline_cost_usd']} "
          f"vs proposed ${e['proposed_cost_usd']}")
    print(f"  estimated savings        : ${e['estimated_savings_usd']} "
          f"({e['estimated_savings_pct']}%)")
    print(f"\n[save] results/metrics/cost_lifespan_estimate.json")


if __name__ == "__main__":
    main()
