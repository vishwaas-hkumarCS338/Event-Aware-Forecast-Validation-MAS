# scripts/07_eval_lp.py
import requests
import json
import csv
from pathlib import Path

API_URL = "http://127.0.0.1:8000/validate"
SCENARIOS_CSV = Path("evaluations/lp_scenarios.csv")
OUTPUT_JSON = Path("evaluations/lp_eval.json")


def load_scenarios(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["forecast_units"] = int(row["forecast_units"])
            rows.append(row)
    return rows


def main():
    scenarios = load_scenarios(SCENARIOS_CSV)
    results = []
    agg = {
        "count": 0,
        "feasible_count": 0,
        "total_flex_sum": 0.0,
        "remaining_gap_sum": 0.0,
    }

    for s in scenarios:
        resp = requests.post(API_URL, json=s)
        data = resp.json()
        sim = data.get("simulation", {})

        entry = {
            "scenario": s,
            "lp_success": sim.get("lp_success"),
            "total_flex": sim.get("total_flex"),
            "remaining_gap": sim.get("remaining_gap"),
            "supply_feasible": data.get("supply_feasible"),
        }
        results.append(entry)

        agg["count"] += 1
        agg["total_flex_sum"] += entry["total_flex"] or 0
        agg["remaining_gap_sum"] += entry["remaining_gap"] or 0
        if entry["supply_feasible"]:
            agg["feasible_count"] += 1

    # compute aggregates
    if agg["count"] > 0:
        agg["total_flex_mean"] = agg["total_flex_sum"] / agg["count"]
        agg["remaining_gap_mean"] = agg["remaining_gap_sum"] / agg["count"]
        agg["feasible_rate"] = agg["feasible_count"] / agg["count"]

    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"aggregates": agg, "results": results}, f, indent=2)

    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
