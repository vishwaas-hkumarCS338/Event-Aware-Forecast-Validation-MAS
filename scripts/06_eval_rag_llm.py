# scripts/06_eval_rag_llm.py
import requests
import json
import csv
import time
from pathlib import Path

API_URL = "http://127.0.0.1:8000/validate"
SCENARIOS_CSV = Path("evaluations/rag_scenarios.csv")
OUTPUT_JSONL = Path("evaluations/rag_eval.jsonl")


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
    OUTPUT_JSONL.parent.mkdir(exist_ok=True)

    with open(OUTPUT_JSONL, "w") as outfile:
        for scenario in scenarios:
            t0 = time.time()
            resp = requests.post(API_URL, json=scenario)
            latency_ms = (time.time() - t0) * 1000

            result = {
                "scenario": scenario,
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
            }

            try:
                data = resp.json()
                result["parsed_json"] = True
                result["event_rag_ok"] = data.get("event_rag", {}).get("ok")
                result["llm_json"] = data.get("event_rag", {}).get("llm_json")
                result["retriever_top1_score"] = (
                    data.get("retrieved_context_chunks", [{}])[0].get("score")
                    if data.get("retrieved_context_chunks")
                    else None
                )
            except Exception as e:
                result["parsed_json"] = False
                result["error"] = str(e)

            outfile.write(json.dumps(result) + "\n")
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
