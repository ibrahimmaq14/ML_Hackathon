"""Run real API requests, retaining only aggregate evidence."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

BASE = Path(__file__).resolve().parent


def run(api):
    with urlopen(api + "/api/trips") as response:
        trips = json.load(response)["trips"]
    expected = {row["trip"]: row for row in json.loads(
        (BASE / "results/refinement/comparison.json").read_text())["per_trip"]}
    if set(expected) != set(trips):
        raise ValueError("Rehearsal requires the student-test split, never reserved evaluator trips.")
    records = []
    for trip in trips:
        with urlopen(Request(api + "/api/predict/" + trip, method="POST"), timeout=300) as response:
            payload = json.load(response)
        count = len(payload["gps"])
        assert count == expected[trip]["observations"] == len(payload["decoded_arc_ids"])
        with np.load(BASE / f"results/refinement/{trip}.npz") as saved:
            np.testing.assert_array_equal(payload["decoded_arc_ids"], saved["decoder_arc_ids"])
        assert payload["sections"] and abs(payload["accuracy"] - expected[trip]["top1"]) < 1e-10
        assert set(payload) == {"gps","roads","trip","decoded_arc_ids","breaks","accuracy","inference_seconds","sections"}
        row = dict(trip=trip, observations=count, accuracy=payload["accuracy"],
                   sections=len(payload["sections"]), inference_seconds=payload["inference_seconds"])
        records.append(row)
        print(trip, count, f"{payload['accuracy']:.4%}", flush=True)
    (BASE / "results/demo").mkdir(exist_ok=True)
    (BASE / "results/demo/rehearsal.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    run(parser.parse_args().api.rstrip("/"))
