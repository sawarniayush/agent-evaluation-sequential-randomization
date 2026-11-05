"""Read figure inputs from the included result tables."""

import csv
import json
import math

from ..analysis import ROOT

TARGETS = ("blip_randomized", "blip_no_future", "selective_policy")
BLUE, ORANGE = "#0072B2", "#D55E00"
STUDY_LABELS = {
    "retail": "Retail: GPT-5 nano",
    "retail_haiku": "Retail: Claude Haiku 5.5",
    "toolsandbox": "ToolSandbox: GPT-5 nano",
    "toolsandbox_haiku": "ToolSandbox: Claude Haiku 5.5",
}
NAMES = {
    "retail": "retail_nano",
    "retail_haiku": "retail_haiku",
    "toolsandbox": "toolsandbox_nano",
    "toolsandbox_haiku": "toolsandbox_haiku",
    "steering_qwen": "impossiblebench",
}


def load(study):
    with (ROOT / "data" / NAMES[study] / "plot_rows.csv").open() as stream:
        return list(csv.DictReader(stream))


def metadata(study):
    return json.loads((ROOT / "data" / NAMES[study] / "metadata.json").read_text())


def primary(rows, target, group):
    found = [
        r
        for r in rows
        if r["role"] == "primary" and r["estimand"] == target and r["group_id"] == group
    ]
    assert len(found) == 1, (target, group, len(found))
    return found[0]


def valid(row):
    if row["supported"].lower() != "true" or row["fit_status"] != "estimated":
        return False
    try:
        point, low, high = [float(row[k]) for k in ("estimate", "lower", "upper")]
        return all(map(math.isfinite, (point, low, high))) and low <= high
    except (TypeError, ValueError):
        return False
