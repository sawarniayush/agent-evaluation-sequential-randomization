"""Regenerate the ten result figures included in the paper."""

import csv

from .analysis import ROOT
from .plotting import drawing, hierarchies, toolsandbox
from .plotting.data import load


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    drawing.OUTPUT = output
    for study in ("retail", "retail_haiku"):
        rows = load(study)
        lines = hierarchies.retail_lines(study)
        axes = hierarchies.zoomed_axes(study, rows, lines)
        hierarchies.hierarchy_forest(
            study, rows, lines, axes, study + "_hierarchy_targets", zoom=True
        )
    for study in ("toolsandbox", "toolsandbox_haiku"):
        toolsandbox.render_hierarchy_forests(study, load(study))
    drawing.steering_forest(
        load("steering_qwen"), "steering_state_targets", drawing.STEERING_MAIN_SECTIONS
    )
    with (ROOT / "data/retail_mc/estimates.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    hierarchies.mc_forest(rows)
    print(f"Wrote ten result figures to {output}")
