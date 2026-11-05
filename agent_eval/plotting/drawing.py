"""Shared drawing primitives for the paper figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as Line2D
from matplotlib.legend_handler import HandlerPatch
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from .data import TARGETS, BLUE, ORANGE, primary, valid

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
AUDIT = {}
OUTPUT = None


def comparison(rows, group):
    candidates = [
        r
        for r in rows
        if r["group_id"] == group and r["estimand"] == "blip_randomized" and r["method"] == "SNMM"
    ]
    assert len(candidates) == 1
    return candidates[0]


def save(fig, name):
    fig.savefig(OUTPUT / (name + ".pdf"))
    plt.close(fig)


AUDIT = {}

WIDE_ARROW_WIDTH_PT = 22.0

WIDE_ARROW_HEIGHT_PT = 7.0

WIDE_ARROW_HEAD_PT = 8.5

WIDE_ARROW_LINEWIDTH = 1.15

ESTIMAND_HEADERS = (
    "Blip with\nrandomized\ncontinuation\n" + r"$\tau(G;g)$",
    "Blip with\nno-treatment\ncontinuation\n" + r"$\tau(G;0)$",
    "Policy gain\n" + r"$\Delta(G)$",
)


def interval_extent(point, low, high, arrow_threshold=None):
    """Use the estimate, rather than an undisplayed interval, to set arrow axes."""
    return [point] if arrow_threshold is not None and high - low > arrow_threshold else [low, high]


def draw_interval(ax, point, low, high, ypos, color, marker, arrow_threshold=None, markersize=3):
    """A fixed-size double arrow denotes a wide CI; the dot stays at its estimate."""
    arrow = arrow_threshold is not None and high - low > arrow_threshold
    if arrow:
        symbol = DrawingArea(WIDE_ARROW_WIDTH_PT, WIDE_ARROW_HEIGHT_PT)
        middle = WIDE_ARROW_HEIGHT_PT / 2
        symbol.add_artist(
            FancyArrowPatch(
                (0, middle),
                (WIDE_ARROW_WIDTH_PT, middle),
                arrowstyle="<->",
                mutation_scale=WIDE_ARROW_HEAD_PT,
                shrinkA=0,
                shrinkB=0,
                color=color,
                linewidth=WIDE_ARROW_LINEWIDTH,
            )
        )
        ax.add_artist(
            AnnotationBbox(
                symbol, (point, ypos), frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=2
            )
        )
    else:
        ax.plot([low, high], [ypos, ypos], color=color, lw=1.05)
    ax.plot(point, ypos, marker, color=color, markersize=markersize, zorder=3)
    return "arrow" if arrow else "interval"


def wide_interval_legend(fig, threshold, unit, max_right, fontsize=8):
    """Explain schematic wide-CI arrows in the unused lower-left margin."""
    label = f"95% CI width > {threshold:g} {unit}"
    handle = FancyArrowPatch(
        (0, 0),
        (WIDE_ARROW_WIDTH_PT, 0),
        arrowstyle="<->",
        mutation_scale=WIDE_ARROW_HEAD_PT,
        shrinkA=0,
        shrinkB=0,
        linewidth=WIDE_ARROW_LINEWIDTH,
        color="#55616b",
        label=label,
    )

    def legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        middle = height / 2 - ydescent
        return FancyArrowPatch(
            (-xdescent, middle),
            (width - xdescent, middle),
            arrowstyle="<->",
            mutation_scale=WIDE_ARROW_HEAD_PT,
            shrinkA=0,
            shrinkB=0,
        )

    key = fig.legend(
        handles=[handle],
        handler_map={FancyArrowPatch: HandlerPatch(patch_func=legend_arrow)},
        loc="lower left",
        bbox_to_anchor=(0.008, 0.025 / fig.get_figheight()),
        frameon=False,
        fontsize=fontsize,
        handlelength=WIDE_ARROW_WIDTH_PT / fontsize,
        handletextpad=0.6,
        borderaxespad=0,
        borderpad=0,
    )
    fig.canvas.draw()
    bounds = key.get_window_extent(fig.canvas.get_renderer())
    assert bounds.x1 < fig.bbox.x0 + max_right * fig.bbox.width, label
    assert bounds.y0 >= fig.bbox.y0 and bounds.y1 <= fig.bbox.y1, label
    return dict(
        label=label,
        arrow_width_points=WIDE_ARROW_WIDTH_PT,
        arrow_head_points=WIDE_ARROW_HEAD_PT,
        location="lower left",
    )


def record(row):
    result = {
        k: row[k]
        for k in (
            "estimand",
            "group_id",
            "group_level",
            "method",
            "role",
            "estimate",
            "lower",
            "upper",
            "effect_unit",
            "matching_decisions",
            "treated_decisions",
            "skip_decisions",
        )
    }
    suffix = "points" if row["effect_unit"] == "score points" else "pp"
    result.update(
        {f"{field}_{suffix}": row[f"{field}_{suffix}"] for field in ("estimate", "lower", "upper")}
    )
    result.update(
        {
            k: row[k]
            for k in (
                "configuration_id",
                "prediction_basis",
                "effect_basis",
                "sharing_mode",
                "supported",
                "fit_status",
                "support_reason",
            )
            if k in row
        }
    )
    return result


def effect_label(rows):
    labels = {f"Effect on {r['outcome_label']} ({r['effect_unit']})" for r in rows}
    assert len(labels) == 1
    return labels.pop()


def short_unit(rows):
    return "score points" if rows[0]["effect_unit"] == "score points" else "pp"


def label_for(row, tools=False):
    label = row["group_label"]
    replacements = {
        "Product/catalog details": "Catalog details",
        "Account/user operations": "Account/user total",
        "Order operations": "Order total",
        "Location and external information": "Location / external info",
        "Shift dates and calculate time differences": "Time arithmetic",
        "Financial information and currency conversion": "Financial info / currency",
    }
    label = replacements.get(label, label)
    if row["group_id"] == "family:catalog":
        label = "Catalog / domain total"
    if tools:
        # Keep the exact API spelling, introducing only line breaks.
        parts = label.split("_")
        lines = [""]
        for part in parts:
            token = ("_" if lines[-1] else "") + part
            if lines[-1] and len(lines[-1] + token) > 24:
                lines.append("_" + part)
            else:
                lines[-1] += token
        return "\n".join(lines)
    return label


STEERING_MAIN_SECTIONS = (
    "Overall",
    "Checkpoint in the trajectory",
    "Visible progress",
    "Contradictory test feedback",
    "Earlier steering",
)

STEERING_SECTION_LABELS = {
    "Checkpoint in the trajectory": "Feedback timing",
    "Contradictory test feedback": "Contradictory test feedback",
}

STEERING_LABELS = {
    "checkpoint_1": "After failed attempt 1",
    "checkpoint_2": "After failed attempt 2",
    "checkpoint_3": "After failed attempt 3",
    "checkpoint_4": "After failed attempt 4",
    "checkpoint_5": "After failed attempt 5",
    "low": "Passing >0%, <50%",
    "middle": "Passing \u226550%, <80%",
    "near": "Passing \u226580%, <100%",
    "near_conflict": "Near completion + conflict",
    "conflict_lower": "Conflict, below 80% passing",
    "no_conflict": "No visible conflict",
    "prior_attack_resisted": "Steered before, still eligible",
}


def steering_forest(rows, name, sections):
    """Three steering targets by pre-assignment state; positive = more violations.

    Whole taxonomy sections are selected, never individual rows by estimate.
    Exact structural-zero policy gains are labeled rather than plotted.
    """
    groups = sorted(
        (r for r in rows if r["role"] == "primary" and r["estimand"] == "blip_randomized"),
        key=lambda r: int(r["taxonomy_order"]),
    )
    assert set(sections) <= {r["section"] for r in groups}
    groups = [r for r in groups if r["section"] in sections]
    lines, y = [], 0.0
    for row in groups:
        if not lines or lines[-1][1]["section"] != row["section"]:
            y += 0.25 if lines else 0
            lines.append(("header", row, y))
            y += 0.8
        lines.append(("group", row, y))
        y += 1
    height = 0.2 * y + 1.2
    fig = plt.figure(figsize=(6.5, height))
    left, right, gap = 0.45, 0.985, 0.025
    width = (right - left - 2 * gap) / 3
    bottom, top = 0.43 / height, 1 - 0.7 / height
    ylim = (y - 0.4, -0.6)
    label_ax = fig.add_axes([0.008, bottom, left - 0.02, top - bottom])
    label_ax.set(xlim=(0, 1), ylim=ylim)
    label_ax.axis("off")
    label_ax.text(
        0.99,
        1.0,
        "Matching decisions",
        transform=label_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#465260",
    )
    for kind, row, ypos in lines:
        if kind == "header":
            label_ax.text(
                0,
                ypos,
                STEERING_SECTION_LABELS.get(row["section"], row["section"]),
                va="center",
                fontsize=8.3,
                fontweight="bold",
            )
            continue
        label_ax.text(
            0.035,
            ypos,
            STEERING_LABELS.get(row["group_id"], row["group_label"]),
            va="center",
            fontsize=8,
        )
        counts = f"{int(row['matching_decisions']):,}"
        label_ax.text(0.99, ypos, counts, ha="right", va="center", fontsize=8, color="#465260")
    audit = {
        "kind": "steering three-target forest; positive = more violations",
        "sections": list(sections),
        "primary_rows": [],
        "comparison_rows": [],
        "structural_zero": [],
        "omitted": [],
    }
    for col, target in enumerate(TARGETS):
        ax = fig.add_axes([left + col * (width + gap), bottom, width, top - bottom])
        ax.set_ylim(*ylim)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#77818c")
        ax.axvline(0, color="#77818c", lw=0.8, zorder=0)
        ax.set_title(ESTIMAND_HEADERS[col], pad=8, fontsize=8.8)
        bounds = [0.0]
        for kind, row, ypos in lines:
            if kind == "header":
                continue
            r = primary(rows, target, row["group_id"])
            if target == "selective_policy" and r["exact_structural_zero"] == "True":
                ax.text(
                    0.5,
                    ypos,
                    "ineligible",
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#707780",
                )
                audit["structural_zero"].append(record(r))
                continue
            if not valid(r):
                reason = "unavailable"
                ax.text(
                    0.5,
                    ypos,
                    reason,
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#707780",
                )
                audit["omitted"].append({**record(r), "reason": r["support_reason"]})
                continue
            items = [r]
            if target == "blip_randomized":
                comp = comparison(rows, row["group_id"])
                if valid(comp):
                    items.append(comp)
            for k, item in enumerate(items):
                offset = (-0.14 if k == 0 else 0.14) if len(items) == 2 else 0
                point, lo, hi = [float(item[f]) for f in ("estimate", "lower", "upper")]
                color = BLUE if item["method"] == "AIPW" else ORANGE
                ax.plot([lo, hi], [ypos + offset] * 2, lw=1.1, color=color)
                ax.plot(
                    point,
                    ypos + offset,
                    "o" if item["method"] == "AIPW" else "s",
                    color=color,
                    markersize=3,
                )
                bounds += [lo, hi]
                audit["primary_rows" if k == 0 else "comparison_rows"].append(record(item))
        span = max(bounds) - min(bounds)
        ax.set_xlim(min(bounds) - 0.09 * span, max(bounds) + 0.09 * span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.tick_params(axis="x", labelsize=8, length=3, pad=2)
    fig.text(
        0.69,
        0.03,
        "Effect on terminal detected violations (pp; positive means more)",
        ha="center",
        fontsize=8,
    )
    save(fig, name)
    AUDIT[name] = audit
