"""ToolSandbox capability, operation and tool forests."""

from . import drawing as figures
from .data import TARGETS, BLUE, ORANGE, metadata, primary, valid
from .hierarchies import TOOLSANDBOX_TOOL_BLOCKS, ARROW_THRESHOLD, plotted, zoomed_axes

plt = figures.plt


PANELS = (
    ("a", ("device_state", "contacts")),
    ("b", ("messaging", "reminders")),
    ("c", ("time_calculations", "location_information")),
)


LABELS = {
    "domain:location_information": "Location / external information",
    "operation:find_person": "Find a person",
    "operation:change_remove_contact": "Change / remove contacts",
    "operation:change_remove_reminders": "Change / remove reminders",
    "operation:time_arithmetic": "Time arithmetic",
    "operation:financial_information": "Financial information",
}


HEADERS = figures.ESTIMAND_HEADERS


def reportable(row):
    """Display finite estimates below the count screen.

    Preserve the source support flag. Unobserved groups and failed model fits
    remain unavailable; this changes presentation, not estimation or inference.
    """
    if valid(row):
        return True
    if (
        row.get("support_reason") != "insufficient_group_support"
        or row.get("fit_status") != "estimated"
        or min(int(row[k]) for k in ("matching_decisions", "treated_decisions", "skip_decisions"))
        <= 0
    ):
        return False
    return valid({**row, "supported": "True"})


def hierarchy_lines(study, rows):
    specification = metadata(study)
    groups = {g["group_id"]: g for g in specification["groups"]}
    ranks = {}
    for _, sections in TOOLSANDBOX_TOOL_BLOCKS:
        for _, tools in sections:
            for tool, _ in tools:
                key = "tool:" + tool
                ranks[key] = len(ranks)
    result = []

    def visit(key):
        group = groups[key]
        children = [g["group_id"] for g in groups.values() if g["parent_group_id"] == key]
        depth = {"domain": 0, "family": 1, "tool": 2}[group["level"]]
        if depth == 1:
            children.sort(key=ranks.__getitem__)
        alias = children[0] if depth == 1 and len(children) == 1 else None
        if alias:
            assert groups[alias]["alias_of"] == key
            assert groups[alias]["tools"] == group["tools"]
            alias_fields = ("estimate", "lower", "upper", "matching_decisions", "supported")
            for target in TARGETS:
                family, tool = primary(rows, target, key), primary(rows, target, alias)
                assert all(family[k] == tool[k] for k in alias_fields)
            family, tool = figures.comparison(rows, key), figures.comparison(rows, alias)
            assert all(family[k] == tool[k] for k in alias_fields)
        assert len({primary(rows, target, key)["matching_decisions"] for target in TARGETS}) == 1
        if children:
            assert set(group["tools"]) == set().union(*(set(groups[c]["tools"]) for c in children))
        # Singleton operations stand in for their tool, so show the API name
        # there as well as on every explicit tool row.
        tool_key = alias or (key if depth == 2 else None)
        label = tool_key.removeprefix("tool:") if tool_key else LABELS.get(key, group["label"])
        result.append(
            dict(
                group=key,
                label=label,
                depth=depth,
                parent=group["parent_group_id"],
                alias=alias,
                tools=group["tools"],
                children=[] if alias else children,
            )
        )
        if not alias:
            for child in children:
                visit(child)

    for _, domains in PANELS:
        for domain in domains:
            visit("domain:" + domain)
    assert len(result) == len({item["group"] for item in result}) == 49
    represented = [
        item["alias"] or item["group"] for item in result if item["alias"] or item["depth"] == 2
    ]
    assert represented == list(ranks) and len(represented) == 33
    return result


def forest(study, rows, lines, axes, name):
    positions, y = {}, 0.0
    for item in lines:
        if item["depth"] == 0 and positions:
            y += 0.48
        positions[item["group"]] = y
        y += 1
    height = 0.185 * y + 1.05
    fig = plt.figure(figsize=(6.5, height))
    bottom, top = 0.40 / height, 1 - 0.65 / height
    left, right, gap = 0.425, 0.992, 0.026
    column_width = (right - left - 2 * gap) / 3
    ylim = (y - 0.40, -0.60)
    labels = fig.add_axes([0.008, bottom, left - 0.032, top - bottom])
    label_width = (left - 0.032) * 6.5
    labels.set(xlim=(0, label_width), ylim=ylim)
    labels.axis("off")
    labels.text(
        0.015,
        1.035,
        "Capability → operation → tool",
        fontsize=7.6,
        transform=labels.transAxes,
        color="#465260",
        va="bottom",
    )
    labels.text(
        0.995,
        1.035,
        r"$N(G)$",
        fontsize=8,
        transform=labels.transAxes,
        color="#465260",
        ha="right",
        va="bottom",
    )
    audit = dict(
        kind="three-target indented hierarchy",
        study=study,
        axis_scope="agent",
        arrow_width_threshold=ARROW_THRESHOLD,
        arrow_edge_margin_fraction=None,
        xlims=axes,
        row_order=[item["group"] for item in lines],
        row_levels={item["group"]: item["depth"] for item in lines},
        row_labels={item["group"]: item["label"] for item in lines},
        tool_label_style="api_name",
        row_tools={item["group"]: item["tools"] for item in lines},
        aliases=[dict(group=item["group"], tool=item["alias"]) for item in lines if item["alias"]],
        parent_child_edges=[],
        primary_rows=[],
        comparison_rows=[],
        omitted=[],
        total_episodes=int(rows[0]["total_episodes"]),
        width_inches=6.5,
        height_inches=height,
    )
    audit["minimum_count_screen_waived"] = True
    audit["reporting_rule"] = (
        "Report finite estimates with observed assignments in both arms, "
        "including groups below the original count thresholds. "
        "Keep source support flags, failed-fit exclusions and unobserved groups."
    )
    text_pairs = []
    for item in lines:
        key, depth = item["group"], item["depth"]
        cy = positions[key]
        row = primary(rows, TARGETS[0], key)
        if depth == 0:
            labels.axhspan(cy - 0.43, cy + 0.43, color="#edf2f5", zorder=0)
        if item["children"]:
            children = item["children"]
            rail_x = 0.08 if depth == 0 else 0.235
            end_x = 0.145 if depth == 0 else 0.30
            labels.plot(
                [rail_x, rail_x],
                [cy + 0.34, positions[children[-1]]],
                color="#aab8c1",
                lw=0.65,
                zorder=1,
            )
            for child in children:
                labels.plot(
                    [rail_x, end_x], [positions[child]] * 2, color="#aab8c1", lw=0.65, zorder=1
                )
                audit["parent_child_edges"].append([key, child])
        x = (0.015, 0.175, 0.33)[depth]
        is_tool = depth == 2 or item["alias"] is not None
        label = labels.text(
            x,
            cy,
            item["label"],
            va="center",
            fontsize=7.2 if is_tool else 7.7 if depth == 0 else 7.4,
            fontfamily="DejaVu Sans Mono" if is_tool else "DejaVu Sans",
            fontweight="bold" if depth == 0 else "normal",
            color="#17212b" if reportable(row) else "#7b838a",
            zorder=2,
        )
        count = labels.text(
            label_width - 0.015,
            cy,
            f"{int(row['matching_decisions']):,}",
            ha="right",
            va="center",
            fontsize=7.3,
            color="#465260",
            zorder=2,
        )
        text_pairs.append((label, count))
    for col, target in enumerate(TARGETS):
        ax = fig.add_axes([left + col * (column_width + gap), bottom, column_width, top - bottom])
        ax.set(ylim=ylim, xlim=axes[target], yticks=[])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#77818c")
        ax.axvline(0, color="#89959f", lw=0.75, zorder=1)
        ax.set_title(HEADERS[col], fontsize=8, pad=8)
        ax.xaxis.set_major_locator(figures.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="x", labelsize=7.5, length=3, pad=3)
        for item in lines:
            key, cy = item["group"], positions[item["group"]]
            if item["depth"] == 0:
                ax.axhspan(cy - 0.43, cy + 0.43, color="#edf2f5", zorder=0)
            row = primary(rows, target, key)
            results = plotted(rows, target, key, reportable)
            if not results:
                ax.text(
                    0.5,
                    cy,
                    "n/a",
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    color="#7b838a",
                )
                audit["omitted"].append(
                    {**figures.record(row), "reason": row.get("support_reason") or "unavailable"}
                )
                if target == TARGETS[0]:
                    other = figures.comparison(rows, key)
                    audit["omitted"].append(
                        {
                            **figures.record(other),
                            "reason": other.get("support_reason") or "unavailable",
                        }
                    )
            for index, result in enumerate(results):
                offset = (-0.15 if index == 0 else 0.15) if len(results) == 2 else 0
                point, low, high = (float(result[k]) for k in ("estimate", "lower", "upper"))
                color = BLUE if result["method"] == "AIPW" else ORANGE
                display = figures.draw_interval(
                    ax,
                    point,
                    low,
                    high,
                    cy + offset,
                    color,
                    "o" if result["method"] == "AIPW" else "s",
                    ARROW_THRESHOLD,
                    markersize=3.1 if item["depth"] == 0 else 2.8,
                )
                entry = {**figures.record(result), "interval_display": display}
                if not valid(result):
                    entry["reporting_override"] = "minimum_group_count_rule"
                audit["primary_rows" if result["role"] == "primary" else "comparison_rows"].append(
                    entry
                )
    handles = [
        figures.Line2D([], [], color=BLUE, marker="o", lw=1, markersize=3, label="AIPW"),
        figures.Line2D([], [], color=ORANGE, marker="s", lw=1, markersize=3, label="SNMM"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.008, 1),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handlelength=1.3,
        borderaxespad=0,
    )
    fig.text(
        (left + right) / 2,
        0.065 / height,
        "Effect on similarity (score points)",
        fontsize=7.6,
        ha="center",
    )
    if any(
        r["interval_display"] == "arrow"
        for key in ("primary_rows", "comparison_rows")
        for r in audit[key]
    ):
        audit["wide_interval_legend"] = figures.wide_interval_legend(
            fig, ARROW_THRESHOLD, "score points", left - 0.02, fontsize=7.4
        )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label, count in text_pairs:
        assert label.get_window_extent(renderer).x1 + 2 < count.get_window_extent(renderer).x0, (
            label.get_text()
        )
    assert len(audit["primary_rows"]) + sum(
        r["role"] == "primary" for r in audit["omitted"]
    ) == 3 * len(lines)
    assert len(audit["comparison_rows"]) + sum(
        r["role"] == "comparison" for r in audit["omitted"]
    ) == len(lines)
    figures.save(fig, name)
    figures.AUDIT[name] = audit


def render_hierarchy_forests(study, rows, panels=None):
    lines = hierarchy_lines(study, rows)
    axes = zoomed_axes(study, rows, lines, reportable)
    names = []
    for suffix, domains in PANELS:
        selected = [
            item
            for item in lines
            if any(
                item["group"] == "domain:" + domain
                or primary(rows, TARGETS[0], item["group"])["domain_id"] == "domain:" + domain
                for domain in domains
            )
        ]
        name = study + "_hierarchy_" + suffix
        if panels is None or suffix in panels:
            forest(study, rows, selected, axes, name)
        else:
            previous = figures.AUDIT[name]
            assert previous["row_order"] == [item["group"] for item in selected]
            assert previous["xlims"] == {key: list(value) for key, value in axes.items()}, name
        names.append(name)
    displayed = [key for name in names for key in figures.AUDIT[name]["row_order"]]
    assert displayed == [item["group"] for item in lines]
    return names
