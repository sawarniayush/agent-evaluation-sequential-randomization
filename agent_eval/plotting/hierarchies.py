"""Retail hierarchies and direct-policy comparison."""

import math
from . import drawing as figures
from .data import TARGETS, BLUE, ORANGE, STUDY_LABELS, load, metadata, primary, valid

plt = figures.plt


MC_MODEL = "retail_snmm"


ARROW_THRESHOLD = 30.0


HIERARCHY_LEFT, HIERARCHY_RIGHT, HIERARCHY_GAP = 0.455, 0.988, 0.026


HIERARCHY_COLUMN_WIDTH = (HIERARCHY_RIGHT - HIERARCHY_LEFT - 2 * HIERARCHY_GAP) / 3


ARROW_EDGE_MARGIN = (figures.WIDE_ARROW_WIDTH_PT / 2 + 1) / (6.5 * 72 * HIERARCHY_COLUMN_WIDTH)


HEADERS = figures.ESTIMAND_HEADERS


FAMILIES = {
    "reads": ("identity", "account_details", "order_details", "catalog"),
    "writes": ("account_changes", "pending_orders", "returns"),
}


TOOL_LABELS = {
    "find_user_id_by_email": "Find by email",
    "find_user_id_by_name_zip": "Find by name / ZIP",
    "get_user_details": "Get user details",
    "get_order_details": "Get order details",
    "get_item_details": "Get item details",
    "get_product_details": "Get product details",
    "list_all_product_types": "List product types",
    "modify_user_address": "Change user address",
    "cancel_pending_order": "Cancel order",
    "modify_pending_order_address": "Change address",
    "modify_pending_order_items": "Change items",
    "modify_pending_order_payment": "Change payment",
    "return_delivered_order_items": "Return items",
    "exchange_delivered_order_items": "Exchange items",
}


TOOLSANDBOX_TOOL_BLOCKS = (
    (
        "a",
        (
            (
                "Device state",
                (
                    ("get_cellular_service_status", "Read cellular status"),
                    ("set_cellular_service_status", "Change cellular status"),
                    ("get_wifi_status", "Read Wi-Fi status"),
                    ("set_wifi_status", "Change Wi-Fi status"),
                    ("get_location_service_status", "Read location access"),
                    ("set_location_service_status", "Change location access"),
                    ("get_low_battery_mode_status", "Read low-battery mode"),
                    ("set_low_battery_mode_status", "Change low-battery mode"),
                ),
            ),
            (
                "Contacts",
                (
                    ("search_contacts", "Find a contact"),
                    ("add_contact", "Add a contact"),
                    ("modify_contact", "Modify a contact"),
                    ("remove_contact", "Remove a contact"),
                ),
            ),
            (
                "Messaging",
                (
                    ("search_messages", "Find previous messages"),
                    ("send_message_with_phone_number", "Send a message"),
                ),
            ),
        ),
    ),
    (
        "b",
        (
            (
                "Reminders",
                (
                    ("search_reminder", "Find reminders"),
                    ("add_reminder", "Create a reminder"),
                    ("modify_reminder", "Modify a reminder"),
                    ("remove_reminder", "Remove a reminder"),
                ),
            ),
            (
                "Time and calculations",
                (
                    ("get_current_timestamp", "Read current time"),
                    ("datetime_info_to_timestamp", "Construct a timestamp"),
                    ("timestamp_to_datetime_info", "Interpret a timestamp"),
                    ("shift_timestamp", "Shift a date or time"),
                    ("timestamp_diff", "Calculate a time difference"),
                    ("seconds_to_hours_minutes_seconds", "Convert seconds to time"),
                    ("unit_conversion", "Convert units"),
                    ("search_holiday", "Find holidays"),
                ),
            ),
            (
                "Location and external information",
                (
                    ("get_current_location", "Read current location"),
                    ("search_lat_lon", "Find coordinates"),
                    ("search_location_around_lat_lon", "Search nearby places"),
                    ("search_weather_around_lat_lon", "Retrieve weather"),
                    ("convert_currency", "Convert currency"),
                    ("search_stock", "Retrieve a stock price"),
                    ("calculate_lat_lon_distance", "Calculate distance"),
                ),
            ),
        ),
    ),
)


def retail_lines(study):
    specification = metadata(study)
    groups = {g["group_id"]: g for g in specification["groups"]}
    result = []
    covered = set()
    for activity, families in FAMILIES.items():
        result.append({"group": "activity:" + activity, "label": activity.title(), "depth": 0})
        union = set()
        for family in families:
            group = "family:" + family
            tools = sorted(groups[group]["tools"])
            union.update(tools)
            label = groups[group]["label"].replace("Product/catalog details", "Catalog details")
            item = {"group": group, "label": label, "depth": 1}
            if len(tools) == 1:
                item["alias"] = "tool:" + tools[0]
                item["sublabel"] = TOOL_LABELS[tools[0]]
            result.append(item)
            if len(tools) > 1:
                result.extend(
                    {"group": "tool:" + tool, "label": TOOL_LABELS[tool], "depth": 2}
                    for tool in tools
                )
            covered.update(tools)
        assert union == set(groups["activity:" + activity]["tools"])
    assert covered == set(TOOL_LABELS) and len(result) == 20
    return result


def shown(row):
    return valid(row)


def plotted(rows, target, group, row_filter=shown):
    row = primary(rows, target, group)
    if not row_filter(row):
        return []
    result = [row]
    if target == "blip_randomized":
        other = figures.comparison(rows, group)
        if row_filter(other):
            result.append(other)
    return result


def limits(values, padding_fraction=0.12, minimum_padding=0.5):
    low, high = min([0.0] + values), max([0.0] + values)
    pad = max(minimum_padding, padding_fraction * (high - low))
    return low - pad, high + pad


def common_axes(studies, line_map, padding_fraction=0.12, minimum_padding=0.5, row_filter=shown):
    return {
        target: limits(
            [
                value
                for study, rows in studies.items()
                for item in line_map[study]
                for r in plotted(rows, target, item["group"], row_filter)
                for value in figures.interval_extent(
                    float(r["estimate"]), float(r["lower"]), float(r["upper"]), ARROW_THRESHOLD
                )
            ],
            padding_fraction,
            minimum_padding,
        )
        for target in TARGETS
    }


def zoomed_axes(study, rows, lines, row_filter=shown):
    axes = common_axes({study: rows}, {study: lines}, 0.06, 0.1, row_filter)
    for target, (low, high) in axes.items():
        arrows = [
            float(r["estimate"])
            for item in lines
            for r in plotted(rows, target, item["group"], row_filter)
            if float(r["upper"]) - float(r["lower"]) > ARROW_THRESHOLD
        ]
        if arrows:
            # A fixed-size arrow needs room beyond its center, even on tight axes.
            margin = ARROW_EDGE_MARGIN
            for _ in range(20):
                previous = (low, high)
                low = min(low, (min(arrows) - margin * high) / (1 - margin))
                high = max(high, (max(arrows) - margin * low) / (1 - margin))
                if max(abs(low - previous[0]), abs(high - previous[1])) < 1e-12:
                    break
        axes[target] = (low, high)
    return axes


def hierarchy_forest(study, rows, lines, axes, name, zoom=False):
    retail = study.startswith("retail")
    # Typography is specified at the final manuscript width, without tight cropping.
    height = 7.0 if retail else 7.45
    fig = plt.figure(figsize=(6.5, height))
    bottom, top = 0.54 / height, 1 - 0.87 / height
    left, right, gap = HIERARCHY_LEFT, HIERARCHY_RIGHT, HIERARCHY_GAP
    width = HIERARCHY_COLUMN_WIDTH
    positions, y = [], 0.0
    for item in lines:
        if item["depth"] == 0 and positions:
            y += 0.28
        positions.append(y)
        y += 1.25 if item.get("sublabel") else 1.0
    ylim = (y - 0.35, -0.65)
    labels = fig.add_axes([0.008, bottom, left - 0.027, top - bottom])
    labels.set(xlim=(0, 1), ylim=ylim)
    labels.axis("off")
    labels.text(
        0.99,
        1.025,
        "Matching decisions",
        transform=labels.transAxes,
        ha="right",
        fontsize=8,
        color="#465260",
    )
    audit = {
        "kind": "three-target hierarchy",
        "axis_scope": "actor" if zoom else "shared",
        "arrow_edge_margin_fraction": ARROW_EDGE_MARGIN if zoom else None,
        "arrow_width_threshold": ARROW_THRESHOLD,
        "study": study,
        "xlims": axes,
        "row_order": [item["group"] for item in lines],
        "aliases": [],
        "primary_rows": [],
        "comparison_rows": [],
        "omitted": [],
        "total_episodes": int(rows[0]["total_episodes"]),
    }
    for item, ypos in zip(lines, positions):
        row = primary(rows, "blip_randomized", item["group"])
        strong = item["depth"] == 0 or retail and item["depth"] == 1
        labels.text(
            0.025 * item["depth"],
            ypos,
            item["label"],
            va="center",
            fontsize=8,
            fontweight="bold" if strong else "normal",
        )
        if item.get("sublabel"):
            labels.text(
                0.025 * item["depth"] + 0.015,
                ypos + 0.42,
                item["sublabel"],
                va="center",
                fontsize=7.5,
                color="#465260",
            )
            for target in TARGETS:
                family = primary(rows, target, item["group"])
                tool = primary(rows, target, item["alias"])
                assert all(
                    math.isclose(float(family[k]), float(tool[k]), abs_tol=1e-9)
                    for k in ("estimate", "lower", "upper", "matching_decisions")
                )
            audit["aliases"].append({"group": item["group"], "tool": item["alias"]})
        labels.text(
            0.99,
            ypos,
            f"{int(row['matching_decisions']):,}",
            ha="right",
            va="center",
            fontsize=8,
            color="#465260",
        )
    for col, target in enumerate(TARGETS):
        ax = fig.add_axes([left + col * (width + gap), bottom, width, top - bottom])
        ax.set(ylim=ylim, xlim=axes[target], yticks=[])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#77818c")
        ax.axvline(0, color="#77818c", lw=0.8, zorder=0)
        ax.set_title(HEADERS[col], fontsize=8.1, pad=10)
        ax.xaxis.set_major_locator(
            figures.MaxNLocator(nbins=6 if retail else 5, steps=[1, 2, 5, 10])
            if zoom
            else figures.MaxNLocator(nbins=3, min_n_ticks=3)
        )
        ax.tick_params(axis="x", length=3, pad=3, labelsize=8)
        for item, ypos in zip(lines, positions):
            row = primary(rows, target, item["group"])
            items = plotted(rows, target, item["group"])
            if not items:
                reason = "unavailable"
                ax.text(
                    0.5,
                    ypos,
                    reason,
                    transform=ax.get_yaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="#707780",
                )
                audit["omitted"].append({**figures.record(row), "reason": reason})
                if target == "blip_randomized":
                    other = figures.comparison(rows, item["group"])
                    audit["omitted"].append(
                        {**figures.record(other), "reason": "primary marker withheld"}
                    )
                continue
            for index, row in enumerate(items):
                offset = (-0.14 if index == 0 else 0.14) if len(items) == 2 else 0
                point, low, high = (float(row[k]) for k in ("estimate", "lower", "upper"))
                color = BLUE if row["method"] == "AIPW" else ORANGE
                display = figures.draw_interval(
                    ax,
                    point,
                    low,
                    high,
                    ypos + offset,
                    color,
                    "o" if row["method"] == "AIPW" else "s",
                    ARROW_THRESHOLD,
                    markersize=3.2,
                )
                audit["primary_rows" if index == 0 else "comparison_rows"].append(
                    {**figures.record(row), "interval_display": display}
                )
            if target == "blip_randomized" and len(items) == 1:
                other = figures.comparison(rows, item["group"])
                audit["omitted"].append(
                    {**figures.record(other), "reason": "comparison unavailable"}
                )
    title = STUDY_LABELS[study].split(": ", 1)[1]
    fig.text(0.008, 1 - 0.19 / height, title, fontsize=10, fontweight="bold", va="center")
    handles = [
        figures.Line2D([], [], color=BLUE, marker="o", lw=1, label="AIPW"),
        figures.Line2D([], [], color=ORANGE, marker="s", lw=1, label="SNMM"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 1 - 0.025 / height),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        borderpad=0,
    )
    fig.text((left + right) / 2, 0.08 / height, figures.effect_label(rows), ha="center", fontsize=8)
    if any(
        r["interval_display"] == "arrow"
        for key in ("primary_rows", "comparison_rows")
        for r in audit[key]
    ):
        audit["wide_interval_legend"] = figures.wide_interval_legend(
            fig, ARROW_THRESHOLD, "pp" if retail else "score points", left - 0.02
        )
    assert (
        len(audit["primary_rows"]) + sum(r["role"] == "primary" for r in audit["omitted"])
        == len(lines) * 3
    )
    figures.save(fig, name)
    figures.AUDIT[name] = audit


def mc_forest(rows, metrics):
    retail = load("retail")  # Include the two MC-only utility policies.
    selected = [r for r in rows if r["policy"] != "never"]
    assert len(selected) == 9
    for row in selected:
        source = primary(retail, "selective_policy", "family:" + row["policy"])
        for mc_key, source_key in (
            ("snmm_gain_pp", "estimate"),
            ("snmm_gain_low_pp", "lower"),
            ("snmm_gain_high_pp", "upper"),
        ):
            assert math.isclose(float(row[mc_key]), float(source[source_key]), abs_tol=1e-9)
        assert row["matching_decisions"] == source["matching_decisions"]
        assert math.isclose(
            float(row["snmm_value_pct"]),
            100 * float(source["target_value_probability"]),
            abs_tol=1e-9,
        )
    rmse = math.sqrt(sum(float(r["mc_minus_snmm_gain_pp"]) ** 2 for r in selected) / len(selected))
    assert math.isclose(rmse, float(metrics["gain_rmse_pp"]), abs_tol=1e-9)
    height = 3.7
    fig = plt.figure(figsize=(6.5, height))
    bottom, top = 0.52 / height, 1 - 0.48 / height
    labels = fig.add_axes([0.008, bottom, 0.46, top - bottom])
    ax = fig.add_axes([0.505, bottom, 0.48, top - bottom])
    labels.set(xlim=(0, 1), ylim=(8.6, -0.7))
    labels.axis("off")
    labels.text(
        0.99,
        1.05,
        "Matching decisions",
        transform=labels.transAxes,
        ha="right",
        fontsize=8,
        color="#465260",
    )
    ax.set(ylim=(8.6, -0.7), yticks=[])
    audit = {
        "kind": "direct policy gains versus selected SNMM",
        "model": MC_MODEL,
        "arrow_width_threshold": ARROW_THRESHOLD,
        "interval_displays": [],
        "mc_episodes_per_policy": 456,
        "mc_total_episodes": 4560,
        "mc_rounds": [1, 2],
        "mc_degrees_of_freedom": 4559,
        "randomized_episodes": 2280,
        "interval_type": "pointwise_95_trajectory_t",
        "policy_rows": selected,
        "metrics": metrics,
        "gain_rmse_recomputed_pp": rmse,
        "source_agreement_tolerance": 1e-9,
    }
    bounds = []
    for ypos, row in enumerate(selected):
        labels.text(0, ypos, row["label"], va="center", fontsize=8.5)
        labels.text(
            0.99,
            ypos,
            f"{int(row['matching_decisions']):,}",
            va="center",
            ha="right",
            fontsize=8,
            color="#465260",
        )
        for prefix, offset, color, marker in [
            ("mc", -0.15, "#323e48", "D"),
            ("snmm", 0.15, ORANGE, "s"),
        ]:
            point, low, high = (
                float(row[f"{prefix}_gain{suffix}_pp"]) for suffix in ("", "_low", "_high")
            )
            display = figures.draw_interval(
                ax, point, low, high, ypos + offset, color, marker, ARROW_THRESHOLD, markersize=3.5
            )
            bounds.extend(figures.interval_extent(point, low, high, ARROW_THRESHOLD))
            audit["interval_displays"].append(
                {"policy": row["policy"], "series": prefix, "interval_display": display}
            )
    ax.set_xlim(limits(bounds))
    audit["xlim"] = list(ax.get_xlim())
    ax.axvline(0, color="#77818c", lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(figures.MaxNLocator(5))
    ax.tick_params(axis="x", labelsize=8)
    handles = [
        figures.Line2D([], [], color="#323e48", marker="D", lw=1, label="Direct rollouts"),
        figures.Line2D([], [], color=ORANGE, marker="s", lw=1, label="SNMM"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.995),
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.4,
    )
    fig.text(0.73, 0.065 / height, "Policy gain (percentage points)", ha="center", fontsize=8)
    figures.save(fig, "retail_mc_policy_gains")
    figures.AUDIT["retail_mc_policy_gains"] = audit
