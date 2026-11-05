"""ToolSandbox features from the state before proposal execution."""

import json


FAMILIES = {
    "settings_read": (
        "get_low_battery_mode_status",
        "get_location_service_status",
        "get_cellular_service_status",
        "get_wifi_status",
        "get_current_location",
    ),
    "settings_write": (
        "set_low_battery_mode_status",
        "set_location_service_status",
        "set_cellular_service_status",
        "set_wifi_status",
    ),
    "contact_lookup": ("search_contacts",),
    "contact_edit": ("add_contact", "modify_contact", "remove_contact"),
    "message_search": ("search_messages",),
    "message_send": ("send_message_with_phone_number",),
    "reminder_read": ("search_reminder",),
    "reminder_write": ("add_reminder", "modify_reminder", "remove_reminder"),
    "utility": (
        "get_current_timestamp",
        "timestamp_to_datetime_info",
        "datetime_info_to_timestamp",
        "shift_timestamp",
        "timestamp_diff",
        "seconds_to_hours_minutes_seconds",
        "unit_conversion",
        "calculate_lat_lon_distance",
        "search_holiday",
    ),
    "web_search": (
        "search_lat_lon",
        "search_location_around_lat_lon",
        "search_weather_around_lat_lon",
        "search_stock",
        "convert_currency",
    ),
}


FAMILY_OF = {tool: family for family, tools in FAMILIES.items() for tool in tools}


TOOLS = tuple(FAMILY_OF)


SERVICES = {
    "set_wifi_status": "wifi",
    "set_cellular_service_status": "cellular",
    "set_location_service_status": "location_service",
}


STATE_ERRORS = ("ConnectionError", "PermissionError")


def arguments(call):
    try:
        value = json.loads(call["arguments"])
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def precondition_unmet(name, args, settings):
    """Whether a call would fail on device state alone, by the rules in tool_sandbox/tools."""
    args = args or {}
    if name == "send_message_with_phone_number":
        return not settings["cellular"]
    if name in FAMILIES["web_search"]:
        if not settings["wifi"]:
            return True
        coordinates = name in ("search_location_around_lat_lon", "search_weather_around_lat_lon")
        return coordinates and args.get("latitude") is None and not settings["location_service"]
    if name == "get_current_location":
        return not settings["location_service"]
    if name == "set_low_battery_mode_status":
        return settings["low_battery_mode"] == args.get("on")
    if name in SERVICES:
        on = args.get("on")
        return settings[SERVICES[name]] == on or bool(on and settings["low_battery_mode"])
    return False


def features(state, calls, decisions):
    """Only the visible past, device state before execution, and earlier assignments."""
    names = [state["tool_names"].get(call["name"], call["name"]) for call in calls]
    parsed = [arguments(call) for call in calls]
    results = state["tool_results"]
    traced = [c["tool_name"] for r in results for c in r["calls"]]
    succeeded = [c["tool_name"] for r in results if not r["exception"] for c in r["calls"]]
    errors = [r["exception"] for r in results if r["exception"]]
    settings = state["settings"]
    value = {f"tool:{tool}": float(tool in names) for tool in TOOLS}
    for family in FAMILIES:
        value[f"current:{family}"] = float(any(FAMILY_OF.get(n) == family for n in names))
        value[f"used:{family}"] = float(any(FAMILY_OF.get(n) == family for n in succeeded))
    value.update(
        unknown_tool=float(any(n not in FAMILY_OF for n in names)),
        malformed_arguments=float(any(p is None for p in parsed)),
        precondition_unmet=float(
            any(precondition_unmet(n, p, settings) for n, p in zip(names, parsed))
        ),
        prior_error=float(bool(errors)),
        prior_state_error=float(any(e.startswith(STATE_ERRORS) for e in errors)),
        n_prior_errors=len(errors),
        retry_tool=float(any(n in traced for n in names)),
        prior_verification=float(any(d["treatment"] for d in decisions)),
        prior_correction=float(any(d["changed"] for d in decisions)),
        n_prior_tools=len(traced),
        n_prior_verifications=sum(d["treatment"] for d in decisions),
        n_proposed_tools=len(names),
        n_user_turns=state["user_turns"],
        **{
            f"setting:{k}": float(bool(settings[k]))
            for k in ("cellular", "wifi", "location_service", "low_battery_mode")
        },
        **{f"rows:{k}": v for k, v in state["database_rows"].items()},
    )
    return value
