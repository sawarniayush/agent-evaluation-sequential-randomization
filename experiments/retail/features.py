"""Retail history measurements and effect vector."""

import re


GROUPS = {
    "identity": {"find_user_id_by_email", "find_user_id_by_name_zip"},
    "user_read": {"get_user_details"},
    "order_read": {"get_order_details"},
    "catalog_read": {"get_product_details", "get_item_details", "list_all_product_types"},
    "pending_write": {
        "cancel_pending_order",
        "modify_pending_order_items",
        "modify_pending_order_address",
        "modify_pending_order_payment",
    },
    "resolution_write": {
        "return_delivered_order_items",
        "exchange_delivered_order_items",
        "modify_user_address",
    },
    "utility": {"calculate", "transfer_to_human_agents"},
}


def group_for(name):
    for group, names in GROUPS.items():
        if name in names:
            return group
    return "utility"


def history_features(messages, proposal, previous):
    """Only the visible prefix and previous assignments, never future outcomes."""
    calls = {}
    successes, failures = [], []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            calls[call.id] = call.name
        if message.role == "tool":
            name = calls.get(getattr(message, "id", ""), getattr(message, "name", "") or "unknown")
            content = (message.content or "").lower()
            failed = bool(getattr(message, "error", False)) or content.startswith("error")
            (failures if failed else successes).append(name)
    proposed = [c.name for c in proposal.tool_calls]
    features = {
        f"tool:{name}": float(name in proposed) for names in GROUPS.values() for name in names
    }
    for group in GROUPS:
        features[f"current:{group}"] = float(any(group_for(n) == group for n in proposed))
        features[f"used:{group}"] = float(any(group_for(n) == group for n in successes))
    features.update(
        prior_error=float(bool(failures)),
        retry_tool=float(any(n in successes + failures for n in proposed)),
        prior_verification=float(any(d["treatment"] for d in previous)),
        prior_correction=float(any(d.get("changed", False) for d in previous)),
        n_prior_tools=len(successes) + len(failures),
        n_prior_verifications=sum(d["treatment"] for d in previous),
        n_prior_errors=len(failures),
        user_affirmation_seen=float(
            any(
                m.role == "user"
                and re.search(r"\b(yes|confirm|go ahead|proceed)\b", m.content or "", re.I)
                for m in messages
            )
        ),
        n_proposed_tools=len(proposed),
    )
    return features


def feature_vector(raw):
    """The 24 effect features used by every Retail SNMM."""
    values = {"intercept": 1.0, **{k: float(v) for k, v in raw.items() if k.startswith("tool:")}}
    for name in (
        "prior_error",
        "retry_tool",
        "prior_correction",
        "used:order_read",
        "used:catalog_read",
    ):
        values[name] = float(raw[name])
    write = max(raw["current:pending_write"], raw["current:resolution_write"])
    values["write_x_affirmation"] = write * raw["user_affirmation_seen"]
    values["write_x_prior_error"] = write * raw["prior_error"]
    return values
