"""Direct evaluation of the nine Retail policies and never-review reference."""

import json

import numpy as np
import pandas as pd

from .analysis import ROOT
from .inference import interval


def reproduce(output):
    folder = ROOT / "data/retail_mc"
    episodes = pd.read_csv(folder / "episodes.csv")
    metadata = json.loads((folder / "metadata.json").read_text())
    expected = pd.read_csv(folder / "estimates.csv").set_index("policy")
    assert len(episodes) == 4560
    assert episodes.groupby("policy").size().eq(456).all()
    assert not episodes.duplicated(["task_id", "replicate", "policy"]).any()
    assert episodes.reward.isin([0, 1]).all()
    y = episodes.reward.to_numpy(float)
    values, contributions = {}, {}
    for policy in metadata["policies"]:
        mask = episodes.policy.eq(policy).to_numpy()
        values[policy] = float(y[mask].mean())
        contributions[policy] = mask * (y - values[policy]) / mask.sum()
    rows = []
    for policy in metadata["policies"]:
        gain = values[policy] - values["never"]
        ci = interval(gain, contributions[policy] - contributions["never"])
        row = dict(
            policy=policy,
            episodes=456,
            successes=int(episodes.loc[episodes.policy.eq(policy), "reward"].sum()),
            mc_value_pct=100 * values[policy],
            mc_gain_pp=100 * gain,
            mc_gain_low_pp=100 * ci["low"],
            mc_gain_high_pp=100 * ci["high"],
            mc_gain_se_pp=100 * ci["se"],
        )
        for key in (
            "mc_value_pct",
            "mc_gain_pp",
            "mc_gain_low_pp",
            "mc_gain_high_pp",
            "mc_gain_se_pp",
        ):
            np.testing.assert_allclose(row[key], expected.loc[policy, key], atol=1e-9, rtol=1e-9)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "retail_mc.csv", index=False)
    print("Verified ten direct-policy estimates from 4,560 trajectories.", flush=True)
