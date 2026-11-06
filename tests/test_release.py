"""Checks on the released cohort, continuation clock, and estimators."""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from agent_eval.analysis import STUDIES, fit, load, verify
from agent_eval.moments import future_sum


class ReleaseTests(unittest.TestCase):
    def test_full_trajectory_outcomes_and_folds(self):
        for study in STUDIES:
            with self.subTest(study=study):
                panel, metadata = load(study)
                self.assertEqual(len(panel["outcome_sums"]), metadata["trajectories"])
                self.assertEqual(
                    panel["group_masks"].shape, (metadata["decisions"], len(metadata["groups"]))
                )
                self.assertTrue(np.isin(panel["treatment"], [0, 1]).all())
                self.assertEqual(panel["x"].shape[1], len(metadata["nuisance_features"]))
                self.assertEqual(
                    panel["target_control_x"].shape[1], len(metadata["future_control_features"])
                )

    def test_same_phase_future_visits_are_retained(self):
        panel = {"episode_id": np.array([0, 0, 0, 1]), "t": np.array([0, 1, 2, 0])}
        features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        assignment = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            future_sum(panel, features, assignment),
            [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        )

    def test_impossiblebench_matches_paper(self):
        with threadpool_limits(limits=2):
            frame, _ = fit("impossiblebench")
        self.assertEqual(verify("impossiblebench", frame), 60)
        row = frame[
            (frame.group_id == "prior_attack_resisted") & (frame.estimand == "selective_policy")
        ].iloc[0]
        self.assertTrue(row.supported)
        self.assertEqual((row.estimate, row.lower, row.upper), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
