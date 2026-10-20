"""End-to-end smoke test on synthetic data (fast grids)."""

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from churn.pipeline import run
from churn.predict import score
from tests.fake_data import make_fake_telco


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data_path = root / "data.csv"
        make_fake_telco(600, seed=1).to_csv(cls.data_path, index=False)
        cls.reports = root / "reports"
        cls.model_path = root / "models" / "model.joblib"
        cls.metrics = run(cls.data_path, cls.reports, cls.model_path, fast=True,
                          log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_outputs_written(self):
        saved = json.loads((self.reports / "metrics.json").read_text())
        self.assertEqual(saved["selected_model"], self.metrics["selected_model"])
        for name in ["churn_drivers", "roc_pr_curves", "calibration",
                     "profit_curve", "permutation_importance"]:
            self.assertTrue((self.reports / "figures" / f"{name}.png").exists(), name)
        self.assertTrue(self.model_path.exists())

    def test_model_learns_signal(self):
        test = self.metrics["test"][self.metrics["selected_model"]]
        self.assertGreater(test["roc_auc"], 0.65)
        self.assertNotEqual(self.metrics["selected_model"], "baseline")

    def test_scoring_new_customers(self):
        bundle = joblib.load(self.model_path)
        new = make_fake_telco(30, seed=99).drop(columns="Churn")
        scored = score(new, bundle)
        self.assertEqual(len(scored), 30)
        self.assertTrue(scored["churn_probability"].between(0, 1).all())
        profits = scored["expected_profit_if_contacted"]
        self.assertTrue(profits.is_monotonic_decreasing)
        self.assertTrue((scored["contact"] == (profits > 0)).all())


if __name__ == "__main__":
    unittest.main()
