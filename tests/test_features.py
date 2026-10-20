import unittest

import numpy as np

from churn import config
from churn.data import clean
from churn.evaluation import FEATURE_GROUPS
from churn.features import ENGINEERED_FEATURES, add_features, build_preprocessor
from tests.fake_data import make_fake_telco


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = clean(make_fake_telco(200))

    def test_default_adds_only_log_tenure(self):
        out = add_features(self.df)
        self.assertEqual(set(out.columns) - set(config.FEATURE_COLS), {"log_tenure"})
        np.testing.assert_allclose(out["log_tenure"], np.log1p(self.df["tenure"]))

    def test_identifier_and_target_are_never_features(self):
        out = add_features(self.df)
        self.assertNotIn(config.ID_COL, out.columns)
        self.assertNotIn(config.TARGET, out.columns)

    def test_all_engineered_features(self):
        out = add_features(self.df, extras=tuple(ENGINEERED_FEATURES))
        self.assertTrue(out["n_addons"].between(0, 6).all())
        self.assertTrue(set(out["auto_pay"].unique()) <= {0, 1})
        auto = self.df["PaymentMethod"].str.contains("automatic")
        self.assertTrue((out.loc[auto, "auto_pay"] == 1).all())
        self.assertFalse(out.isna().any().any())

    def test_charge_drift_is_zero_when_bill_never_changed(self):
        df = self.df.copy()
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
        out = add_features(df, extras=("charge_drift",))
        np.testing.assert_allclose(out["charge_drift"], 0, atol=1e-9)

    def test_preprocessor_output_is_finite_and_handles_unseen_categories(self):
        prep = build_preprocessor().fit(self.df[config.FEATURE_COLS])
        new = self.df[config.FEATURE_COLS].head(5).copy()
        new.loc[new.index[0], "PaymentMethod"] = "Crypto"  # never seen in training
        Xt = prep.transform(new)
        self.assertEqual(Xt.shape[0], 5)
        self.assertTrue(np.isfinite(Xt).all())

    def test_importance_groups_cover_every_feature_once(self):
        grouped = [c for cols in FEATURE_GROUPS.values() for c in cols]
        self.assertEqual(sorted(grouped), sorted(config.FEATURE_COLS))


if __name__ == "__main__":
    unittest.main()
