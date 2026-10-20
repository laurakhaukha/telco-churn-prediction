import hashlib
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from churn.data import clean, download_data
from tests.fake_data import make_fake_telco


class TestClean(unittest.TestCase):
    def setUp(self):
        self.raw = make_fake_telco(50)

    def test_total_charges_becomes_numeric_and_blank_means_zero(self):
        df = clean(self.raw)
        self.assertTrue(pd.api.types.is_float_dtype(df["TotalCharges"]))
        self.assertTrue((df.loc[df["tenure"] == 0, "TotalCharges"] == 0).all())
        self.assertFalse(df["TotalCharges"].isna().any())

    def test_target_mapped_to_int(self):
        df = clean(self.raw)
        self.assertEqual(set(df["Churn"].unique()) <= {0, 1}, True)
        self.assertEqual(df["Churn"].sum(), (self.raw["Churn"] == "Yes").sum())

    def test_works_without_target_for_scoring(self):
        df = clean(self.raw.drop(columns="Churn"))
        self.assertNotIn("Churn", df.columns)

    def test_missing_column_raises(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            clean(self.raw.drop(columns="Contract"))

    def test_blank_total_charges_with_tenure_raises(self):
        bad = self.raw.copy()
        bad.loc[bad["tenure"] > 0, "TotalCharges"] = " "
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            clean(bad)

    def test_bad_target_raises(self):
        bad = self.raw.copy()
        bad.loc[0, "Churn"] = "Maybe"
        with self.assertRaisesRegex(ValueError, "Yes"):
            clean(bad)

    def test_does_not_modify_input(self):
        before = self.raw.copy()
        clean(self.raw)
        pd.testing.assert_frame_equal(self.raw, before)


class TestDownload(unittest.TestCase):
    def test_download_from_url_with_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.csv"
            make_fake_telco(20).to_csv(src, index=False)
            digest = hashlib.sha256(src.read_bytes()).hexdigest()
            dest = Path(tmp) / "raw" / "data.csv"

            download_data(src.as_uri(), dest, expected_sha256=digest)
            self.assertEqual(src.read_bytes(), dest.read_bytes())

            with warnings.catch_warnings():
                warnings.simplefilter("error")  # a correct checksum must not warn
                download_data(src.as_uri(), dest, expected_sha256=digest, force=True)

    def test_checksum_mismatch_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "source.csv"
            make_fake_telco(20).to_csv(src, index=False)
            with self.assertWarns(UserWarning):
                download_data(src.as_uri(), Path(tmp) / "data.csv", expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
