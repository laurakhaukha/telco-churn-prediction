import unittest

import numpy as np

from churn.business import (
    CampaignAssumptions,
    compare_policies,
    customer_value,
    expected_profit,
    profit_curve,
    realized_profit,
    sensitivity,
)

A = CampaignAssumptions(contact_cost=5, incentive_cost=50, save_rate=0.3,
                        value_months=12, gross_margin=0.5)


class TestEconomics(unittest.TestCase):
    def test_customer_value(self):
        self.assertAlmostEqual(customer_value([100.0], A)[0], 100 * 0.5 * 12)

    def test_expected_profit_by_hand(self):
        # value = 600; churner gain = 0.3 * (600 - 50) = 165
        # E = 0.4 * 165 - 0.6 * 50 - 5 = 66 - 30 - 5 = 31
        self.assertAlmostEqual(expected_profit([0.4], [100.0], A)[0], 31.0)

    def test_expected_profit_matches_realized_at_certainty(self):
        for y in (0, 1):
            self.assertAlmostEqual(expected_profit([y], [80.0], A)[0],
                                   realized_profit([y], [80.0], A)[0])

    def test_contacting_a_loyal_customer_costs_offer_plus_contact(self):
        self.assertAlmostEqual(realized_profit([0], [80.0], A)[0], -55.0)


class TestPolicies(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        n = 500
        self.p = rng.random(n)
        self.y = rng.random(n) < self.p  # calibrated by construction
        self.monthly = rng.uniform(20, 110, n)
        self.contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
        self.res = compare_policies(self.p, self.y, self.monthly, self.contract, A)

    def test_trivial_policies(self):
        self.assertEqual(self.res["nobody"]["profit"], 0.0)
        self.assertAlmostEqual(self.res["everyone"]["profit"],
                               realized_profit(self.y, self.monthly, A).sum())

    def test_perfect_foresight_is_an_upper_bound(self):
        best = self.res["perfect_foresight"]["profit"]
        for name, pol in self.res.items():
            self.assertLessEqual(pol["profit"], best + 1e-9, name)

    def test_model_contacts_exactly_positive_expected_profit(self):
        expected = int((expected_profit(self.p, self.monthly, A) > 0).sum())
        self.assertEqual(self.res["model"]["customers_contacted"], expected)
        self.assertEqual(self.res["random"]["customers_contacted"], expected)

    def test_informative_scores_beat_random(self):
        self.assertGreater(self.res["model"]["profit"], self.res["random"]["profit"])

    def test_profit_curve_endpoints(self):
        share, cum = profit_curve(self.p, self.y, self.monthly, A)
        self.assertEqual(share[0], 0.0)
        self.assertEqual(share[-1], 1.0)
        self.assertEqual(cum[0], 0.0)
        self.assertAlmostEqual(cum[-1], self.res["everyone"]["profit"])

    def test_sensitivity_rows(self):
        rows = sensitivity(self.p, self.y, self.monthly, self.contract, A, save_rates=(0.1, 0.5))
        self.assertEqual([r["save_rate"] for r in rows], [0.1, 0.5])
        self.assertLess(rows[0]["model"], rows[1]["model"])


if __name__ == "__main__":
    unittest.main()
