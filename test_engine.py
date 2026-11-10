"""
test_engine.py
Comprehensive Unit & Integration Test Suite for Socio-Economic Grant Prioritization Engine.
Tests:
  - Synthetic Data Ingestion & Scoring Normalization
  - 0-1 Integer Linear Programming Solver (Taha Ch 9)
  - Multi-Objective Goal Programming Solver (Taha Ch 8)
  - LP Duality, Shadow Prices & Integrality Gap (Taha Ch 4)
  - Explicit Ordinal Priority Ranking (#1 to #N)
  - 3-Tier Queuing (Tier 1: Funded, Tier 2: Waitlist, Tier 3: Deferred)
  - Dynamic Proposal Ingestion & Live Determination
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import unittest
import numpy as np
import pandas as pd

from src.data_generator import generate_grant_applications
from src.impact_model import SocioEconomicImpactModel
from src.ilp_solver import IntegerLinearProgrammingSolver
from src.gp_solver import GoalProgrammingGrantSolver
from src.grant_engine import GrantPrioritizationEngine


class TestGrantEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_csv = "data/test_grants.csv"
        cls.raw_df = generate_grant_applications(output_path=cls.test_csv, num_applications=30, random_seed=42)
        cls.impact_model = SocioEconomicImpactModel()
        cls.scored_df = cls.impact_model.compute_impact_scores(cls.raw_df)
        cls.ilp_solver = IntegerLinearProgrammingSolver()
        cls.gp_solver = GoalProgrammingGrantSolver()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)

    def test_dataset_generation_and_integrity(self):
        self.assertEqual(len(self.raw_df), 30)
        required_cols = [
            "grant_id", "title", "sector", "region_tier",
            "requested_amount", "direct_beneficiaries",
            "sroi_ratio", "vulnerability_index", "execution_feasibility"
        ]
        for col in required_cols:
            self.assertIn(col, self.raw_df.columns)
        self.assertFalse(self.raw_df.isnull().values.any())
        self.assertTrue((self.raw_df["requested_amount"] > 0).all())

    def test_impact_score_monotonicity_and_bounds(self):
        scores = self.scored_df["impact_score"]
        self.assertTrue((scores >= 0.0).all() and (scores <= 100.0).all())
        self.assertTrue("efficiency_ratio" in self.scored_df.columns)
        self.assertTrue((self.scored_df["efficiency_ratio"] > 0).all())

    def test_ilp_budget_and_equity_constraints(self):
        budget = 8000000.0  # ₹80 Lakhs
        rural_floor = 0.35
        sector_cap = 0.40

        res = self.ilp_solver.solve(
            df=self.scored_df,
            total_budget=budget,
            rural_floor_pct=rural_floor,
            max_sector_cap_pct=sector_cap
        )

        # Budget ceiling must be respected
        self.assertLessEqual(res["total_budget_utilized"], budget + 1e-3)

        # Rural equity floor must be respected if feasible
        if res["total_grants_funded"] > 0:
            rural_allocated = res["rural_budget_allocated"]
            self.assertGreaterEqual(rural_allocated, (rural_floor * budget) - 1.0)

        # Sector cap respected
        for s, amount in res["sector_allocation_breakdown"].items():
            self.assertLessEqual(amount, (sector_cap * budget) + 1.0)

    def test_ilp_duality_and_shadow_prices(self):
        budget = 10000000.0  # ₹1 Crore
        res = self.ilp_solver.solve(df=self.scored_df, total_budget=budget)

        duality = res["duality_analysis"]
        self.assertIn("budget_shadow_price_per_rupee", duality)
        self.assertGreaterEqual(duality["budget_shadow_price_per_rupee"], 0.0)
        self.assertGreaterEqual(duality["lp_continuous_upper_bound"], res["total_impact_score"] - 1e-3)
        self.assertGreaterEqual(duality["integrality_gap_abs"], 0.0)

    def test_goal_programming_deviational_variables(self):
        budget = 10000000.0
        res = self.gp_solver.solve(df=self.scored_df, total_budget=budget)

        self.assertEqual(res["status"], "Optimal")
        self.assertIn("goals_summary", res)
        goals = res["goals_summary"]
        self.assertGreaterEqual(goals["impact_achieved"], 0.0)
        self.assertGreaterEqual(goals["budget_utilized"], 0.0)
        self.assertGreaterEqual(goals["total_penalty_deviation_score"], 0.0)

    def test_priority_ranking_and_tiers(self):
        engine = GrantPrioritizationEngine(
            dataset_path=self.test_csv,
            total_budget=8000000.0,
            random_seed=42
        )
        benchmark = engine.benchmark_all_strategies(verbose=False)

        self.assertIn("ranked_applications", benchmark)
        apps = benchmark["ranked_applications"]
        self.assertEqual(len(apps), 30)

        # Check strict 1 to N ranking
        ranks = [a["priority_rank"] for a in apps]
        self.assertEqual(ranks, list(range(1, 31)))

        # Check tiers presence
        tier_codes = {a["tier_code"] for a in apps}
        self.assertIn("TIER_1", tier_codes)
        self.assertIn("TIER_2", tier_codes)
        self.assertIn("TIER_3", tier_codes)

    def test_dynamic_add_grant_application(self):
        engine = GrantPrioritizationEngine(
            dataset_path=self.test_csv,
            total_budget=10000000.0,
            random_seed=42
        )
        init_count = len(engine.raw_df)

        result = engine.add_grant_application(
            title="Solar Cold Storage for Tribal Farmers",
            sector="Agriculture & Water",
            region_tier="Rural/Distressed",
            requested_amount=750000.0,
            direct_beneficiaries=3500,
            sroi_ratio=5.5,
            vulnerability_index=0.92,
            execution_feasibility=0.95
        )

        self.assertEqual(len(engine.raw_df), init_count + 1)
        self.assertIn("grant_id", result)
        self.assertIn("priority_rank", result)
        self.assertIn("verdict", result)
        self.assertGreater(result["impact_score"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
