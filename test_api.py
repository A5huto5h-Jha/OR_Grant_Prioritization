"""
test_api.py
Unit tests for FastAPI Grant Prioritization endpoints and schemas.
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
from server import (
    health_check, get_grant_applications, optimize_grants,
    add_new_grant, reset_proposals,
    OptimizationRequest, NewGrantRequest, serve_dashboard_root
)


class TestGrantAPI(unittest.TestCase):

    def test_health_check_endpoint(self):
        res = health_check()
        self.assertEqual(res["status"], "healthy")
        self.assertIn("solvers", res)

    def test_get_grants_endpoint(self):
        res = get_grant_applications()
        self.assertIn("applications", res)
        self.assertGreater(res["total_count"], 0)
        first = res["applications"][0]
        self.assertIn("grant_id", first)
        self.assertIn("impact_score", first)
        self.assertIn("priority_rank", first)

    def test_optimize_grants_endpoint(self):
        req = OptimizationRequest(
            total_budget=12000000.0,
            rural_floor_pct=0.35,
            max_sector_cap_pct=0.40
        )
        res = optimize_grants(req)
        self.assertEqual(res["status"], "success")
        data = res["data"]
        self.assertIn("baseline_greedy", data)
        self.assertIn("integer_linear_programming", data)
        self.assertIn("goal_programming", data)
        self.assertIn("ranked_applications", data)

        ilp = data["integer_linear_programming"]
        self.assertGreater(ilp["total_impact_score"], 0)
        self.assertGreater(ilp["total_grants_funded"], 0)

    def test_add_new_grant_endpoint(self):
        req = NewGrantRequest(
            title="Mobile Eye Clinic for Rural Schools",
            sector="Healthcare",
            region_tier="Rural/Distressed",
            requested_amount=850000.0,
            direct_beneficiaries=6200,
            sroi_ratio=4.8,
            vulnerability_index=0.88,
            execution_feasibility=0.92
        )
        res = add_new_grant(req)
        self.assertEqual(res["status"], "success")
        data = res["data"]
        self.assertIn("grant_id", data)
        self.assertIn("priority_rank", data)
        self.assertIn("verdict", data)

    def test_reset_proposals_endpoint(self):
        res = reset_proposals()
        self.assertEqual(res["status"], "success")

    def test_serve_dashboard_root(self):
        res = serve_dashboard_root()
        self.assertEqual(res.status_code, 200)
        self.assertIn("Grant Prioritization", res.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
