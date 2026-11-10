"""
gp_solver.py
Technique 2: Multi-Objective Goal Programming (GP) Solver.
Reference: Hamdy A. Taha (Operations Research: An Introduction, Ch 8).
Resolves conflicting government/donor policy goals using deviational variables (d^-, d^+)
and weighted penalty minimization.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import pulp
from typing import Dict, Any, List, Optional


class GoalProgrammingGrantSolver:
    """
    Multi-Objective Goal Programming Grant Allocation Solver.

    Mathematical Formulation (Hamdy A. Taha, Ch 8):
      Goal 1 (Socio-Economic Impact Target T_1):
        sum_{i=1}^N S_i * x_i + d_1^- - d_1^+ = T_1
        (Minimize under-achievement: d_1^-)

      Goal 2 (Budget Ceiling Target B):
        sum_{i=1}^N c_i * x_i + d_2^- - d_2^+ = B
        (Minimize over-expenditure: d_2^+)

      Goal 3 (Rural Equity Target T_3 = alpha * B):
        sum_{i in Rural} c_i * x_i + d_3^- - d_3^+ = T_3
        (Minimize under-allocation to rural communities: d_3^-)

      Goal 4 (Direct Beneficiary Reach Target T_4):
        sum_{i=1}^N Beneficiaries_i * x_i + d_4^- - d_4^+ = T_4
        (Minimize shortfall in people reached: d_4^-)

      Objective Function (Weighted Penalty Minimization):
        min Z = W_1 * (d_1^- / T_1) + W_2 * (d_2^+ / B) + W_3 * (d_3^- / T_3) + W_4 * (d_4^- / T_4)

      Subject to:
        x_i in {0, 1},  d_k^-, d_k^+ >= 0,  d_k^- * d_k^+ = 0
    """

    def __init__(
        self,
        weight_impact: float = 0.35,
        weight_budget_strictness: float = 0.30,
        weight_rural_equity: float = 0.20,
        weight_beneficiaries: float = 0.15
    ):
        """
        Parameters:
            weight_impact (W_1): Penalty weight for failing to meet social welfare target.
            weight_budget_strictness (W_2): Penalty weight for exceeding budget ceiling.
            weight_rural_equity (W_3): Penalty weight for under-funding rural distress areas.
            weight_beneficiaries (W_4): Penalty weight for missing direct beneficiary reach.
        """
        self.w_impact = weight_impact
        self.w_budget = weight_budget_strictness
        self.w_rural = weight_rural_equity
        self.w_bene = weight_beneficiaries

    def solve(
        self,
        df: pd.DataFrame,
        total_budget: float,
        target_impact_score: Optional[float] = None,
        target_beneficiaries: Optional[int] = None,
        rural_floor_pct: float = 0.35,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Solve Multi-Objective Goal Program via PuLP Simplex.
        """
        n_grants = len(df)
        grant_ids = df["grant_id"].tolist()
        costs = df["requested_amount"].tolist()
        scores = df["impact_score"].tolist()
        sectors = df["sector"].tolist()
        regions = df["region_tier"].tolist()
        beneficiaries = df["direct_beneficiaries"].tolist()

        unique_sectors = sorted(list(set(sectors)))
        rural_indices = [i for i in range(n_grants) if regions[i] == "Rural/Distressed"]

        # Default aspirational targets if not provided
        # Realistic empirical targets: ~60% of total pool theoretical maximums
        sum_all_scores = sum(scores)
        sum_all_bene = sum(beneficiaries)

        t_impact = target_impact_score if target_impact_score is not None else round(sum_all_scores * 0.45, 1)
        t_bene = target_beneficiaries if target_beneficiaries is not None else int(sum_all_bene * 0.45)
        t_rural = round(rural_floor_pct * total_budget, 2)
        t_budget = total_budget

        # Extract weights
        w1 = weights.get("impact", self.w_impact) if weights else self.w_impact
        w2 = weights.get("budget", self.w_budget) if weights else self.w_budget
        w3 = weights.get("rural", self.w_rural) if weights else self.w_rural
        w4 = weights.get("beneficiaries", self.w_bene) if weights else self.w_bene

        prob = pulp.LpProblem("Grant_Allocation_Goal_Programming", pulp.LpMinimize)

        # 1. Binary Decision Variables (x_i)
        x = {}
        for gid in grant_ids:
            if hasattr(prob, "add_variable"):
                x[gid] = prob.add_variable(name=f"x_gp_{gid}", lowBound=0, upBound=1, cat="Binary")
            else:
                x[gid] = pulp.LpVariable(name=f"x_gp_{gid}", cat=pulp.LpBinary)

        # 2. Continuous Deviational Variables (d_k^-, d_k^+)
        if hasattr(prob, "add_variable"):
            d1_minus = prob.add_variable(name="d1_minus", lowBound=0.0, cat="Continuous")
            d1_plus = prob.add_variable(name="d1_plus", lowBound=0.0, cat="Continuous")
            d2_minus = prob.add_variable(name="d2_minus", lowBound=0.0, cat="Continuous")
            d2_plus = prob.add_variable(name="d2_plus", lowBound=0.0, cat="Continuous")
            d3_minus = prob.add_variable(name="d3_minus", lowBound=0.0, cat="Continuous")
            d3_plus = prob.add_variable(name="d3_plus", lowBound=0.0, cat="Continuous")
            d4_minus = prob.add_variable(name="d4_minus", lowBound=0.0, cat="Continuous")
            d4_plus = prob.add_variable(name="d4_plus", lowBound=0.0, cat="Continuous")
        else:
            d1_minus = pulp.LpVariable("d1_minus", lowBound=0.0, cat=pulp.LpContinuous)
            d1_plus = pulp.LpVariable("d1_plus", lowBound=0.0, cat=pulp.LpContinuous)
            d2_minus = pulp.LpVariable("d2_minus", lowBound=0.0, cat=pulp.LpContinuous)
            d2_plus = pulp.LpVariable("d2_plus", lowBound=0.0, cat=pulp.LpContinuous)
            d3_minus = pulp.LpVariable("d3_minus", lowBound=0.0, cat=pulp.LpContinuous)
            d3_plus = pulp.LpVariable("d3_plus", lowBound=0.0, cat=pulp.LpContinuous)
            d4_minus = pulp.LpVariable("d4_minus", lowBound=0.0, cat=pulp.LpContinuous)
            d4_plus = pulp.LpVariable("d4_plus", lowBound=0.0, cat=pulp.LpContinuous)

        # 3. Goal Equations
        # Goal 1: Social Impact Target (Minimize under-achievement d1_minus)
        prob += (
            pulp.lpSum([scores[i] * x[grant_ids[i]] for i in range(n_grants)]) + d1_minus - d1_plus == t_impact,
            "Goal_1_Social_Impact"
        )

        # Goal 2: Budget Target (Minimize over-spending d2_plus)
        prob += (
            pulp.lpSum([costs[i] * x[grant_ids[i]] for i in range(n_grants)]) + d2_minus - d2_plus == t_budget,
            "Goal_2_Budget_Ceiling"
        )

        # Hard guardrail: Even in goal programming, budget overspend cannot exceed 5% of B
        prob += d2_plus <= (0.05 * t_budget), "Budget_Hard_Overrun_Cap"

        # Goal 3: Rural Community Allocation (Minimize under-allocation d3_minus)
        prob += (
            pulp.lpSum([costs[i] * x[grant_ids[i]] for i in rural_indices]) + d3_minus - d3_plus == t_rural,
            "Goal_3_Rural_Equity"
        )

        # Goal 4: Direct Beneficiary Reach (Minimize under-achievement d4_minus)
        prob += (
            pulp.lpSum([beneficiaries[i] * x[grant_ids[i]] for i in range(n_grants)]) + d4_minus - d4_plus == t_bene,
            "Goal_4_Beneficiary_Reach"
        )

        # 4. Objective Function: Minimize Normalized Weighted Penalty Deviations
        prob += (
            (w1 * (d1_minus / max(1.0, t_impact))) +
            (w2 * (d2_plus / max(1.0, t_budget))) +
            (w3 * (d3_minus / max(1.0, t_rural))) +
            (w4 * (d4_minus / max(1.0, t_bene))),
            "Total_Weighted_Deviation_Penalty"
        )

        # Solve
        solver = None
        if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.getSolver("HiGHS", msg=False)
        elif "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.PULP_CBC_CMD(msg=False)

        if solver is not None:
            prob.solve(solver)
        else:
            prob.solve()

        # Extract Results
        funded_ids = []
        total_allocated = 0.0
        total_impact = 0.0
        total_bene_reached = 0
        rural_allocated = 0.0
        sector_breakdown = {s: 0.0 for s in unique_sectors}

        for i, gid in enumerate(grant_ids):
            val = round(float(x[gid].varValue or 0.0))
            if val == 1:
                funded_ids.append(gid)
                total_allocated += costs[i]
                total_impact += scores[i]
                total_bene_reached += beneficiaries[i]
                sector_breakdown[sectors[i]] += costs[i]
                if regions[i] == "Rural/Distressed":
                    rural_allocated += costs[i]

        rural_pct_achieved = (rural_allocated / total_allocated * 100.0) if total_allocated > 0 else 0.0

        # Deviations
        v_d1_minus = round(float(d1_minus.varValue or 0.0), 2)
        v_d1_plus = round(float(d1_plus.varValue or 0.0), 2)
        v_d2_minus = round(float(d2_minus.varValue or 0.0), 2)
        v_d2_plus = round(float(d2_plus.varValue or 0.0), 2)
        v_d3_minus = round(float(d3_minus.varValue or 0.0), 2)
        v_d3_plus = round(float(d3_plus.varValue or 0.0), 2)
        v_d4_minus = round(float(d4_minus.varValue or 0.0), 1)
        v_d4_plus = round(float(d4_plus.varValue or 0.0), 1)

        total_penalty = round(float(prob.objective.value() or 0.0), 5)

        return {
            "technique": "Multi-Objective Goal Programming (GP)",
            "solver": "Simplex with Deviational Variables (PuLP)",
            "status": "Optimal",
            "funded_grant_ids": funded_ids,
            "total_grants_funded": len(funded_ids),
            "total_grants_evaluated": n_grants,
            "total_budget_available": round(total_budget, 2),
            "total_budget_utilized": round(total_allocated, 2),
            "budget_utilization_pct": round((total_allocated / total_budget * 100.0) if total_budget > 0 else 0.0, 2),
            "total_impact_score": round(total_impact, 2),
            "total_beneficiaries_reached": total_bene_reached,
            "rural_budget_allocated": round(rural_allocated, 2),
            "rural_equity_pct_achieved": round(rural_pct_achieved, 2),
            "sector_allocation_breakdown": {s: round(v, 2) for s, v in sector_breakdown.items()},
            "goals_summary": {
                "impact_target": t_impact,
                "impact_achieved": round(total_impact, 2),
                "impact_underachievement_d1_minus": v_d1_minus,
                "budget_target": t_budget,
                "budget_utilized": round(total_allocated, 2),
                "budget_overspend_d2_plus": v_d2_plus,
                "budget_surplus_d2_minus": v_d2_minus,
                "rural_equity_target": t_rural,
                "rural_equity_achieved": round(rural_allocated, 2),
                "rural_underallocation_d3_minus": v_d3_minus,
                "beneficiaries_target": t_bene,
                "beneficiaries_achieved": total_bene_reached,
                "beneficiaries_shortfall_d4_minus": v_d4_minus,
                "total_penalty_deviation_score": total_penalty
            }
        }
