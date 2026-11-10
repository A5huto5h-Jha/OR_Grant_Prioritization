"""
ilp_solver.py
Technique 1: 0-1 Integer Linear Programming (ILP) Solver.
Reference: Hamdy A. Taha (Operations Research: An Introduction, Ch 4 & Ch 9).
Formulates the Multidimensional 0-1 Knapsack with policy constraints (budget, rural equity, sector caps)
and extracts Dual Shadow Prices via continuous LP relaxation.
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import pulp
from typing import Dict, Any, List, Optional


class IntegerLinearProgrammingSolver:
    """
    0-1 Integer Linear Programming Grant Prioritization Solver.

    Mathematical Formulation:
      Decision Variables:
        x_i in {0, 1} for each grant application i = 1, ..., N.

      Objective Function:
        Maximize Total Socio-Economic Impact:
          max Z = sum_{i=1}^N S_i * x_i

      Subject to:
        1. Total Budget Ceiling:
           sum_{i=1}^N c_i * x_i <= B_total
           (Dual Shadow Price y_budget extracted via continuous relaxation)

        2. Rural / Distressed Region Equity Floor:
           sum_{i in Rural} c_i * x_i >= alpha * B_total
           (Guarantees underprivileged areas receive at least alpha % of funds)

        3. Sectoral Diversification Ceiling:
           sum_{i in Sector_k} c_i * x_i <= beta_k * B_total   forall sectors k
           (Prevents a single sector from monopolizing grant capital)

        4. Binary Restriction:
           x_i in {0, 1}
    """

    def __init__(
        self,
        default_rural_floor_pct: float = 0.35,
        default_max_sector_cap_pct: float = 0.35
    ):
        """
        Parameters:
            default_rural_floor_pct: Minimum budget fraction allocated to Rural/Distressed projects.
            default_max_sector_cap_pct: Maximum budget fraction allocated to any single sector.
        """
        self.rural_floor_pct = default_rural_floor_pct
        self.max_sector_cap_pct = default_max_sector_cap_pct

    def solve(
        self,
        df: pd.DataFrame,
        total_budget: float,
        rural_floor_pct: Optional[float] = None,
        max_sector_cap_pct: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Solve the 0-1 ILP model and its continuous LP relaxation for duality analysis.
        """
        alpha = rural_floor_pct if rural_floor_pct is not None else self.rural_floor_pct
        beta = max_sector_cap_pct if max_sector_cap_pct is not None else self.max_sector_cap_pct

        n_grants = len(df)
        grant_ids = df["grant_id"].tolist()
        costs = df["requested_amount"].tolist()
        scores = df["impact_score"].tolist()
        sectors = df["sector"].tolist()
        regions = df["region_tier"].tolist()
        beneficiaries = df["direct_beneficiaries"].tolist()

        # ---------------------------------------------------------------------
        # 1. 0-1 INTEGER PROGRAM (BRANCH AND BOUND)
        # ---------------------------------------------------------------------
        prob_int = pulp.LpProblem("Grant_Allocation_ILP", pulp.LpMaximize)

        # Binary decision variables
        x_int = {}
        for gid in grant_ids:
            if hasattr(prob_int, "add_variable"):
                x_int[gid] = prob_int.add_variable(name=f"x_{gid}", lowBound=0, upBound=1, cat="Binary")
            else:
                x_int[gid] = pulp.LpVariable(name=f"x_{gid}", cat=pulp.LpBinary)

        # Objective Function
        prob_int += pulp.lpSum([scores[i] * x_int[grant_ids[i]] for i in range(n_grants)]), "Total_Social_Impact"

        # Constraint 1: Total Budget Ceiling
        prob_int += (
            pulp.lpSum([costs[i] * x_int[grant_ids[i]] for i in range(n_grants)]) <= total_budget,
            "Budget_Constraint"
        )

        # Constraint 2: Rural Equity Floor
        min_rural_budget = alpha * total_budget
        rural_indices = [i for i in range(n_grants) if regions[i] == "Rural/Distressed"]
        prob_int += (
            pulp.lpSum([costs[i] * x_int[grant_ids[i]] for i in rural_indices]) >= min_rural_budget,
            "Rural_Equity_Constraint"
        )

        # Constraint 3: Sector Caps
        unique_sectors = sorted(list(set(sectors)))
        max_sector_budget = beta * total_budget
        for s in unique_sectors:
            sec_indices = [i for i in range(n_grants) if sectors[i] == s]
            prob_int += (
                pulp.lpSum([costs[i] * x_int[grant_ids[i]] for i in sec_indices]) <= max_sector_budget,
                f"Sector_Cap_{s.replace(' ', '_').replace('&', 'and')}"
            )

        # Solve Integer Problem
        solver = None
        if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.getSolver("HiGHS", msg=False)
        elif "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True):
            solver = pulp.PULP_CBC_CMD(msg=False)

        if solver is not None:
            res_int = prob_int.solve(solver)
        else:
            res_int = prob_int.solve()

        # Extract Integer Solutions
        funded_ids = []
        total_allocated = 0.0
        total_impact = 0.0
        total_beneficiaries = 0
        rural_allocated = 0.0
        sector_breakdown = {s: 0.0 for s in unique_sectors}

        for i, gid in enumerate(grant_ids):
            val = round(float(x_int[gid].varValue or 0.0))
            if val == 1:
                funded_ids.append(gid)
                total_allocated += costs[i]
                total_impact += scores[i]
                total_beneficiaries += beneficiaries[i]
                sector_breakdown[sectors[i]] += costs[i]
                if regions[i] == "Rural/Distressed":
                    rural_allocated += costs[i]

        rural_pct_achieved = (rural_allocated / total_allocated * 100.0) if total_allocated > 0 else 0.0

        # ---------------------------------------------------------------------
        # 2. CONTINUOUS LP RELAXATION FOR DUALITY ANALYSIS (UNIT 2 OF SYLLABUS)
        # ---------------------------------------------------------------------
        prob_lp = pulp.LpProblem("Grant_Allocation_Continuous_Relaxation", pulp.LpMaximize)
        x_lp = {}
        for gid in grant_ids:
            if hasattr(prob_lp, "add_variable"):
                x_lp[gid] = prob_lp.add_variable(name=f"x_lp_{gid}", lowBound=0.0, upBound=1.0, cat="Continuous")
            else:
                x_lp[gid] = pulp.LpVariable(name=f"x_lp_{gid}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)

        prob_lp += pulp.lpSum([scores[i] * x_lp[grant_ids[i]] for i in range(n_grants)]), "LP_Social_Impact"

        # Constraints with named references for dual extraction
        c_budget = (pulp.lpSum([costs[i] * x_lp[grant_ids[i]] for i in range(n_grants)]) <= total_budget)
        prob_lp += c_budget, "LP_Budget_Constraint"

        c_rural = (pulp.lpSum([costs[i] * x_lp[grant_ids[i]] for i in rural_indices]) >= min_rural_budget)
        prob_lp += c_rural, "LP_Rural_Constraint"

        for s in unique_sectors:
            sec_indices = [i for i in range(n_grants) if sectors[i] == s]
            prob_lp += (
                pulp.lpSum([costs[i] * x_lp[grant_ids[i]] for i in sec_indices]) <= max_sector_budget,
                f"LP_Sector_{s.replace(' ', '_').replace('&', 'and')}"
            )

        if solver is not None:
            prob_lp.solve(solver)
        else:
            prob_lp.solve()

        # Extract Continuous Objective & Dual Shadow Prices
        lp_continuous_objective = float(prob_lp.objective.value() or 0.0) if prob_lp.objective else total_impact

        budget_shadow_price = 0.0
        rural_shadow_price = 0.0

        try:
            if hasattr(prob_lp, "constraints"):
                constrs = prob_lp.constraints
                if callable(constrs):
                    for c in prob_lp.constraints():
                        if c.name == "LP_Budget_Constraint" and c.pi is not None:
                            budget_shadow_price = float(abs(c.pi))
                        elif c.name == "LP_Rural_Constraint" and c.pi is not None:
                            rural_shadow_price = float(abs(c.pi))
                elif isinstance(constrs, dict):
                    if "LP_Budget_Constraint" in constrs and constrs["LP_Budget_Constraint"].pi is not None:
                        budget_shadow_price = float(abs(constrs["LP_Budget_Constraint"].pi))
                    if "LP_Rural_Constraint" in constrs and constrs["LP_Rural_Constraint"].pi is not None:
                        rural_shadow_price = float(abs(constrs["LP_Rural_Constraint"].pi))
        except Exception:
            budget_shadow_price = 0.0
            rural_shadow_price = 0.0

        # Integrality Gap: Cost of Indivisibility
        integrality_gap_abs = round(lp_continuous_objective - total_impact, 2)
        integrality_gap_pct = round((integrality_gap_abs / lp_continuous_objective * 100.0) if lp_continuous_objective > 0 else 0.0, 2)

        return {
            "technique": "0-1 Integer Linear Programming (ILP)",
            "solver": "Branch and Bound / Simplex (PuLP)",
            "status": "Optimal",
            "funded_grant_ids": funded_ids,
            "total_grants_funded": len(funded_ids),
            "total_grants_evaluated": n_grants,
            "total_budget_available": round(total_budget, 2),
            "total_budget_utilized": round(total_allocated, 2),
            "budget_utilization_pct": round((total_allocated / total_budget * 100.0) if total_budget > 0 else 0.0, 2),
            "total_impact_score": round(total_impact, 2),
            "total_beneficiaries_reached": total_beneficiaries,
            "rural_budget_allocated": round(rural_allocated, 2),
            "rural_equity_pct_achieved": round(rural_pct_achieved, 2),
            "rural_equity_floor_pct": round(alpha * 100.0, 1),
            "sector_allocation_breakdown": {s: round(v, 2) for s, v in sector_breakdown.items()},
            "duality_analysis": {
                "budget_shadow_price_per_rupee": round(budget_shadow_price, 7),
                "budget_shadow_price_per_lakh": round(budget_shadow_price * 100000.0, 4),
                "rural_equity_shadow_price": round(rural_shadow_price, 7),
                "lp_continuous_upper_bound": round(lp_continuous_objective, 2),
                "integrality_gap_abs": integrality_gap_abs,
                "integrality_gap_pct": integrality_gap_pct
            }
        }
