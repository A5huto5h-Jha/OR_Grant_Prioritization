"""
grant_engine.py
Core Grant Prioritization Engine & Comparative Simulation Benchmarking.
Evaluates:
  1. Naive Greedy Benefit-to-Cost Heuristic (Baseline)
  2. Technique 1: 0-1 Integer Linear Programming (Hamdy A. Taha, Ch 9)
  3. Technique 2: Multi-Objective Goal Programming (Hamdy A. Taha, Ch 8)
With LP Duality Analysis, Explicit Priority Rankings (#1 to #N),
and 3-Tier Queuing (Tier 1: Funded, Tier 2: Waitlist, Tier 3: Deferred).
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from src.data_generator import generate_grant_applications
from src.impact_model import SocioEconomicImpactModel
from src.ilp_solver import IntegerLinearProgrammingSolver
from src.gp_solver import GoalProgrammingGrantSolver


class GrantPrioritizationEngine:
    """
    End-to-End Operations Research Grant Prioritization & Benchmarking Engine.
    """

    def __init__(
        self,
        dataset_path: str = "data/grant_applications.csv",
        total_budget: float = 15000000.0,  # Default: ₹1.50 Crore
        rural_floor_pct: float = 0.35,      # 35% minimum rural floor
        max_sector_cap_pct: float = 0.35,   # Max 35% to any single sector
        random_seed: int = 42
    ):
        self.dataset_path = dataset_path
        self.default_budget = total_budget
        self.rural_floor_pct = rural_floor_pct
        self.max_sector_cap_pct = max_sector_cap_pct
        self.random_seed = random_seed

        # Initialize OR components
        self.impact_model = SocioEconomicImpactModel()
        self.ilp_solver = IntegerLinearProgrammingSolver(
            default_rural_floor_pct=rural_floor_pct,
            default_max_sector_cap_pct=max_sector_cap_pct
        )
        self.gp_solver = GoalProgrammingGrantSolver()

        # Load and score dataset
        self.raw_df = self.load_or_generate_dataset()
        self.scored_df = self.impact_model.compute_impact_scores(self.raw_df)

    def load_or_generate_dataset(self) -> pd.DataFrame:
        """Load dataset from disk; generate synthetic feed if absent."""
        if not os.path.exists(self.dataset_path):
            print(f"[Engine] Dataset not found at '{self.dataset_path}'. Generating synthetic grant feed...")
            return generate_grant_applications(
                output_path=self.dataset_path,
                num_applications=60,
                random_seed=self.random_seed
            )
        return pd.read_csv(self.dataset_path)

    def compute_priority_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute an explicit, global ordinal priority ranking (Rank #1 to #N).
        Combines social efficiency density (S_i / c_i) with rural affirmative action weighting.
        """
        ranked_df = df.copy()

        # Equity weighting: 25% boost in priority score for rural/distressed areas
        rural_multiplier = np.where(ranked_df["region_tier"] == "Rural/Distressed", 1.25, 1.0)
        priority_index = (ranked_df["impact_score"] / ranked_df["requested_amount"]) * 10000.0 * rural_multiplier
        ranked_df["priority_index"] = np.round(priority_index, 4)

        # Sort in descending order to assign explicit ranks #1 to #N
        ranked_df = ranked_df.sort_values("priority_index", ascending=False).reset_index(drop=True)
        ranked_df["priority_rank"] = np.arange(1, len(ranked_df) + 1)

        return ranked_df

    def assign_priority_tiers(
        self,
        ranked_df: pd.DataFrame,
        funded_ids: List[str],
        total_budget: float,
        utilized_budget: float,
        waitlist_cutoff: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Assign each application to one of three operational priority tiers:
          - Tier 1: High Priority (Fully Funded in Optimal Solution)
          - Tier 2: Contingency / Waitlist (Top unselected candidates if budget increases)
          - Tier 3: Low Priority / Deferred
        """
        funded_set = set(funded_ids)
        records = []
        unspent_budget = max(0.0, total_budget - utilized_budget)

        waitlist_counter = 1
        for _, row in ranked_df.iterrows():
            gid = row["grant_id"]
            cost = float(row["requested_amount"])
            rec = row.to_dict()

            if gid in funded_set:
                rec["tier_code"] = "TIER_1"
                rec["tier_label"] = "Tier 1: High Priority (Funded)"
                rec["tier_badge"] = "badge-funded"
                rec["waitlist_pos"] = None
                rec["marginal_budget_needed"] = 0.0
                rec["recommendation"] = "Award Full Funding Immediately"
            else:
                if waitlist_counter <= waitlist_cutoff:
                    rec["tier_code"] = "TIER_2"
                    rec["tier_label"] = f"Tier 2: Waitlist (Contingency #{waitlist_counter})"
                    rec["tier_badge"] = "badge-waitlist"
                    rec["waitlist_pos"] = waitlist_counter
                    needed = max(0.0, cost - unspent_budget)
                    rec["marginal_budget_needed"] = round(needed, 2)
                    rec["recommendation"] = f"Priority Reserve #{waitlist_counter} (Needs ₹{needed:,.0f} more capital)"
                    waitlist_counter += 1
                else:
                    rec["tier_code"] = "TIER_3"
                    rec["tier_label"] = "Tier 3: Low Priority (Deferred)"
                    rec["tier_badge"] = "badge-deferred"
                    rec["waitlist_pos"] = None
                    rec["marginal_budget_needed"] = round(cost, 2)
                    rec["recommendation"] = "Defer to Next Annual Grant Cycle"

            records.append(rec)

        return records

    def add_grant_application(
        self,
        title: str,
        sector: str,
        region_tier: str,
        requested_amount: float,
        direct_beneficiaries: int,
        sroi_ratio: float,
        vulnerability_index: float,
        execution_feasibility: float,
        evaluation_budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dynamically inject a new grant application, compute its socio-economic score,
        and evaluate where it ranks and whether it is funded.
        """
        new_id = f"GNT-{len(self.raw_df) + 1:03d}"
        new_record = {
            "grant_id": new_id,
            "title": str(title).strip(),
            "sector": str(sector).strip(),
            "region_tier": str(region_tier).strip(),
            "requested_amount": float(requested_amount),
            "direct_beneficiaries": int(direct_beneficiaries),
            "sroi_ratio": float(sroi_ratio),
            "vulnerability_index": float(vulnerability_index),
            "execution_feasibility": float(execution_feasibility)
        }

        # Append to dataframe and re-score
        new_df = pd.DataFrame([new_record])
        self.raw_df = pd.concat([self.raw_df, new_df], ignore_index=True)
        self.scored_df = self.impact_model.compute_impact_scores(self.raw_df)

        # Re-run benchmark to assess new proposal
        budget = evaluation_budget or self.default_budget
        benchmark_results = self.benchmark_all_strategies(total_budget=budget, verbose=False)

        # Locate the new grant in rankings
        apps = benchmark_results["ranked_applications"]
        new_app_info = next((a for a in apps if a["grant_id"] == new_id), None)

        is_funded_ilp = new_id in benchmark_results["integer_linear_programming"]["funded_grant_ids"]
        is_funded_gp = new_id in benchmark_results["goal_programming"]["funded_grant_ids"]
        is_funded_greedy = new_id in benchmark_results["baseline_greedy"]["funded_grant_ids"]

        if is_funded_ilp:
            verdict = "FUNDED IN TIER 1 (High Priority)"
            verdict_desc = f"Selected by 0-1 ILP optimizer. High social return on capital of ₹{requested_amount:,.2f}."
        elif new_app_info and new_app_info.get("tier_code") == "TIER_2":
            pos = new_app_info.get("waitlist_pos", 1)
            verdict = f"CONTINGENCY WAITLIST (Position #{pos})"
            verdict_desc = f"Next in line for funding if budget increases by ₹{new_app_info['marginal_budget_needed']:,.2f}."
        else:
            verdict = "DEFERRED (Tier 3)"
            verdict_desc = "Capital requirements exceed remaining marginal allocation threshold for this cycle."

        return {
            "grant_id": new_id,
            "title": title,
            "impact_score": float(new_app_info["impact_score"]) if new_app_info else 0.0,
            "priority_rank": int(new_app_info["priority_rank"]) if new_app_info else len(self.scored_df),
            "total_evaluated": len(self.scored_df),
            "verdict": verdict,
            "verdict_description": verdict_desc,
            "is_funded_ilp": is_funded_ilp,
            "is_funded_gp": is_funded_gp,
            "is_funded_greedy": is_funded_greedy,
            "tier_details": new_app_info,
            "updated_benchmark": benchmark_results
        }

    def run_greedy_baseline(
        self,
        total_budget: float
    ) -> Dict[str, Any]:
        """
        Execute Naive Greedy Benefit-to-Cost ratio baseline.
        Sorts grants in descending order of efficiency density: rho_i = S_i / c_i
        and funds applications sequentially until budget is exhausted.
        """
        df_sorted = self.scored_df.sort_values("efficiency_ratio", ascending=False).reset_index(drop=True)

        funded_ids = []
        allocated_budget = 0.0
        total_impact = 0.0
        total_beneficiaries = 0
        rural_allocated = 0.0
        sector_breakdown = {s: 0.0 for s in sorted(self.scored_df["sector"].unique())}

        for _, row in df_sorted.iterrows():
            cost = float(row["requested_amount"])
            if (allocated_budget + cost) <= total_budget:
                funded_ids.append(row["grant_id"])
                allocated_budget += cost
                total_impact += float(row["impact_score"])
                total_beneficiaries += int(row["direct_beneficiaries"])
                sector_breakdown[row["sector"]] += cost
                if row["region_tier"] == "Rural/Distressed":
                    rural_allocated += cost

        rural_pct = (rural_allocated / allocated_budget * 100.0) if allocated_budget > 0 else 0.0

        return {
            "technique": "Naive Greedy Benefit-Cost Heuristic (Baseline)",
            "solver": "Descending Efficiency Density Sort (S_i / c_i)",
            "status": "Heuristic Finished",
            "funded_grant_ids": funded_ids,
            "total_grants_funded": len(funded_ids),
            "total_grants_evaluated": len(df_sorted),
            "total_budget_available": round(total_budget, 2),
            "total_budget_utilized": round(allocated_budget, 2),
            "budget_utilization_pct": round((allocated_budget / total_budget * 100.0) if total_budget > 0 else 0.0, 2),
            "total_impact_score": round(total_impact, 2),
            "total_beneficiaries_reached": total_beneficiaries,
            "rural_budget_allocated": round(rural_allocated, 2),
            "rural_equity_pct_achieved": round(rural_pct, 2),
            "sector_allocation_breakdown": {s: round(v, 2) for s, v in sector_breakdown.items()}
        }

    def benchmark_all_strategies(
        self,
        total_budget: Optional[float] = None,
        rural_floor_pct: Optional[float] = None,
        max_sector_cap_pct: Optional[float] = None,
        gp_weights: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run 3-way comparative benchmark: Greedy Baseline vs. 0-1 ILP vs. Goal Programming.
        Computes explicit priority rankings (#1 to #N) and 3-tier allocations.
        """
        b_total = total_budget if total_budget is not None else self.default_budget
        r_floor = rural_floor_pct if rural_floor_pct is not None else self.rural_floor_pct
        s_cap = max_sector_cap_pct if max_sector_cap_pct is not None else self.max_sector_cap_pct

        # 1. Baseline
        res_baseline = self.run_greedy_baseline(total_budget=b_total)

        # 2. Technique 1: 0-1 ILP
        res_ilp = self.ilp_solver.solve(
            df=self.scored_df,
            total_budget=b_total,
            rural_floor_pct=r_floor,
            max_sector_cap_pct=s_cap
        )

        # 3. Technique 2: Goal Programming
        res_gp = self.gp_solver.solve(
            df=self.scored_df,
            total_budget=b_total,
            rural_floor_pct=r_floor,
            weights=gp_weights
        )

        # Explicit Priority Rankings and Tier assignment based on 0-1 ILP baseline solution
        ranked_df = self.compute_priority_rankings(self.scored_df)
        ranked_applications = self.assign_priority_tiers(
            ranked_df=ranked_df,
            funded_ids=res_ilp["funded_grant_ids"],
            total_budget=b_total,
            utilized_budget=res_ilp["total_budget_utilized"]
        )

        # Uplift calculations vs Baseline
        base_impact = res_baseline["total_impact_score"]
        ilp_impact = res_ilp["total_impact_score"]
        gp_impact = res_gp["total_impact_score"]

        ilp_uplift_abs = round(ilp_impact - base_impact, 2)
        ilp_uplift_pct = round((ilp_uplift_abs / base_impact * 100.0) if base_impact > 0 else 0.0, 2)

        gp_uplift_abs = round(gp_impact - base_impact, 2)
        gp_uplift_pct = round((gp_uplift_abs / base_impact * 100.0) if base_impact > 0 else 0.0, 2)

        summary = {
            "parameters": {
                "total_budget": round(b_total, 2),
                "rural_equity_floor_pct": round(r_floor * 100.0, 1),
                "max_sector_cap_pct": round(s_cap * 100.0, 1),
                "total_proposals": len(self.scored_df)
            },
            "ranked_applications": ranked_applications,
            "baseline_greedy": res_baseline,
            "integer_linear_programming": res_ilp,
            "goal_programming": res_gp,
            "comparative_analysis": {
                "ilp_impact_uplift_abs": ilp_uplift_abs,
                "ilp_impact_uplift_pct": ilp_uplift_pct,
                "gp_impact_uplift_abs": gp_uplift_abs,
                "gp_impact_uplift_pct": gp_uplift_pct,
                "budget_utilization_gap": round(res_ilp["budget_utilization_pct"] - res_baseline["budget_utilization_pct"], 2),
                "beneficiaries_additional_ilp": res_ilp["total_beneficiaries_reached"] - res_baseline["total_beneficiaries_reached"],
                "beneficiaries_additional_gp": res_gp["total_beneficiaries_reached"] - res_baseline["total_beneficiaries_reached"]
            }
        }

        if verbose:
            self._print_benchmark_table(summary)

        return summary

    def _print_benchmark_table(self, summary: Dict[str, Any]) -> None:
        """Print comparative performance benchmark table in terminal."""
        p = summary["parameters"]
        b = summary["baseline_greedy"]
        ilp = summary["integer_linear_programming"]
        gp = summary["goal_programming"]
        comp = summary["comparative_analysis"]
        dual = ilp["duality_analysis"]

        print("\n" + "=" * 100)
        print("    OPERATIONS RESEARCH GRANT PRIORITIZATION ENGINE: COMPARATIVE BENCHMARK (INR ₹)")
        print(f"    Total Budget: ₹{p['total_budget']:,.2f} | Proposals: {p['total_proposals']} | Rural Floor: {p['rural_equity_floor_pct']}%")
        print("=" * 100)
        header = f"{'Evaluation Metric':<32} | {'Greedy Heuristic (Base)':<20} | {'0-1 ILP (Taha Ch 9)':<20} | {'Goal Prog (Taha Ch 8)':<20}"
        print(header)
        print("-" * 100)

        rows = [
            ("Total Socio-Economic Impact", f"{b['total_impact_score']:,.2f} pts", f"{ilp['total_impact_score']:,.2f} pts", f"{gp['total_impact_score']:,.2f} pts"),
            ("Impact Uplift vs. Greedy (%)", "0.0% (Reference)", f"+{comp['ilp_impact_uplift_pct']:.2f}%", f"+{comp['gp_impact_uplift_pct']:.2f}%"),
            ("Total Budget Utilized", f"₹{b['total_budget_utilized']:,.2f}", f"₹{ilp['total_budget_utilized']:,.2f}", f"₹{gp['total_budget_utilized']:,.2f}"),
            ("Budget Utilization %", f"{b['budget_utilization_pct']:.1f}%", f"{ilp['budget_utilization_pct']:.1f}%", f"{gp['budget_utilization_pct']:.1f}%"),
            ("Number of Grants Funded", f"{b['total_grants_funded']} / {b['total_grants_evaluated']}", f"{ilp['total_grants_funded']} / {ilp['total_grants_evaluated']}", f"{gp['total_grants_funded']} / {gp['total_grants_evaluated']}"),
            ("Direct Lives Benefited", f"{b['total_beneficiaries_reached']:,}", f"{ilp['total_beneficiaries_reached']:,}", f"{gp['total_beneficiaries_reached']:,}"),
            ("Rural Equity Allocation", f"₹{b['rural_budget_allocated']:,.2f} ({b['rural_equity_pct_achieved']:.1f}%)", f"₹{ilp['rural_budget_allocated']:,.2f} ({ilp['rural_equity_pct_achieved']:.1f}%)", f"₹{gp['rural_budget_allocated']:,.2f} ({gp['rural_equity_pct_achieved']:.1f}%)"),
            ("Rural Equity Quota Status", "❌ Non-Enforced" if b['rural_equity_pct_achieved'] < p['rural_equity_floor_pct'] else "✅ Compliant", "✅ Strictly Enforced", "✅ Target Reconciled")
        ]

        for label, v_b, v_ilp, v_gp in rows:
            print(f"{label:<32} | {v_b:<20} | {v_ilp:<20} | {v_gp:<20}")

        print("-" * 100)
        print("  DUALITY & SENSITIVITY INSIGHTS (UNIT 2 - HAMDY A. TAHA CH 4):")
        print(f"  * Budget Dual Shadow Price (y*_budget): ₹{dual['budget_shadow_price_per_lakh']:.4f} score points per additional ₹1 Lakh allocated")
        print(f"  * Continuous LP Upper Bound:          {dual['lp_continuous_upper_bound']:,.2f} points")
        print(f"  * Integrality Gap (Cost of Indivisibility): {dual['integrality_gap_abs']:,.2f} points ({dual['integrality_gap_pct']:.2f}%)")
        print("=" * 100 + "\n")
