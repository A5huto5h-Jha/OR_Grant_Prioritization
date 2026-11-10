"""
impact_model.py
Multi-Criteria Socio-Economic Impact Scoring Model for Grant Prioritization.
Normalizes heterogeneous metrics (SROI, log beneficiaries, vulnerability, feasibility)
into an objective composite score S_i in [0, 100] without subjective arbitrary scales.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class SocioEconomicImpactModel:
    """
    Standardized multi-criteria evaluation model for grant applications.

    Composite Score Formula:
        Raw_Score_i = w_sroi * Norm(SROI_i)
                    + w_bene * Norm(log(1 + Beneficiaries_i))
                    + w_vuln * VulnerabilityIndex_i

        Final_Impact_S_i = Feasibility_i * Raw_Score_i * 100.0

    Where:
        - Norm(x) maps values to [0.0, 1.0] using min-max scaling across candidate proposals.
        - Feasibility_i in [0.60, 0.95] represents project execution probability / track record.
        - Resulting S_i is an intuitive score in [0.0, 100.0].
    """

    def __init__(
        self,
        weight_sroi: float = 0.35,
        weight_beneficiaries: float = 0.35,
        weight_vulnerability: float = 0.30
    ):
        """
        Parameters:
            weight_sroi: Importance of Social Return on Investment (economic multiplier).
            weight_beneficiaries: Importance of direct population coverage (human scale).
            weight_vulnerability: Importance of targeting marginalized / distressed groups (equity).
        """
        total = weight_sroi + weight_beneficiaries + weight_vulnerability
        if total <= 0:
            raise ValueError("Criterion weights must sum to a positive number.")
        self.w_sroi = weight_sroi / total
        self.w_bene = weight_beneficiaries / total
        self.w_vuln = weight_vulnerability / total

    def compute_impact_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich dataframe of proposals with standardized socio-economic scores (S_i)
        and efficiency density (S_i / c_i).
        """
        scored_df = df.copy()

        # Min-Max normalization for SROI
        sroi_min = scored_df["sroi_ratio"].min()
        sroi_max = scored_df["sroi_ratio"].max()
        sroi_range = (sroi_max - sroi_min) if (sroi_max - sroi_min) > 0 else 1.0
        norm_sroi = (scored_df["sroi_ratio"] - sroi_min) / sroi_range

        # Logarithmic normalization for Beneficiaries (prevents mega-projects from skewing linear scales)
        log_bene = np.log1p(scored_df["direct_beneficiaries"])
        bene_min = log_bene.min()
        bene_max = log_bene.max()
        bene_range = (bene_max - bene_min) if (bene_max - bene_min) > 0 else 1.0
        norm_bene = (log_bene - bene_min) / bene_range

        # Vulnerability index is already calibrated in [0.0, 1.0]
        norm_vuln = scored_df["vulnerability_index"].clip(0.0, 1.0)

        # Composite multi-criteria score adjusted by project feasibility track record
        raw_score = (
            self.w_sroi * norm_sroi +
            self.w_bene * norm_bene +
            self.w_vuln * norm_vuln
        )

        impact_score = raw_score * scored_df["execution_feasibility"] * 100.0
        scored_df["impact_score"] = np.round(impact_score, 2)

        # Efficiency ratio: Points of social welfare generated per ₹10,000 of grant capital
        # rho_i = (S_i / c_i) * 10,000
        scored_df["efficiency_ratio"] = np.round(
            (scored_df["impact_score"] / scored_df["requested_amount"]) * 10000.0, 4
        )

        return scored_df
