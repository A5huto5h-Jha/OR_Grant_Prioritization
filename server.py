"""
server.py
Local FastAPI Backend for Grant Prioritization Engine.
Exposes REST endpoints for:
  - 0-1 Integer Linear Programming (Taha Ch 9)
  - Multi-Objective Goal Programming (Taha Ch 8)
  - Duality Analysis & Shadow Prices (Taha Ch 4)
  - Explicit Ordinal Priority Ranking (#1 to #N) & 3-Tier Queuing
  - Dynamic Proposal Submission & Live Impact/Funding Evaluation
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import pandas as pd

from src.grant_engine import GrantPrioritizationEngine

# Initialize FastAPI app
app = FastAPI(
    title="Socio-Economic Grant Prioritization Engine API",
    description="Operations Research Decision Portal (0-1 ILP & Goal Programming, Hamdy A. Taha)",
    version="1.1.0"
)

# CORS enabled for local frontend execution
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core OR Engine
engine = GrantPrioritizationEngine(
    dataset_path="data/grant_applications.csv",
    total_budget=15000000.0,
    rural_floor_pct=0.35,
    max_sector_cap_pct=0.35
)


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------
class OptimizationRequest(BaseModel):
    total_budget: float = Field(default=15000000.0, gt=0, description="Total grant budget available (₹)")
    rural_floor_pct: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Rural equity quota (0.0 to 1.0)")
    max_sector_cap_pct: Optional[float] = Field(default=0.35, ge=0.1, le=1.0, description="Max budget cap per sector (0.1 to 1.0)")
    weight_impact: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Goal Programming weight: Impact")
    weight_budget: Optional[float] = Field(default=0.30, ge=0.0, le=1.0, description="Goal Programming weight: Budget ceiling")
    weight_rural: Optional[float] = Field(default=0.20, ge=0.0, le=1.0, description="Goal Programming weight: Rural equity")
    weight_beneficiaries: Optional[float] = Field(default=0.15, ge=0.0, le=1.0, description="Goal Programming weight: Beneficiaries")


class NewGrantRequest(BaseModel):
    title: str = Field(..., min_length=3, description="Project title")
    sector: str = Field(..., description="Healthcare, Education, Agriculture & Water, Clean Energy, etc.")
    region_tier: str = Field(..., description="Rural/Distressed, Semi-Urban, Urban")
    requested_amount: float = Field(..., gt=10000, description="Funding requested in INR (₹)")
    direct_beneficiaries: int = Field(..., ge=1, description="Estimated direct lives benefited")
    sroi_ratio: float = Field(default=3.0, ge=0.5, le=10.0, description="Social Return on Investment multiplier")
    vulnerability_index: float = Field(default=0.75, ge=0.0, le=1.0, description="Target community distress index (0 to 1)")
    execution_feasibility: float = Field(default=0.85, ge=0.1, le=1.0, description="NGO execution track record / probability")
    evaluation_budget: Optional[float] = Field(default=None, description="Optional evaluation budget context")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend connectivity."""
    return {
        "status": "healthy",
        "service": "Grant Prioritization OR Decision Engine",
        "version": "1.1.0",
        "textbook_reference": "Hamdy A. Taha (Operations Research, Ch 4, 8, 9)",
        "solvers": [
            "0-1 Integer Linear Programming (Branch and Bound / Simplex)",
            "Multi-Objective Goal Programming (Deviational Variables)",
            "Continuous LP Duality & Shadow Price Analysis"
        ],
        "features": [
            "0-1 Integer Linear Programming (Branch and Bound / Simplex)",
            "Multi-Objective Goal Programming (Deviational Variables)",
            "Continuous LP Duality & Shadow Price Analysis",
            "Explicit Ordinal Priority Rankings (#1 to #N)",
            "3-Tier Queuing (Tier 1: Funded, Tier 2: Waitlist, Tier 3: Deferred)",
            "Dynamic Proposal Ingestion & Real-Time Impact Scoring"
        ]
    }


@app.get("/api/grants")
def get_grant_applications():
    """Returns candidate grant applications with computed socio-economic scores."""
    ranked_df = engine.compute_priority_rankings(engine.scored_df)
    records = ranked_df.to_dict(orient="records")
    return {
        "total_count": len(records),
        "applications": records
    }


@app.post("/api/optimize")
def optimize_grants(payload: OptimizationRequest):
    """
    Core Optimization Endpoint.
    Runs 3-way comparative benchmark: Baseline Greedy vs. 0-1 ILP vs. Goal Programming.
    Includes explicit Priority Ranks and 3-Tier classification.
    """
    try:
        gp_weights = {
            "impact": payload.weight_impact or 0.35,
            "budget": payload.weight_budget or 0.30,
            "rural": payload.weight_rural or 0.20,
            "beneficiaries": payload.weight_beneficiaries or 0.15
        }

        results = engine.benchmark_all_strategies(
            total_budget=payload.total_budget,
            rural_floor_pct=payload.rural_floor_pct,
            max_sector_cap_pct=payload.max_sector_cap_pct,
            gp_weights=gp_weights,
            verbose=False
        )

        return {
            "status": "success",
            "data": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grant optimization failed: {str(e)}"
        )


@app.post("/api/grants/add")
def add_new_grant(payload: NewGrantRequest):
    """
    Dynamic Proposal Evaluation Endpoint.
    Adds a new grant application, computes its multi-criteria socio-economic score,
    and returns its exact priority ranking and funding determination.
    """
    try:
        result = engine.add_grant_application(
            title=payload.title,
            sector=payload.sector,
            region_tier=payload.region_tier,
            requested_amount=payload.requested_amount,
            direct_beneficiaries=payload.direct_beneficiaries,
            sroi_ratio=payload.sroi_ratio,
            vulnerability_index=payload.vulnerability_index,
            execution_feasibility=payload.execution_feasibility,
            evaluation_budget=payload.evaluation_budget
        )
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to evaluate new grant proposal: {str(e)}"
        )


@app.post("/api/grants/reset")
def reset_proposals():
    """Resets the grant pool back to the default synthetic baseline."""
    global engine
    engine = GrantPrioritizationEngine(
        dataset_path="data/grant_applications.csv",
        total_budget=15000000.0,
        rural_floor_pct=0.35,
        max_sector_cap_pct=0.35
    )
    return {"status": "success", "message": "Grant pool reset to initial 60 proposals."}


# -----------------------------------------------------------------------------
# Static Files & Dashboard Serving
# -----------------------------------------------------------------------------
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard_root():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>Frontend index.html not found</h3>", status_code=404)


@app.get("/app.js")
def serve_app_js():
    js_file = os.path.join(frontend_dir, "app.js")
    if os.path.exists(js_file):
        with open(js_file, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="application/javascript")
    return Response(content="", status_code=404)


@app.get("/style.css")
def serve_style_css():
    css_file = os.path.join(frontend_dir, "style.css")
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            return Response(content=f.read(), media_type="text/css")
    return Response(content="", status_code=404)


def run_server(host: str = "127.0.0.1", port: int = 8000):
    print("=" * 80)
    print(" SOCIO-ECONOMIC GRANT PRIORITIZATION ENGINE - LOCAL FASTAPI SERVER")
    print(f" Dashboard: http://{host}:{port}")
    print(f" API Docs:  http://{host}:{port}/docs")
    print("=" * 80)
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
