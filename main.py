"""
main.py
Unified Entry Point for Socio-Economic Grant Prioritization Engine.
Supports:
  - Local FastAPI API Server & Web Dashboard (default)
  - Offline CLI Benchmark Mode (--cli)
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import argparse

from server import run_server
from src.grant_engine import GrantPrioritizationEngine


def run_cli_benchmark(args):
    """Run full 3-way comparative benchmark in terminal."""
    print("\n[Phase 1] Initializing OR Grant Prioritization Engine...")
    engine = GrantPrioritizationEngine(
        dataset_path=args.dataset,
        total_budget=args.budget,
        rural_floor_pct=args.rural_floor,
        max_sector_cap_pct=args.sector_cap,
        random_seed=args.seed
    )

    print(f"[Phase 2] Evaluating {len(engine.scored_df)} proposals against budget of ₹{args.budget:,.2f}...")
    engine.benchmark_all_strategies(
        total_budget=args.budget,
        rural_floor_pct=args.rural_floor,
        max_sector_cap_pct=args.sector_cap,
        verbose=True
    )
    print("[Complete] Optimization and duality analysis completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Socio-Economic Grant Prioritization Engine")
    parser.add_argument("--cli", action="store_true", help="Run offline CLI comparative benchmark")
    parser.add_argument("--budget", type=float, default=15000000.0, help="Total grant budget in ₹ (default: ₹1.5 Cr)")
    parser.add_argument("--rural-floor", type=float, default=0.35, help="Rural equity quota fraction (default: 0.35)")
    parser.add_argument("--sector-cap", type=float, default=0.35, help="Max budget cap per sector (default: 0.35)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API Port (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation")
    parser.add_argument("--dataset", type=str, default="data/grant_applications.csv", help="Path to grant CSV")
    args = parser.parse_args()

    if args.cli:
        run_cli_benchmark(args)
    else:
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
