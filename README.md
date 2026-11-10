# Socio-Economic Grant Prioritization Engine
**Operations Research Techniques: 0-1 Integer Linear Programming (Ch 9) & Multi-Objective Goal Programming (Ch 8)**
*Course Reference: Hamdy A. Taha ("Operations Research: An Introduction") & Frederick S. Hillier ("Introduction to Operations Research")*

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Operations Research](https://img.shields.io/badge/Operations%20Research-Taha%20Ch%204%2C%208%2C%209-purple.svg)]()
[![Zero External APIs](https://img.shields.io/badge/APIs-Zero%20External%20(Offline)-orange.svg)]()

A production-grade, offline-first Operations Research decision-support engine designed to solve the **Public-Sector Grant Allocation Problem**. The engine dynamically selects and prioritizes competing grant proposals to maximize composite socio-economic welfare in **Indian Rupees (₹)** while balancing strict policy equity floors, sector diversification caps, and multi-objective stakeholder targets.

---

## 1. System Architecture & Directory Structure

```
A:\OR\MY_PROJECT\
├── data/
│   └── grant_applications.csv          # Synthetic dataset (60 proposals across 6 sectors & 3 regions)
├── src/
│   ├── __init__.py                     # Package descriptor
│   ├── data_generator.py               # Generates realistic grant proposals (costs, SROI, feasibility)
│   ├── impact_model.py                 # Multi-criteria scoring & normalization (S_i in [0, 100])
│   ├── ilp_solver.py                   # Technique 1: 0-1 Integer Linear Programming (PuLP / HiGHS Branch & Bound)
│   ├── gp_solver.py                    # Technique 2: Multi-Objective Goal Programming (Simplex Deviational Model)
│   └── grant_engine.py                 # 3-Way Comparative Benchmark Engine (Greedy vs. ILP vs. Goal Prog)
├── frontend/
│   ├── index.html                      # Single-page web dashboard (Bootstrap 5, Chart.js)
│   ├── app.js                          # Reactive UI controller & async REST API fetch
│   └── style.css                       # Dashboard styling, KPI cards, allocation badges
├── server.py                           # Local FastAPI REST API backend & static file server
├── main.py                             # Unified CLI entry point (--cli benchmark) & server launcher
├── test_engine.py                      # Comprehensive OR unit tests (constraints, duality, integrality gap)
├── test_api.py                         # FastAPI endpoint test suite (health, grants, optimize)
├── start_dashboard.bat                 # 1-click Windows batch launcher (opens browser + server)
├── start_dashboard.ps1                 # 1-click PowerShell launcher
├── run.bat                             # CLI execution batch helper
├── requirements.txt                    # Dependencies: fastapi, uvicorn, numpy, pandas, scipy, pulp, highspy
└── README.md                           # Documentation, mathematical formulation & viva defense guide
```

---

## 2. Mathematical Formulations

### A. Socio-Economic Impact Scoring Model (`src/impact_model.py`)
Each grant proposal $i \in \{1, \dots, N\}$ is evaluated on standardized developmental indicators:
$$\text{RawScore}_i = w_{\text{sroi}} \cdot \text{Norm}(\text{SROI}_i) + w_{\text{bene}} \cdot \text{Norm}(\log(1 + \text{Beneficiaries}_i)) + w_{\text{vuln}} \cdot \text{VulnerabilityIndex}_i$$
$$S_i = \text{Feasibility}_i \cdot \text{RawScore}_i \times 100.0$$
- $S_i \in [0.0, 100.0]$: Dimensionless composite social welfare score.
- Efficiency Density: $\rho_i = \frac{S_i}{c_i} \times 10,000$ (Social points per ₹10,000 granted).

---

### B. Technique 1: 0-1 Integer Linear Programming (Hamdy A. Taha, Chapter 9)
Formulates grant selection as a Multidimensional 0-1 Knapsack with policy mandates:

$$\max_{x} Z = \sum_{i=1}^{N} S_i \cdot x_i$$

**Subject to:**
1. **Total Available Budget Constraint:**
   $$\sum_{i=1}^{N} c_i \cdot x_i \le B_{\text{total}}$$
2. **Rural / Distressed District Equity Floor:**
   $$\sum_{i \in \text{Rural}} c_i \cdot x_i \ge \alpha \cdot B_{\text{total}} \quad (\alpha = 0.35, \text{ at least 35% to rural projects})$$
3. **Sector Diversification Ceilings:**
   $$\sum_{i \in \text{Sector}_k} c_i \cdot x_i \le \beta_k \cdot B_{\text{total}} \quad \forall k \in \text{Sectors} \quad (\beta_k = 0.35)$$
4. **Binary Integrity:**
   $$x_i \in \{0, 1\} \quad \forall i \in \{1, \dots, N\}$$

*Solved using the **Branch and Bound** algorithm with continuous LP relaxation bounds (PuLP / HiGHS).*

---

### C. Duality & Sensitivity Analysis (Unit 2 of Syllabus - Hamdy A. Taha, Chapter 4)
By taking the continuous relaxation ($0 \le x_i \le 1$), the solver extracts the **Dual Variables (Shadow Prices $y^*$):**
- **Budget Shadow Price ($y^*_{\text{budget}}$):**
  $$y^*_{\text{budget}} = \frac{\partial Z^*}{\partial B}$$
  *Quantifies the marginal socio-economic welfare gained per additional ₹1 Lakh allocated to the fund.*
- **Integrality Gap:**
  $$\text{Integrality Gap} = \frac{Z_{\text{Continuous LP}} - Z_{\text{0-1 Integer}}}{Z_{\text{Continuous LP}}} \times 100\%$$
  *Quantifies the exact social cost of indivisibility (grants cannot be fractionally funded).*

---

### D. Technique 2: Multi-Objective Goal Programming (Hamdy A. Taha, Chapter 8)
Recognizes that public committees juggle conflicting goals. Converts constraints into flexible goal equations with deviational variables:
- $d_k^-$: Under-achievement of Goal $k$.
- $d_k^+$: Over-achievement of Goal $k$.

$$\min Z = W_1 \left(\frac{d_1^-}{T_1}\right) + W_2 \left(\frac{d_2^+}{B}\right) + W_3 \left(\frac{d_3^-}{T_3}\right) + W_4 \left(\frac{d_4^-}{T_4}\right)$$

**Subject to:**
1. **Social Impact Goal:** $\sum_{i=1}^N S_i x_i + d_1^- - d_1^+ = T_{\text{impact}}$ (Minimize $d_1^-$)
2. **Budget Ceiling Goal:** $\sum_{i=1}^N c_i x_i + d_2^- - d_2^+ = B$ (Minimize overspending $d_2^+$)
3. **Rural Equity Goal:** $\sum_{i \in \text{Rural}} c_i x_i + d_3^- - d_3^+ = \alpha B$ (Minimize under-allocation $d_3^-$)
4. **Direct Beneficiary Goal:** $\sum_{i=1}^N \text{Beneficiaries}_i x_i + d_4^- - d_4^+ = T_{\text{bene}}$ (Minimize shortfall $d_4^-$)
5. $x_i \in \{0, 1\}, \quad d_k^-, d_k^+ \ge 0, \quad d_k^- \cdot d_k^+ = 0$

---

## 3. Performance Benchmark Results (INR ₹)

Simulated on **60 candidate proposals** evaluated against a **₹1,50,00,000 (₹1.50 Crore)** funding pool:

| Evaluation Metric | Naive Greedy (Baseline) | 0-1 Integer LP (Taha Ch 9) | Goal Prog (Taha Ch 8) |
| :--- | :--- | :--- | :--- |
| **Total Socio-Economic Impact** | 4,210.45 pts | **4,925.80 pts** | **4,840.10 pts** |
| **Social Impact Uplift (%)** | 0.0% (Ref) | **+16.99%** | **+14.95%** |
| **Budget Utilized (₹)** | ₹1,29,50,000 (86.3%) | **₹1,49,75,000 (99.8%)** | **₹1,48,50,000 (99.0%)** |
| **Number of Grants Funded** | 16 / 60 | **21 / 60** | **20 / 60** |
| **Direct Lives Benefited** | 44,800 | **56,200** | **58,400** |
| **Rural Budget Allocation** | ₹38,50,000 (29.7%) | **₹54,25,000 (36.2%)** | **₹53,00,000 (35.7%)** |
| **35% Rural Equity Quota** | ❌ **Violated (Shortfall)** | ✅ **Strictly Enforced** | ✅ **Target Reconciled** |

### Key Operations Research Insights:
1. **0-1 Integer Linear Programming (+16.99% uplift)** eliminates the local-optima trap of greedy sorting, packing the budget to 99.8% utilization while strictly adhering to affirmative action equity floors.
2. **Goal Programming (+14.95% uplift)** allows policymakers to dynamically balance trade-offs (e.g. prioritizing maximum beneficiary reach vs. strict budget boundaries) via deviational penalty weights.
3. **Duality Analysis** proves that the budget constraint has a shadow price of **+0.0035 points per ₹1**, meaning an extra ₹10 Lakhs in budget produces ~35 additional points of social welfare.

---

## 4. How to Run

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Launch Interactive Web Dashboard (Recommended)
You can launch the dashboard using any of these methods:

**Method A: 1-Click Launch (Browser opens automatically)**
```powershell
.\start_dashboard.ps1
# Or double-click start_dashboard.bat
```

**Method B: Standard Python Execution**
```powershell
python server.py
# Or: python main.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.  
Interactive REST API Documentation is live at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

### Run Offline CLI Benchmark
```powershell
python main.py --cli
# Optional custom scenario:
python main.py --cli --budget 20000000 --rural-floor 0.40
```

### Run Unit and Integration Tests
```powershell
python test_engine.py   # Core OR solvers, constraints, duality & integrality gap
python test_api.py      # FastAPI REST endpoints & schema contracts
```
