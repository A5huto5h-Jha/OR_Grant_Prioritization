/**
 * app.js
 * Frontend controller for Socio-Economic Grant Prioritization Dashboard (INR ₹).
 * Handles user controls, async REST calls, Chart.js reactivity,
 * explicit Priority Rankings (#1 to #N), 3-Tier Queuing, and Dynamic Proposal Ingestion.
 */

const API_BASE_URL = (window.location.protocol.startsWith("http") && window.location.port === "8000")
  ? window.location.origin
  : "http://127.0.0.1:8000";

let comparisonChart = null;
let sectorDoughnutChart = null;
let currentOptimizationData = null;
let activeStrategyKey = "ILP"; // "ILP", "GP", "Greedy"
let activeTableFilter = "ALL"; // "ALL", "FUNDED", "WAITLIST"

// Goal Programming Weights Profile
let currentGpWeights = {
  impact: 0.35,
  budget: 0.30,
  rural: 0.20,
  beneficiaries: 0.15
};

// Initialize on DOM Ready
document.addEventListener("DOMContentLoaded", async () => {
  initCharts();
  await checkApiHealth();
  await calculateOptimalPortfolio();
});

/**
 * Currency Formatter helper for Indian Rupees
 */
function formatINR(val, decimals = 2) {
  if (isNaN(val) || val === null) return "₹0.00";
  return "₹" + Number(val).toLocaleString("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}

/**
 * Initialize Chart.js instances
 */
function initCharts() {
  // 1. Comparison Bar Chart
  const ctxCompare = document.getElementById("strategyComparisonChart").getContext("2d");
  comparisonChart = new Chart(ctxCompare, {
    type: "bar",
    data: {
      labels: ["Greedy Baseline", "0-1 Integer LP (Ch 9)", "Goal Prog (Ch 8)"],
      datasets: [
        {
          label: "Total Impact Score (pts)",
          data: [0, 0, 0],
          backgroundColor: ["#94a3b8", "#2563eb", "#0d9488"],
          borderRadius: 6,
          yAxisID: "y"
        },
        {
          label: "Direct Lives (Thousands)",
          data: [0, 0, 0],
          backgroundColor: ["#cbd5e1", "#93c5fd", "#5eead4"],
          borderRadius: 6,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Social Impact (pts)" },
          grid: { color: "rgba(226, 232, 240, 0.6)" }
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Lives Benefited (k)" },
          grid: { drawOnChartArea: false }
        },
        x: { grid: { display: false } }
      }
    }
  });

  // 2. Sectoral Doughnut Chart
  const ctxSector = document.getElementById("sectorChart").getContext("2d");
  sectorDoughnutChart = new Chart(ctxSector, {
    type: "doughnut",
    data: {
      labels: ["Healthcare", "Education", "Agriculture & Water", "Clean Energy", "Infrastructure", "Women"],
      datasets: [{
        data: [1, 1, 1, 1, 1, 1],
        backgroundColor: [
          "#2563eb", "#10b981", "#f59e0b", "#06b6d4", "#8b5cf6", "#ec4899"
        ],
        borderWidth: 2,
        borderColor: "#ffffff"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom", labels: { boxWidth: 10, font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${formatINR(ctx.raw, 0)}`
          }
        }
      }
    }
  });
}

/**
 * Health check endpoint
 */
async function checkApiHealth() {
  const badge = document.getElementById("apiStatusBadge");
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`);
    if (res.ok) {
      badge.className = "badge py-2 px-3 rounded-pill badge-connected";
      badge.innerHTML = `<i class="bi bi-check-circle-fill text-success"></i> Solvers Ready (Port 8000)`;
    } else {
      throw new Error();
    }
  } catch (err) {
    badge.className = "badge py-2 px-3 rounded-pill badge-disconnected";
    badge.innerHTML = `<i class="bi bi-exclamation-triangle-fill"></i> Offline`;
  }
}

/**
 * Sync Input Controls
 */
function syncBudget(val) {
  const num = Number(val);
  document.getElementById("budgetRange").value = num;
  document.getElementById("budgetInput").value = num;
  document.getElementById("budgetValueDisplay").innerText = formatINR(num, 0);
}

function syncRural(val) {
  const num = Number(val);
  document.getElementById("ruralRange").value = num;
  document.getElementById("ruralValueDisplay").innerText = `${num}% Minimum`;
}

function syncSectorCap(val) {
  const num = Number(val);
  document.getElementById("sectorCapRange").value = num;
  document.getElementById("sectorCapDisplay").innerText = `${num}% Maximum`;
}

function applyPreset(budget, ruralPct, sectorCap, label) {
  syncBudget(budget);
  syncRural(ruralPct);
  syncSectorCap(sectorCap);
  calculateOptimalPortfolio();
}

function applyGoalProfile(profile) {
  if (profile === "impact_first") {
    currentGpWeights = { impact: 0.60, budget: 0.20, rural: 0.10, beneficiaries: 0.10 };
  } else if (profile === "equity_first") {
    currentGpWeights = { impact: 0.20, budget: 0.20, rural: 0.50, beneficiaries: 0.10 };
  } else if (profile === "budget_strict") {
    currentGpWeights = { impact: 0.25, budget: 0.55, rural: 0.10, beneficiaries: 0.10 };
  } else {
    currentGpWeights = { impact: 0.35, budget: 0.30, rural: 0.20, beneficiaries: 0.15 };
  }
  calculateOptimalPortfolio();
}

/**
 * Run Optimization
 */
async function calculateOptimalPortfolio() {
  const btn = document.getElementById("optimizeBtn");
  btn.innerHTML = `<i class="bi bi-arrow-repeat spin me-1"></i> Optimizing Portfolio...`;
  btn.disabled = true;

  const budget = parseFloat(document.getElementById("budgetInput").value);
  const ruralFloor = parseFloat(document.getElementById("ruralRange").value) / 100.0;
  const sectorCap = parseFloat(document.getElementById("sectorCapRange").value) / 100.0;

  try {
    const res = await fetch(`${API_BASE_URL}/api/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        total_budget: budget,
        rural_floor_pct: ruralFloor,
        max_sector_cap_pct: sectorCap,
        weight_impact: currentGpWeights.impact,
        weight_budget: currentGpWeights.budget,
        weight_rural: currentGpWeights.rural,
        weight_beneficiaries: currentGpWeights.beneficiaries
      })
    });

    const responseJson = await res.json();
    currentOptimizationData = responseJson.data;
    updateUIWithResults();
  } catch (err) {
    console.error("Optimization call failed:", err);
  } finally {
    btn.innerHTML = `<i class="bi bi-cpu-fill me-1"></i> Re-Calculate Optimal Portfolio (Run Solvers)`;
    btn.disabled = false;
  }
}

/**
 * Update UI Elements with Solvers Results
 */
function updateUIWithResults() {
  if (!currentOptimizationData) return;

  const d = currentOptimizationData;
  const b = d.baseline_greedy;
  const ilp = d.integer_linear_programming;
  const gp = d.goal_programming;
  const dual = ilp.duality_analysis;

  // 1. Update Duality Card (Unit 2)
  document.getElementById("dualShadowPriceDisplay").innerText = `+${dual.budget_shadow_price_per_lakh.toFixed(4)} pts / ₹1 Lakh`;
  document.getElementById("lpUpperBoundDisplay").innerText = `${dual.lp_continuous_upper_bound.toLocaleString()} pts`;
  document.getElementById("integralityGapDisplay").innerText = `${dual.integrality_gap_pct.toFixed(2)}% (${dual.integrality_gap_abs} pts)`;

  // 2. Update Charts
  comparisonChart.data.datasets[0].data = [
    b.total_impact_score,
    ilp.total_impact_score,
    gp.total_impact_score
  ];
  comparisonChart.data.datasets[1].data = [
    Math.round(b.total_beneficiaries_reached / 1000),
    Math.round(ilp.total_beneficiaries_reached / 1000),
    Math.round(gp.total_beneficiaries_reached / 1000)
  ];
  comparisonChart.update();

  // 3. Render Active Strategy View & Table
  renderActiveStrategyKPIs();
  renderGrantsTable();
}

/**
 * Switch Active Strategy Tab
 */
function switchActiveStrategy(key) {
  activeStrategyKey = key;
  renderActiveStrategyKPIs();
  renderGrantsTable();
}

/**
 * Update KPI cards for currently selected strategy
 */
function renderActiveStrategyKPIs() {
  if (!currentOptimizationData) return;

  let strat = currentOptimizationData.integer_linear_programming;
  let detailText = "Technique 1: 0-1 Integer Linear Program (Branch & Bound, Hamdy A. Taha Ch 9)";
  let upliftText = `+${currentOptimizationData.comparative_analysis.ilp_impact_uplift_pct.toFixed(2)}% vs. Baseline`;

  if (activeStrategyKey === "GP") {
    strat = currentOptimizationData.goal_programming;
    detailText = "Technique 2: Multi-Objective Goal Program (Deviational Simplex, Hamdy A. Taha Ch 8)";
    upliftText = `+${currentOptimizationData.comparative_analysis.gp_impact_uplift_pct.toFixed(2)}% vs. Baseline`;
  } else if (activeStrategyKey === "Greedy") {
    strat = currentOptimizationData.baseline_greedy;
    detailText = "Baseline Heuristic: Sorted by Efficiency Density (S_i / c_i)";
    upliftText = "0.0% (Reference Baseline)";
  }

  document.getElementById("strategyDetailBadge").innerText = detailText;
  document.getElementById("kpiTotalImpact").innerText = `${strat.total_impact_score.toLocaleString()} pts`;
  document.getElementById("kpiUpliftPct").innerText = upliftText;
  document.getElementById("kpiBudgetUtilized").innerText = formatINR(strat.total_budget_utilized, 0);
  document.getElementById("kpiBudgetPct").innerText = `${strat.budget_utilization_pct.toFixed(1)}% of available`;
  document.getElementById("kpiGrantsCount").innerText = `${strat.total_grants_funded} / ${strat.total_grants_evaluated}`;
  document.getElementById("kpiBeneficiaries").innerText = `${strat.total_beneficiaries_reached.toLocaleString()} direct lives`;

  document.getElementById("kpiRuralPct").innerText = `${strat.rural_equity_pct_achieved.toFixed(1)}%`;
  const ruralStatus = document.getElementById("kpiRuralStatus");
  if (strat.rural_equity_pct_achieved >= currentOptimizationData.parameters.rural_equity_floor_pct) {
    ruralStatus.className = "badge bg-success-subtle text-success";
    ruralStatus.innerText = "Quota Compliant";
  } else {
    ruralStatus.className = "badge bg-danger-subtle text-danger";
    ruralStatus.innerText = "Quota Shortfall";
  }

  // Update Doughnut Chart for active strategy
  const sec = strat.sector_allocation_breakdown;
  sectorDoughnutChart.data.labels = Object.keys(sec);
  sectorDoughnutChart.data.datasets[0].data = Object.values(sec);
  sectorDoughnutChart.update();
}

/**
 * Filter Table by Tier
 */
function setFilter(filterType) {
  activeTableFilter = filterType;
  document.getElementById("btnFilterAll").className = `btn btn-outline-secondary ${filterType === 'ALL' ? 'active' : ''}`;
  document.getElementById("btnFilterFunded").className = `btn btn-outline-secondary ${filterType === 'FUNDED' ? 'active' : ''}`;
  document.getElementById("btnFilterWaitlist").className = `btn btn-outline-secondary ${filterType === 'WAITLIST' ? 'active' : ''}`;
  renderGrantsTable(document.getElementById("tableSearchInput").value);
}

/**
 * Render Table of Grants with Explicit Priority Ranks & Tiers
 */
function renderGrantsTable(filterQuery = "") {
  const tbody = document.getElementById("grantsTableBody");
  if (!currentOptimizationData || !currentOptimizationData.ranked_applications) return;

  const applications = currentOptimizationData.ranked_applications;
  const fundedSet = new Set(
    activeStrategyKey === "ILP"
      ? currentOptimizationData.integer_linear_programming.funded_grant_ids
      : (activeStrategyKey === "GP"
          ? currentOptimizationData.goal_programming.funded_grant_ids
          : currentOptimizationData.baseline_greedy.funded_grant_ids)
  );

  const query = filterQuery.toLowerCase().trim();

  const filteredApps = applications.filter(app => {
    // 1. Text Search Filter
    const matchesQuery = !query ||
      app.title.toLowerCase().includes(query) ||
      app.sector.toLowerCase().includes(query) ||
      app.grant_id.toLowerCase().includes(query);

    if (!matchesQuery) return false;

    // 2. Tier Button Filter
    const isFunded = fundedSet.has(app.grant_id);
    if (activeTableFilter === "FUNDED" && !isFunded) return false;
    if (activeTableFilter === "WAITLIST" && (isFunded || app.tier_code !== "TIER_2")) return false;

    return true;
  });

  if (filteredApps.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-4 text-muted">No grant proposals match the selected filter.</td></tr>`;
    return;
  }

  const rows = filteredApps.map(app => {
    const isFunded = fundedSet.has(app.grant_id);
    let badgeClass = "badge-tier-3";
    let badgeText = "Tier 3: Deferred";
    let icon = "bi-dash-circle";

    if (isFunded) {
      badgeClass = "badge-tier-1";
      badgeText = "Tier 1: Funded";
      icon = "bi-check-circle-fill text-success";
    } else if (app.tier_code === "TIER_2") {
      badgeClass = "badge-tier-2";
      badgeText = `Tier 2: Waitlist #${app.waitlist_pos || 1}`;
      icon = "bi-hourglass-split text-warning";
    }

    const rankClass = app.priority_rank <= 3 ? "rank-top3" : "rank-normal";

    return `
      <tr>
        <td class="text-center">
          <span class="rank-badge ${rankClass}">#${app.priority_rank}</span>
        </td>
        <td>
          <div class="fw-semibold text-truncate" style="max-width: 250px;" title="${app.title}">${app.title}</div>
          <small class="font-monospace text-muted">${app.grant_id}</small>
        </td>
        <td><span class="badge bg-light text-dark border">${app.sector}</span></td>
        <td><small class="text-muted">${app.region_tier}</small></td>
        <td class="text-end fw-semibold">${formatINR(app.requested_amount, 0)}</td>
        <td class="text-center fw-bold text-primary">${Number(app.impact_score).toFixed(1)}</td>
        <td class="text-center">
          <span class="badge ${badgeClass} px-2 py-1 d-inline-flex align-items-center gap-1">
            <i class="bi ${icon}"></i>
            <span>${badgeText}</span>
          </span>
        </td>
        <td>
          <small class="text-muted text-truncate d-block" style="max-width: 220px;" title="${app.recommendation}">${app.recommendation}</small>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = rows.join("");
}

function filterTable(val) {
  renderGrantsTable(val);
}

/**
 * Handle Submission of New Grant Proposal
 */
async function submitNewProposal() {
  const submitBtn = document.getElementById("submitProposalBtn");
  submitBtn.disabled = true;
  submitBtn.innerHTML = `<i class="bi bi-arrow-repeat spin me-1"></i> Optimizing Portfolio...`;

  const payload = {
    title: document.getElementById("newTitle").value,
    sector: document.getElementById("newSector").value,
    region_tier: document.getElementById("newRegion").value,
    requested_amount: parseFloat(document.getElementById("newAmount").value),
    direct_beneficiaries: parseInt(document.getElementById("newBeneficiaries").value),
    sroi_ratio: parseFloat(document.getElementById("newSroi").value),
    vulnerability_index: parseFloat(document.getElementById("newVuln").value),
    execution_feasibility: parseFloat(document.getElementById("newFeasibility").value),
    evaluation_budget: parseFloat(document.getElementById("budgetInput").value)
  };

  try {
    const res = await fetch(`${API_BASE_URL}/api/grants/add`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const responseJson = await res.json();
    const result = responseJson.data;

    // Close Modal
    const modalEl = document.getElementById("newGrantModal");
    const modalInstance = bootstrap.Modal.getInstance(modalEl);
    if (modalInstance) modalInstance.hide();

    // Reset Form
    document.getElementById("newGrantForm").reset();

    // Update state and refresh UI
    currentOptimizationData = result.updated_benchmark;
    updateUIWithResults();

    // Show Flash Alert with detailed OR evaluation
    showSubmissionAlert(result);
  } catch (err) {
    alert("Error submitting proposal: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<i class="bi bi-calculator-fill me-1"></i> Evaluate & Prioritize Proposal`;
  }
}

/**
 * Display Celebratory Evaluation Banner
 */
function showSubmissionAlert(result) {
  const container = document.getElementById("newGrantAlertContainer");
  const isFunded = result.is_funded_ilp;
  const alertType = isFunded ? "alert-success" : "alert-warning";
  const icon = isFunded ? "bi-check-circle-fill text-success" : "bi-exclamation-circle-fill text-warning";

  container.innerHTML = `
    <div class="alert ${alertType} alert-dismissible fade show shadow-sm border-2 mb-4" role="alert">
      <div class="d-flex align-items-start gap-3">
        <i class="bi ${icon} fs-3"></i>
        <div class="flex-grow-1">
          <h5 class="alert-heading fw-bold mb-1">
            Proposal Evaluated: ${result.grant_id} — "${result.title}"
          </h5>
          <div class="d-flex flex-wrap gap-3 my-2">
            <span><strong>Computed Social Score:</strong> <span class="badge bg-primary fs-6">${result.impact_score.toFixed(1)} pts</span></span>
            <span><strong>Priority Rank:</strong> <span class="badge bg-dark fs-6">#${result.priority_rank} of ${result.total_evaluated}</span></span>
            <span><strong>Status Determination:</strong> <strong class="text-uppercase">${result.verdict}</strong></span>
          </div>
          <div class="small text-muted">${result.verdict_description}</div>
        </div>
      </div>
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    </div>
  `;
}

/**
 * Reset default proposals
 */
async function resetDefaultProposals() {
  if (!confirm("Reset grant proposals to the initial 60 proposals?")) return;
  try {
    await fetch(`${API_BASE_URL}/api/grants/reset`, { method: "POST" });
    await calculateOptimalPortfolio();
  } catch (err) {
    console.error("Failed to reset:", err);
  }
}
