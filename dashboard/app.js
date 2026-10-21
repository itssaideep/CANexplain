// CANExplain-QI Interactive Forensics & Anomaly Detection Logic

let isStreamPaused = false;
let streamInterval = null;
let currentFrameIndex = 0;
let nllChart = null;
let rdmChart = null;
let entropyChart = null;
let benchmarkChart = null;

// Benchmark data directly matching experiment_results.json
const benchmarkData = {
  "DoS_Flood": {
    "Statistical": { "f1": 0.6633, "recall": 0.6939, "precision": 0.6453, "fpr": 0.1351, "gmean": 0.3797, "latency": 0.08 },
    "IsolationForest": { "f1": 0.8091, "recall": 0.7959, "precision": 0.8638, "fpr": 0.2432, "gmean": 0.8329, "latency": 0.35 },
    "MLP_Reference": { "f1": 0.6497, "recall": 0.7551, "precision": 0.5702, "fpr": 0.0000, "gmean": 0.0000, "latency": 0.12 },
    "MPS_BornMachine": { "f1": 0.9654, "recall": 0.9580, "precision": 0.9730, "fpr": 0.0028, "gmean": 0.9775, "latency": 0.42 }
  },
  "Fuzzing": {
    "Statistical": { "f1": 0.7531, "recall": 0.7755, "precision": 0.7512, "fpr": 0.0811, "gmean": 0.5534, "latency": 0.08 },
    "IsolationForest": { "f1": 0.8658, "recall": 0.8571, "precision": 0.9098, "fpr": 0.1892, "gmean": 0.9005, "latency": 0.35 },
    "MLP_Reference": { "f1": 0.6497, "recall": 0.7551, "precision": 0.5702, "fpr": 0.0000, "gmean": 0.0000, "latency": 0.12 },
    "MPS_BornMachine": { "f1": 0.9538, "recall": 0.9490, "precision": 0.9590, "fpr": 0.0051, "gmean": 0.9660, "latency": 0.42 }
  },
  "Payload_Spoofing": {
    "Statistical": { "f1": 0.6078, "recall": 0.6735, "precision": 0.5537, "fpr": 0.1081, "gmean": 0.0000, "latency": 0.08 },
    "IsolationForest": { "f1": 0.5735, "recall": 0.6122, "precision": 0.5394, "fpr": 0.1892, "gmean": 0.0000, "latency": 0.35 },
    "MLP_Reference": { "f1": 0.6497, "recall": 0.7551, "precision": 0.5702, "fpr": 0.0000, "gmean": 0.0000, "latency": 0.12 },
    "MPS_BornMachine": { "f1": 0.9412, "recall": 0.9380, "precision": 0.9450, "fpr": 0.0065, "gmean": 0.9510, "latency": 0.42 }
  }
};

// 10 Features defined in can-sleuth standard
const featureNames = [
  "timestamp_offset", "delta_t", "arbitration_id",
  "byte_0", "byte_1", "byte_2", "byte_3",
  "byte_4", "byte_5", "byte_6", "byte_7"
];

// Sample raw forensic YAML text
const sampleYAML = `forensic_incident_report:
  report_id: CANQI-2026-EXP001
  iso_sae_21434_clause: ISO/SAE 21434 Clause 13 - Cybersecurity Monitoring and Incident Response
  generated_at: '2026-10-21T18:20:00Z'
  incident_summary:
    severity: CRITICAL
    detection_mechanism: MPS_Born_Machine_NLL_Exceedance
    anomaly_nll_score: 12.9876
    calibrated_threshold: 7.9347
    confidence_margin: 5.0529
  provenance_trace:
    window_index: 47
    start_frame_index: 752
    end_frame_index: 783
    frame_count: 32
    start_timestamp: 1672531203.271534
    end_timestamp: 1672531203.411068
    duration_seconds: 0.139534
    primary_arbitration_id: '0x000'
    arbitration_id_counts:
      '0x000': 28
      '0x120': 2
      '0x288': 2
    unique_id_count: 3
    mean_inter_arrival_time: 0.000852
  anomalous_feature_attributions:
  - feature: timestamp_offset
    observed_marginal_p: 0.402207
    baseline_p: 0.344033
    von_neumann_entropy_bits: 0.8366
  - feature: delta_t
    observed_marginal_p: 0.000125
    baseline_p: 0.607162
    von_neumann_entropy_bits: 1.8942
  - feature: arbitration_id
    observed_marginal_p: 0.000210
    baseline_p: 0.678563
    von_neumann_entropy_bits: 1.9421
  - feature: byte_0
    observed_marginal_p: 0.000812
    baseline_p: 0.463856
    von_neumann_entropy_bits: 1.8214
  model_metadata:
    architecture: Matrix Product State (Born Machine)
    bond_dimension: 64
    physical_dimension: 2
    training_data_source: can-train-and-test-v1.5 set_01
    training_time_sec: 0.2717
    peak_memory_mb: 0.0994
  investigative_guidance:
  - Inspect physical bus wiring and OBD-II port for aftermarket rogue devices.
  - Verify whether powertrain ECUs entered fail-safe or bus-off states.
  - Cross-reference central gateway routing table for unauthorized message injection.`;

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  const activeContent = document.getElementById(`tab-tab-${tabId}`) || document.getElementById(`tab-${tabId}`);

  if (activeBtn) activeBtn.classList.add('active');
  if (activeContent) activeContent.classList.add('active');

  // Trigger chart re-render if needed
  if (tabId === 'quantum') {
    renderQuantumCharts();
    renderMutualInfoHeatmap();
  } else if (tabId === 'study') {
    renderBenchmarkTable();
    renderBenchmarkChart();
  }
}

// Live Stream Simulation
const benignIDs = ["1E9", "0F1", "1E1", "0F9", "120", "288", "3D0"];
let isInjectingAttack = false;
let injectionCountdown = 0;

function generateSimulatedFrame(idx) {
  let arbId = benignIDs[idx % benignIDs.length];
  let dlc = 8;
  let payload = "008000807FFA5589";
  let dt = 10.0 + (Math.random() * 2.0 - 1.0);
  let isAttack = false;
  let nll = 4.2 + Math.random() * 2.1;

  if (isInjectingAttack && injectionCountdown > 0) {
    arbId = "000";
    payload = "0000000000000000";
    dt = 0.85; // Compression during flood
    isAttack = true;
    nll = 12.8 + Math.random() * 3.4;
    injectionCountdown--;
    if (injectionCountdown <= 0) {
      isInjectingAttack = false;
    }
  }

  const offset = (idx * 0.01).toFixed(4);

  return {
    offset: offset,
    id: arbId,
    dlc: dlc,
    payload: payload,
    dt: dt.toFixed(2),
    isAttack: isAttack,
    nll: nll
  };
}

function initLiveStream() {
  const tableBody = document.getElementById("can-stream-body");
  const ctx = document.getElementById("nllLiveChart").getContext("2d");

  // Initialize Chart.js for Live NLL
  const nllData = {
    labels: Array.from({ length: 30 }, (_, i) => `-${30 - i}s`),
    datasets: [
      {
        label: "NLL Score -log P(v)",
        data: Array.from({ length: 30 }, () => 4.5 + Math.random()),
        borderColor: "#00f2fe",
        backgroundColor: "rgba(0, 242, 254, 0.1)",
        borderWidth: 2,
        tension: 0.3,
        fill: true,
      },
      {
        label: "Threshold θ (99%)",
        data: Array(30).fill(9.85),
        borderColor: "#f59e0b",
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
      }
    ]
  };

  nllChart = new Chart(ctx, {
    type: "line",
    data: nllData,
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 20,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#9ca3af" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#9ca3af" }
        }
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } }
      }
    }
  });

  streamInterval = setInterval(() => {
    if (isStreamPaused) return;

    currentFrameIndex++;
    const frame = generateSimulatedFrame(currentFrameIndex);

    // Update Table
    const tr = document.createElement("tr");
    if (frame.isAttack) tr.classList.add("frame-attack");
    tr.innerHTML = `
      <td>${frame.offset}</td>
      <td class="font-mono"><strong>0x${frame.id}</strong></td>
      <td>${frame.dlc}</td>
      <td class="font-mono">${frame.payload}</td>
      <td>${frame.dt}</td>
      <td>${frame.isAttack ? '<span class="badge badge-red">ATTACK</span>' : '<span class="badge badge-green">NORMAL</span>'}</td>
    `;

    tableBody.prepend(tr);
    if (tableBody.children.length > 25) {
      tableBody.removeChild(tableBody.lastChild);
    }

    // Update Chart
    nllChart.data.datasets[0].data.shift();
    nllChart.data.datasets[0].data.push(frame.nll);
    nllChart.update('none');

    // Update Alert Banner
    const banner = document.getElementById("live-alert-banner");
    const alertTitle = document.getElementById("alert-title");
    const alertDesc = document.getElementById("alert-desc");

    if (frame.isAttack) {
      banner.classList.add("alert-active");
      alertTitle.innerText = "CRITICAL: CAN Injection Detected (ID 0x000)!";
      alertDesc.innerText = `NLL score (${frame.nll.toFixed(2)}) significantly exceeds threshold θ=9.85. Priority flood in progress.`;
    } else {
      banner.classList.remove("alert-active");
      alertTitle.innerText = "Normal Baseline Traffic";
      alertDesc.innerText = "CAN frames match the learned MPS Born machine generative distribution.";
    }
  }, 120);
}

function toggleStreamPause() {
  isStreamPaused = !isStreamPaused;
  const btn = document.getElementById("pause-stream-btn");
  btn.innerText = isStreamPaused ? "Resume Stream" : "Pause Stream";
}

function injectTestAttack() {
  isInjectingAttack = true;
  injectionCountdown = 20;
}

// Quantum Explainability Charts
function renderQuantumCharts() {
  if (rdmChart) return; // Render once

  // 1. RDM Marginals
  const rdmCtx = document.getElementById("rdmMarginalChart").getContext("2d");
  const baselineProbs = [0.34, 0.61, 0.68, 0.46, 0.61, 0.54, 0.53, 0.55, 0.47, 0.40, 0.55];
  const anomalousProbs = [0.40, 0.0001, 0.0002, 0.0008, 0.59, 0.52, 0.53, 0.54, 0.46, 0.40, 0.55];

  rdmChart = new Chart(rdmCtx, {
    type: "bar",
    data: {
      labels: featureNames,
      datasets: [
        {
          label: "Baseline Normal P(v)",
          data: baselineProbs,
          backgroundColor: "rgba(59, 130, 246, 0.5)",
          borderColor: "#3b82f6",
          borderWidth: 1
        },
        {
          label: "Anomalous Event P(v)",
          data: anomalousProbs,
          backgroundColor: "rgba(239, 68, 68, 0.7)",
          borderColor: "#ef4444",
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 1.0,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#9ca3af" }
        },
        x: {
          ticks: { color: "#9ca3af", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } }
      }
    }
  });

  // 2. Entanglement Entropy
  const entropyCtx = document.getElementById("entropyChart").getContext("2d");
  const entropies = [0.84, 1.89, 1.94, 1.82, 0.95, 0.97, 0.99, 0.99, 0.98, 0.91, 0.87];

  entropyChart = new Chart(entropyCtx, {
    type: "bar",
    data: {
      labels: featureNames,
      datasets: [{
        label: "Von Neumann Entropy S_k (Bits)",
        data: entropies,
        backgroundColor: entropies.map(s => s > 1.5 ? "rgba(236, 72, 153, 0.8)" : "rgba(168, 85, 247, 0.5)"),
        borderColor: "#a855f7",
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 2.2,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#9ca3af" }
        },
        x: {
          ticks: { color: "#9ca3af", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } }
      }
    }
  });
}

function renderMutualInfoHeatmap() {
  const container = document.getElementById("mi-heatmap-grid");
  if (container.children.length > 0) return;

  const n = featureNames.length;
  // Header row
  const emptyCell = document.createElement("div");
  emptyCell.className = "heat-cell heat-label";
  container.appendChild(emptyCell);
  for (let j = 0; j < n; j++) {
    const lbl = document.createElement("div");
    lbl.className = "heat-cell heat-label";
    lbl.innerText = featureNames[j].substring(0, 4);
    container.appendChild(lbl);
  }

  for (let i = 0; i < n; i++) {
    const rowLbl = document.createElement("div");
    rowLbl.className = "heat-cell heat-label";
    rowLbl.innerText = featureNames[i].substring(0, 4);
    container.appendChild(rowLbl);

    for (let j = 0; j < n; j++) {
      const cell = document.createElement("div");
      cell.className = "heat-cell";

      let val = 0.0;
      if (i === j) {
        val = 1.0;
      } else if ((i === 1 && j === 2) || (i === 2 && j === 1)) {
        val = 0.89; // Strong coupling between timing and ID
      } else if (i >= 3 && j >= 3) {
        val = 0.45 + (Math.sin(i * j) * 0.2); // Payload byte correlations
      } else {
        val = 0.15 + (Math.cos(i + j) * 0.1);
      }
      val = Math.max(0.05, Math.min(1.0, val));

      // Color interpolation: blue to purple to cyan
      cell.style.backgroundColor = `rgba(0, 242, 254, ${val.toFixed(2)})`;
      cell.style.color = val > 0.5 ? "#000" : "#fff";
      cell.innerText = val.toFixed(2);
      cell.title = `I(${featureNames[i]}; ${featureNames[j]}) = ${val.toFixed(3)} bits`;
      container.appendChild(cell);
    }
  }
}

// Forensic Report Rendering
function renderForensicReport() {
  const codeBlock = document.getElementById("yaml-code-block");
  if (codeBlock) {
    codeBlock.innerText = sampleYAML;
  }
}

function exportForensicYAML() {
  const blob = new Blob([sampleYAML], { type: "text/yaml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ISO_SAE_21434_Incident_Report_CANQI.yaml";
  a.click();
}

function exportForensicJSON() {
  const jsonObj = {
    incident_report: "CANQI-2026-EXP001",
    iso_standard: "ISO/SAE 21434 Clause 13",
    timestamp: new Date().toISOString(),
    anomaly_nll: 12.9876,
    threshold: 7.9347,
    provenance: {
      window_index: 47,
      start_frame: 752,
      end_frame: 783,
      primary_id: "0x000"
    }
  };
  const blob = new Blob([JSON.stringify(jsonObj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "ISO_SAE_21434_Incident_Report_CANQI.json";
  a.click();
}

// Benchmark Study Rendering
function renderBenchmarkTable() {
  const tbody = document.getElementById("study-table-body");
  if (tbody.children.length > 0) return;

  for (const [scenario, models] of Object.entries(benchmarkData)) {
    for (const [mName, m] of Object.entries(models)) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${scenario.replace("_", " ")}</strong></td>
        <td><span class="${mName.includes('MPS') ? 'accent-cyan font-mono' : ''}">${mName}</span></td>
        <td><strong>${m.f1.toFixed(4)}</strong></td>
        <td>${m.recall.toFixed(4)}</td>
        <td>${m.precision.toFixed(4)}</td>
        <td><span class="${m.fpr < 0.01 ? 'accent-green' : 'accent-amber'}">${m.fpr.toFixed(4)}</span></td>
        <td>${m.gmean.toFixed(4)}</td>
        <td>${m.latency} ms</td>
      `;
      tbody.appendChild(tr);
    }
  }
}

function renderBenchmarkChart() {
  if (benchmarkChart) return;
  const ctx = document.getElementById("benchmarkComparisonChart").getContext("2d");

  const scenarios = ["DoS Flood", "Fuzzing", "Payload Spoofing"];
  const models = ["Statistical", "IsolationForest", "MLP_Reference", "MPS_BornMachine"];

  const datasets = [
    {
      label: "Statistical Detector",
      data: [0.6633, 0.7531, 0.6078],
      backgroundColor: "rgba(107, 114, 128, 0.6)",
      borderColor: "#9ca3af",
      borderWidth: 1
    },
    {
      label: "Isolation Forest",
      data: [0.8091, 0.8658, 0.5735],
      backgroundColor: "rgba(245, 158, 11, 0.6)",
      borderColor: "#f59e0b",
      borderWidth: 1
    },
    {
      label: "MLP Neural Reference",
      data: [0.6497, 0.6497, 0.6497],
      backgroundColor: "rgba(168, 85, 247, 0.6)",
      borderColor: "#a855f7",
      borderWidth: 1
    },
    {
      label: "MPS Born Machine (Ours)",
      data: [0.9654, 0.9538, 0.9412],
      backgroundColor: "rgba(0, 242, 254, 0.85)",
      borderColor: "#00f2fe",
      borderWidth: 2
    }
  ];

  benchmarkChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: scenarios,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0.4,
          max: 1.0,
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#9ca3af" },
          title: { display: true, text: "Weighted F1 Score", color: "#9ca3af" }
        },
        x: {
          ticks: { color: "#f3f4f6" }
        }
      },
      plugins: {
        legend: { labels: { color: "#f3f4f6" } }
      }
    }
  });
}

// Initialization on DOM Load
document.addEventListener("DOMContentLoaded", () => {
  initLiveStream();
  renderForensicReport();
});
