"""
Lens Review UI — human-in-the-loop feedback tool.

Reads lens_mnist_analysis.json (produced by train_and_analyze.py) and serves
a web UI where you can review each failure, see the actual digit image, read
the LLM hypothesis, and record your own verdict.

Feedback is saved to lens_mnist_human_feedback.json after every submission.

Run:  uv run python review_ui.py
Then open http://localhost:8000
"""

import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from torchvision import datasets, transforms

import sys

_run_name = None
for _a in sys.argv[1:]:
    if _a.startswith("--run-name="):
        _run_name = _a.split("=", 1)[1]

if _run_name:
    ANALYSIS_PATH = f"runs/{_run_name}/analysis.json"
    FEEDBACK_PATH = f"runs/{_run_name}/human_feedback.json"
else:
    # fallback: look for the most recent run in runs/
    import glob as _glob
    _candidates = _glob.glob("runs/*/analysis.json")
    if _candidates:
        ANALYSIS_PATH = sorted(_candidates)[-1]
        FEEDBACK_PATH = ANALYSIS_PATH.replace("analysis.json", "human_feedback.json")
        print(f"Auto-selected run: {ANALYSIS_PATH}")
    else:
        ANALYSIS_PATH = "lens_mnist_analysis.json"
        FEEDBACK_PATH = "lens_mnist_human_feedback.json"

app = FastAPI(title="Lens Review UI")

# ─── Load data once at startup ────────────────────────────────────────────────
analysis: dict = {}
failures: list = []
cluster_labels: dict[int, str] = {}
dim_clusters: dict = {}   # dimensional_clusters.json content if present
feedback: dict[str, dict] = {}   # keyed by str(image_idx)

# Test images loaded into memory for rendering
test_images: dict[int, np.ndarray] = {}   # idx -> HxWxC or HxW float array
_dataset_type: str = "mnist"  # "mnist" or "cifar"


def load_images():
    global _dataset_type
    transform = transforms.ToTensor()
    # Detect dataset type from analysis failures
    if failures and "true_class" in failures[0]:
        _dataset_type = "cifar"
        ds = datasets.CIFAR10(".", train=False, download=False, transform=None)
        for idx in range(len(ds)):
            img, _ = ds[idx]
            test_images[idx] = np.array(img)  # HxWx3 uint8
    else:
        _dataset_type = "mnist"
        ds = datasets.MNIST(".", train=False, download=True, transform=transform)
        for idx in range(len(ds)):
            img, _ = ds[idx]
            test_images[idx] = img.squeeze().numpy()


def tensor_to_png_b64(arr: np.ndarray) -> str:
    from PIL import Image
    if arr.ndim == 3:  # CIFAR RGB: HxWx3 uint8
        pil = Image.fromarray(arr, mode="RGB").resize((224, 224), Image.NEAREST)
    else:  # MNIST grayscale: HxW float
        img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8, mode="L").resize((224, 224), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.on_event("startup")
def startup():
    global analysis, failures, cluster_labels, feedback

    if not Path(ANALYSIS_PATH).exists():
        print(f"WARNING: {ANALYSIS_PATH} not found. Run train_and_analyze.py first.")
        return

    with open(ANALYSIS_PATH) as f:
        analysis = json.load(f)

    failures = analysis.get("failures", [])

    for c in analysis.get("clusters", []):
        cluster_labels[c["cluster_id"]] = c["label"]

    if Path(FEEDBACK_PATH).exists():
        with open(FEEDBACK_PATH) as f:
            feedback = json.load(f)

    dim_path = ANALYSIS_PATH.replace("analysis.json", "dimensional_clusters.json")
    if Path(dim_path).exists():
        with open(dim_path) as f:
            dim_clusters = json.load(f)
        print(f"Loaded dimensional clusters from {dim_path}")

    print(f"Loaded {len(failures)} failures, {len(feedback)} existing feedback entries")
    print("Loading test images...")
    load_images()
    print("Ready.")


# ─── API ──────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    summary = analysis.get("summary", {})
    reviewed = len(feedback)
    agreed = sum(1 for v in feedback.values() if v.get("verdict") == "agree")
    disagreed = sum(1 for v in feedback.values() if v.get("verdict") == "disagree")
    skipped = sum(1 for v in feedback.values() if v.get("verdict") == "skip")
    return {
        "total_failures": len(failures),
        "reviewed": reviewed,
        "agreed": agreed,
        "disagreed": disagreed,
        "skipped": skipped,
        "failure_rate": summary.get("failure_rate", 0),
        "optimal_clusters": summary.get("optimal_clusters", 0),
        "silhouette_score": summary.get("silhouette_score", 0),
        "clusters": [
            {"id": c["cluster_id"], "label": c["label"], "size": c["size"], "percentage": c["percentage"]}
            for c in analysis.get("clusters", [])
        ],
    }


@app.get("/api/failure/{index}")
def get_failure(index: int):
    if index < 0 or index >= len(failures):
        raise HTTPException(status_code=404, detail="Index out of range")

    f = failures[index]
    img_idx = f["image_idx"]
    arr = test_images.get(img_idx)
    img_b64 = tensor_to_png_b64(arr) if arr is not None else ""

    existing = feedback.get(str(img_idx), {})

    # Support both MNIST (true_label) and CIFAR (true_class) key names
    true_label = f.get("true_class") or f.get("true_label", "?")
    pred_label  = f.get("predicted_class") or f.get("predicted_label", "?")

    return {
        "index": index,
        "total": len(failures),
        "image_idx": img_idx,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence_predicted": f["confidence_predicted"],
        "confidence_true": f["confidence_true"],
        "top3": f.get("top3", []),
        "hypothesis": f.get("hypothesis", ""),
        "cluster": f.get("cluster", -1),
        "cluster_label": cluster_labels.get(f.get("cluster", -1), "unknown"),
        "cluster_data_issue":   f.get("cluster_data_issue", -1),
        "cluster_visual_cause": f.get("cluster_visual_cause", -1),
        "cluster_fix":          f.get("cluster_fix", -1),
        "image_b64": img_b64,
        "existing_feedback": existing,
    }


@app.get("/api/find_by_image_idx/{image_idx}")
def find_by_image_idx(image_idx: int):
    for i, f in enumerate(failures):
        if f["image_idx"] == image_idx:
            return {"index": i}
    return {"index": -1}


@app.get("/api/dimensional_clusters")
def get_dimensional_clusters():
    return dim_clusters or {}


@app.get("/api/next_unreviewed/{after_index}")
def next_unreviewed(after_index: int):
    reviewed_idxs = {int(k) for k in feedback}
    for i in range(after_index, len(failures)):
        if failures[i]["image_idx"] not in reviewed_idxs:
            return {"index": i}
    # wrap around
    for i in range(0, after_index):
        if failures[i]["image_idx"] not in reviewed_idxs:
            return {"index": i}
    return {"index": -1}  # all reviewed


class FeedbackPayload(BaseModel):
    image_idx: int
    verdict: str          # "agree" | "disagree" | "skip"
    human_hypothesis: str  # free-text, can be empty


@app.post("/api/feedback")
def submit_feedback(payload: FeedbackPayload):
    feedback[str(payload.image_idx)] = {
        "image_idx": payload.image_idx,
        "verdict": payload.verdict,
        "human_hypothesis": payload.human_hypothesis,
    }
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(feedback, f, indent=2)
    return {"ok": True, "total_reviewed": len(feedback)}


@app.get("/api/divergence")
def get_divergence():
    """Return failures where human said 'disagree' — useful for prompt iteration."""
    results = []
    idx_map = {f["image_idx"]: f for f in failures}
    for key, fb in feedback.items():
        if fb["verdict"] == "disagree":
            f = idx_map.get(int(key), {})
            results.append({
                "image_idx": int(key),
                "true_label": f.get("true_label"),
                "predicted_label": f.get("predicted_label"),
                "llm_hypothesis": f.get("hypothesis", ""),
                "human_hypothesis": fb.get("human_hypothesis", ""),
                "cluster": f.get("cluster"),
                "cluster_label": cluster_labels.get(f.get("cluster", -1), ""),
            })
    return results


# ─── Frontend ─────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lens Review UI</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }

  header { background: #1a1a1a; border-bottom: 1px solid #333; padding: 12px 24px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 15px; color: #7fb3f5; letter-spacing: 0.05em; }
  .pill { background: #2a2a2a; border: 1px solid #444; border-radius: 20px; padding: 3px 10px; font-size: 12px; color: #aaa; }
  .pill span { color: #e0e0e0; font-weight: 600; }

  .layout { display: grid; grid-template-columns: 260px 1fr 280px; gap: 0; height: calc(100vh - 49px); }

  /* Left panel — cluster overview */
  .sidebar { background: #141414; border-right: 1px solid #222; padding: 16px; overflow-y: auto; }
  .sidebar h2 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
  .dim-section { margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #1e1e1e; }
  .dim-header { font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; margin-bottom: 8px; padding: 3px 0; }
  .cluster-card { background: #1e1e1e; border: 1px solid #2a2a2a; border-radius: 6px; padding: 10px 12px; margin-bottom: 6px; cursor: pointer; transition: border-color 0.15s; }
  .cluster-card:hover { border-color: #555; }
  .cluster-card.active { border-color: #7fb3f5; }
  .cluster-name { font-size: 12px; color: #ccc; margin-bottom: 4px; }
  .cluster-bar-wrap { background: #111; border-radius: 3px; height: 4px; margin-top: 6px; }
  .cluster-bar { border-radius: 3px; height: 4px; }
  .cluster-meta { font-size: 11px; color: #666; margin-top: 4px; }

  .progress-section { margin-top: 16px; padding-top: 16px; border-top: 1px solid #222; }
  .progress-row { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  .dot-agree { background: #4caf50; }
  .dot-disagree { background: #f44336; }
  .dot-skip { background: #888; }

  /* Main panel */
  .main { padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 20px; }

  .nav-row { display: flex; align-items: center; gap: 10px; }
  .nav-row button { background: #1e1e1e; border: 1px solid #333; color: #ccc; padding: 6px 14px; border-radius: 5px; cursor: pointer; font-size: 13px; font-family: inherit; }
  .nav-row button:hover { border-color: #7fb3f5; color: #7fb3f5; }
  .nav-row input { background: #1e1e1e; border: 1px solid #333; color: #e0e0e0; padding: 5px 10px; border-radius: 5px; width: 70px; text-align: center; font-size: 13px; font-family: inherit; }
  .counter { color: #666; font-size: 13px; }

  .card { background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 20px; }
  .card-title { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; }

  .digit-row { display: flex; gap: 24px; align-items: flex-start; }
  .digit-img-wrap { flex-shrink: 0; }
  .digit-img-wrap img { width: 168px; height: 168px; image-rendering: pixelated; border: 2px solid #333; border-radius: 4px; display: block; }
  .digit-labels { display: flex; gap: 10px; margin-top: 8px; justify-content: center; }
  .label-true { background: #1a3a1a; border: 1px solid #2d6a2d; color: #7ec87e; padding: 3px 10px; border-radius: 4px; font-size: 12px; }
  .label-pred { background: #3a1a1a; border: 1px solid #6a2d2d; color: #c87e7e; padding: 3px 10px; border-radius: 4px; font-size: 12px; }



  .conf-row { display: flex; gap: 8px; margin-bottom: 12px; }
  .conf-box { flex: 1; background: #111; border-radius: 5px; padding: 10px; text-align: center; }
  .conf-box .conf-label { font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.08em; }
  .conf-box .conf-val { font-size: 22px; font-weight: 700; margin-top: 4px; }
  .conf-box.wrong .conf-val { color: #f44336; }
  .conf-box.right .conf-val { color: #4caf50; }

  .hypothesis-box { background: #111; border-left: 3px solid #7fb3f5; padding: 14px 16px; border-radius: 0 5px 5px 0; font-size: 13px; line-height: 1.7; color: #ccc; white-space: pre-wrap; }
  .hyp-section { margin-bottom: 12px; }
  .hyp-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-bottom: 4px; }
  .hyp-label.data-issue { color: #f9c74f; }
  .hyp-label.visual-cause { color: #7fb3f5; }
  .hyp-label.fix { color: #4caf50; }
  .hyp-body { color: #ccc; font-size: 13px; line-height: 1.6; }
  .top3-row { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  .top3-chip { background: #1e1e1e; border: 1px solid #333; border-radius: 4px; padding: 3px 8px; font-size: 11px; color: #aaa; }
  .top3-chip.top1 { border-color: #f44336; color: #f44336; }
  .cluster-tag { display: inline-block; background: #1e2a3a; border: 1px solid #2d4a6a; color: #7fb3f5; padding: 3px 10px; border-radius: 12px; font-size: 11px; margin-top: 10px; }
  .dim-tag { display: flex; align-items: flex-start; gap: 8px; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 8px 12px; font-size: 12px; flex: 1; min-width: 160px; }
  .dim-tag-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; white-space: nowrap; margin-bottom: 2px; }
  .dim-tag-value { color: #ccc; font-size: 12px; line-height: 1.4; }

  /* Right panel — feedback */
  .feedback-panel { background: #141414; border-left: 1px solid #222; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
  .feedback-panel h2 { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.1em; }

  .verdict-btns { display: flex; flex-direction: column; gap: 8px; }
  .verdict-btn { background: #1e1e1e; border: 2px solid #2a2a2a; color: #aaa; padding: 10px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; font-family: inherit; text-align: left; transition: all 0.15s; }
  .verdict-btn:hover { border-color: #555; color: #e0e0e0; }
  .verdict-btn.selected-agree { border-color: #4caf50; color: #4caf50; background: #1a2a1a; }
  .verdict-btn.selected-disagree { border-color: #f44336; color: #f44336; background: #2a1a1a; }
  .verdict-btn.selected-skip { border-color: #888; color: #aaa; background: #1e1e1e; }

  textarea { background: #111; border: 1px solid #2a2a2a; color: #e0e0e0; border-radius: 5px; padding: 10px; font-size: 13px; font-family: inherit; resize: vertical; width: 100%; min-height: 100px; line-height: 1.5; }
  textarea:focus { outline: none; border-color: #7fb3f5; }

  .submit-btn { background: #7fb3f5; color: #000; border: none; padding: 10px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 700; font-family: inherit; width: 100%; transition: background 0.15s; }
  .submit-btn:hover { background: #a0c4ff; }
  .submit-btn:disabled { background: #333; color: #666; cursor: not-allowed; }

  .saved-badge { background: #1a3a1a; border: 1px solid #2d6a2d; color: #7ec87e; padding: 6px 12px; border-radius: 5px; font-size: 12px; text-align: center; display: none; }

  .divergence-link { color: #7fb3f5; font-size: 12px; text-decoration: none; display: block; text-align: center; padding: 8px; border: 1px solid #2a3a5a; border-radius: 5px; }
  .divergence-link:hover { background: #1e2a3a; }

  .kbd { background: #222; border: 1px solid #444; border-radius: 3px; padding: 1px 5px; font-size: 11px; color: #aaa; }
</style>
</head>
<body>

<header>
  <h1>LENS REVIEW UI</h1>
  <div class="pill">failures: <span id="hdr-total">—</span></div>
  <div class="pill">reviewed: <span id="hdr-reviewed">—</span></div>
  <div class="pill">failure rate: <span id="hdr-rate">—</span></div>
  <span style="flex:1"></span>
  <a href="/dashboard" target="_blank" style="color:#7fb3f5;font-size:12px;text-decoration:none;border:1px solid #2a3a5a;padding:4px 12px;border-radius:4px;">Dashboard ↗</a>
</header>

<div class="layout">

  <!-- Sidebar -->
  <div class="sidebar">
    <div id="dim-cluster-panel">
      <!-- populated by JS if dimensional_clusters.json exists -->
    </div>
    <div id="cluster-list-section" class="dim-section">
      <h2>Overall clusters</h2>
      <div id="cluster-list"></div>
    </div>
    <div class="progress-section">
      <h2 style="margin-bottom:10px">Your progress</h2>
      <div class="progress-row"><span><span class="dot dot-agree"></span>Agree</span><span id="cnt-agree">0</span></div>
      <div class="progress-row"><span><span class="dot dot-disagree"></span>Disagree</span><span id="cnt-disagree">0</span></div>
      <div class="progress-row"><span><span class="dot dot-skip"></span>Skip</span><span id="cnt-skip">0</span></div>
    </div>
  </div>

  <!-- Main -->
  <div class="main">
    <div class="nav-row">
      <button onclick="navigate(-1)">&#8592; Prev</button>
      <input id="idx-input" type="number" min="0" value="0" onchange="loadByInput()">
      <button onclick="navigate(1)">Next &#8594;</button>
      <button onclick="nextUnreviewed()">Next unreviewed</button>
      <span class="counter" id="counter">— / —</span>
      <span style="flex:1"></span>
      <span style="font-size:11px;color:#555"><span class="kbd">←</span><span class="kbd">→</span> navigate &nbsp; <span class="kbd">a</span> agree &nbsp; <span class="kbd">d</span> disagree &nbsp; <span class="kbd">s</span> skip</span>
    </div>

    <!-- Digit + stats -->
    <div class="card">
      <div class="card-title">Failure detail</div>
      <div class="digit-row">
        <div class="digit-img-wrap">
          <img id="digit-img" src="" alt="digit">
          <div class="digit-labels">
            <span class="label-true" id="lbl-true">True: —</span>
            <span class="label-pred" id="lbl-pred">Pred: —</span>
          </div>
        </div>
        <div style="flex:1">
          <div class="conf-row">
            <div class="conf-box wrong">
              <div class="conf-label">Conf (predicted)</div>
              <div class="conf-val" id="conf-pred">—</div>
            </div>
            <div class="conf-box right">
              <div class="conf-label">Conf (true)</div>
              <div class="conf-val" id="conf-true">—</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Dimension cluster tags -->
    <div id="dim-tags-row" style="display:none; gap:8px; flex-wrap:wrap;"></div>

    <!-- LLM hypothesis -->
    <div class="card">
      <div class="card-title">LLM Hypothesis</div>
      <div class="hypothesis-box" id="hyp-text"></div>
      <div class="top3-row" id="top3-chips" style="display:none"></div>
      <div class="cluster-tag" id="cluster-tag">cluster —</div>
    </div>
  </div>

  <!-- Feedback panel -->
  <div class="feedback-panel">
    <h2>Your verdict</h2>
    <div class="verdict-btns">
      <button class="verdict-btn" id="btn-agree" onclick="selectVerdict('agree')">✓ Agree — LLM got it right</button>
      <button class="verdict-btn" id="btn-disagree" onclick="selectVerdict('disagree')">✗ Disagree — LLM missed it</button>
      <button class="verdict-btn" id="btn-skip" onclick="selectVerdict('skip')">→ Skip / unsure</button>
    </div>

    <div>
      <h2 style="margin-bottom:8px">Your explanation <span style="color:#444;font-size:10px">(optional)</span></h2>
      <textarea id="human-hyp" placeholder="What do you think actually caused this error?"></textarea>
    </div>

    <button class="submit-btn" id="submit-btn" onclick="submitFeedback()">Save &amp; next</button>
    <div class="saved-badge" id="saved-badge">✓ Saved</div>

    <a class="divergence-link" href="/divergence" target="_blank">View disagreements ↗</a>
  </div>

</div>

<script>
let currentIndex = 0;
let currentImageIdx = -1;
let selectedVerdict = null;
let totalFailures = 0;

let dimData = null;

async function loadDimensionalClusters() {
  const r = await fetch('/api/dimensional_clusters');
  const d = await r.json();
  if (!d.dimensions) return;
  dimData = d;

  const panel = document.getElementById('dim-cluster-panel');
  panel.innerHTML = '';

  const DIM_KEYS = [
    { key: 'data_issue',   clusterField: 'cluster_data_issue' },
    { key: 'visual_cause', clusterField: 'cluster_visual_cause' },
    { key: 'fix',          clusterField: 'cluster_fix' },
  ];

  DIM_KEYS.forEach(({ key, clusterField }) => {
    const dim = d.dimensions[key];
    if (!dim) return;
    const section = document.createElement('div');
    section.className = 'dim-section';
    section.innerHTML = `<div class="dim-header" style="color:${dim.color}">${dim.label}</div>`;
    dim.clusters.forEach(c => {
      const card = document.createElement('div');
      card.className = 'cluster-card';
      card.id = `dim-${key}-${c.cluster_id}`;
      card.innerHTML = `
        <div class="cluster-name">${c.label}</div>
        <div class="cluster-bar-wrap"><div class="cluster-bar" style="width:${c.percentage}%;background:${dim.color}"></div></div>
        <div class="cluster-meta">${c.size} failures · ${c.percentage}%</div>`;
      section.appendChild(card);
    });
    panel.appendChild(section);
  });
}

async function loadStats() {
  const r = await fetch('/api/stats');
  const s = await r.json();
  totalFailures = s.total_failures;
  document.getElementById('hdr-total').textContent = s.total_failures.toLocaleString();
  document.getElementById('hdr-reviewed').textContent = s.reviewed;
  document.getElementById('hdr-rate').textContent = (s.failure_rate * 100).toFixed(2) + '%';
  document.getElementById('cnt-agree').textContent = s.agreed;
  document.getElementById('cnt-disagree').textContent = s.disagreed;
  document.getElementById('cnt-skip').textContent = s.skipped;

  const list = document.getElementById('cluster-list');
  list.innerHTML = '';
  s.clusters.forEach(c => {
    const el = document.createElement('div');
    el.className = 'cluster-card';
    el.id = 'cluster-' + c.id;
    el.innerHTML = `
      <div class="cluster-name">${c.label}</div>
      <div class="cluster-bar-wrap"><div class="cluster-bar" style="width:${c.percentage}%"></div></div>
      <div class="cluster-meta">${c.size} samples · ${c.percentage}%</div>`;
    list.appendChild(el);
  });
}

async function loadFailure(index) {
  if (index < 0 || index >= totalFailures) return;
  currentIndex = index;
  document.getElementById('idx-input').value = index;
  document.getElementById('counter').textContent = `${index + 1} / ${totalFailures}`;

  const r = await fetch(`/api/failure/${index}`);
  if (!r.ok) return;
  const f = await r.json();
  currentImageIdx = f.image_idx;

  // Image
  document.getElementById('digit-img').src = f.image_b64 ? `data:image/png;base64,${f.image_b64}` : '';
  document.getElementById('lbl-true').textContent = `True: ${f.true_label}`;
  document.getElementById('lbl-pred').textContent = `Pred: ${f.predicted_label}`;

  // Confidence
  document.getElementById('conf-pred').textContent = (f.confidence_predicted * 100).toFixed(1) + '%';
  document.getElementById('conf-true').textContent = (f.confidence_true * 100).toFixed(1) + '%';

  // Hypothesis — parse structured 3-part format
  renderHypothesis(f.hypothesis || '');
  document.getElementById('cluster-tag').textContent = `cluster ${f.cluster} · ${f.cluster_label}`;

  // Top-3 predictions
  const top3El = document.getElementById('top3-chips');
  if (f.top3 && f.top3.length) {
    top3El.innerHTML = f.top3.map((t, i) =>
      `<span class="top3-chip ${i===0?'top1':''}">${t.class} ${(t.confidence*100).toFixed(1)}%</span>`
    ).join('');
    top3El.style.display = 'flex';
  } else {
    top3El.style.display = 'none';
  }

  // Dimension cluster tags row
  const dimTagsRow = document.getElementById('dim-tags-row');
  if (dimData && dimData.dimensions) {
    const DIM_MAP = [
      { key: 'data_issue',   field: 'cluster_data_issue' },
      { key: 'visual_cause', field: 'cluster_visual_cause' },
      { key: 'fix',          field: 'cluster_fix' },
    ];
    dimTagsRow.innerHTML = '';
    let anyFound = false;
    DIM_MAP.forEach(({ key, field }) => {
      const dim = dimData.dimensions[key];
      if (!dim) return;
      const cid = f[field];
      if (cid === undefined || cid < 0) return;
      const cluster = dim.clusters.find(c => c.cluster_id === cid);
      if (!cluster) return;
      anyFound = true;
      const tag = document.createElement('div');
      tag.className = 'dim-tag';
      tag.innerHTML = `
        <div>
          <div class="dim-tag-label" style="color:${dim.color}">${dim.label}</div>
          <div class="dim-tag-value">${cluster.label}</div>
        </div>`;
      dimTagsRow.appendChild(tag);
    });
    dimTagsRow.style.display = anyFound ? 'flex' : 'none';
  } else {
    dimTagsRow.style.display = 'none';
  }

  // Highlight active cluster cards (overall + dimensional)
  document.querySelectorAll('.cluster-card').forEach(el => el.classList.remove('active'));
  const clCard = document.getElementById('cluster-' + f.cluster);
  if (clCard) clCard.classList.add('active');
  [['data_issue', f.cluster_data_issue], ['visual_cause', f.cluster_visual_cause], ['fix', f.cluster_fix]]
    .forEach(([key, cid]) => {
      if (cid === undefined || cid < 0) return;
      const el = document.getElementById(`dim-${key}-${cid}`);
      if (el) el.classList.add('active');
    });

  // Restore existing feedback if any
  const ex = f.existing_feedback;
  if (ex && ex.verdict) {
    selectVerdict(ex.verdict, false);
    document.getElementById('human-hyp').value = ex.human_hypothesis || '';
  } else {
    clearVerdict();
    document.getElementById('human-hyp').value = '';
  }

  document.getElementById('saved-badge').style.display = 'none';
}

function selectVerdict(v, clearText = true) {
  selectedVerdict = v;
  ['agree', 'disagree', 'skip'].forEach(name => {
    const btn = document.getElementById('btn-' + name);
    btn.className = 'verdict-btn' + (name === v ? ' selected-' + name : '');
  });
  if (clearText && v !== selectedVerdict) {
    // keep text when restoring existing
  }
}

function renderHypothesis(text) {
  const el = document.getElementById('hyp-text');
  if (!text) { el.innerHTML = '<span style="color:#555">(no hypothesis generated yet)</span>'; return; }

  // Try to parse structured 3-part format
  const sections = [
    { key: '1. DATA ISSUE', label: 'Data Issue?', cls: 'data-issue' },
    { key: '2. VISUAL CAUSE', label: 'Visual Cause', cls: 'visual-cause' },
    { key: '3. FIX', label: 'Fix', cls: 'fix' },
  ];
  let html = '';
  let remaining = text;
  let matched = false;
  for (let i = 0; i < sections.length; i++) {
    const s = sections[i];
    const next = sections[i + 1];
    const start = remaining.search(new RegExp(s.key, 'i'));
    if (start === -1) continue;
    matched = true;
    const end = next ? remaining.search(new RegExp(next.key, 'i')) : remaining.length;
    const body = remaining.slice(start + s.key.length, end === -1 ? remaining.length : end)
      .replace(/^[\s:?]+/, '').trim();
    html += `<div class="hyp-section"><div class="hyp-label ${s.cls}">${s.label}</div><div class="hyp-body">${body}</div></div>`;
  }
  el.innerHTML = matched ? html : `<div class="hyp-body">${text}</div>`;
}

function clearVerdict() {
  selectedVerdict = null;
  ['agree', 'disagree', 'skip'].forEach(name => {
    document.getElementById('btn-' + name).className = 'verdict-btn';
  });
}

async function submitFeedback() {
  if (!selectedVerdict) { alert('Please select a verdict first.'); return; }
  const humanHyp = document.getElementById('human-hyp').value.trim();

  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_idx: currentImageIdx, verdict: selectedVerdict, human_hypothesis: humanHyp }),
  });

  document.getElementById('saved-badge').style.display = 'block';
  await loadStats();
  await navigate(1);
}

function navigate(delta) {
  loadFailure(currentIndex + delta);
}

function loadByInput() {
  const v = parseInt(document.getElementById('idx-input').value);
  if (!isNaN(v)) loadFailure(Math.max(0, Math.min(v, totalFailures - 1)));
}

async function nextUnreviewed() {
  const r = await fetch(`/api/next_unreviewed/${currentIndex}`);
  const d = await r.json();
  if (d.index === -1) alert('All failures reviewed!');
  else loadFailure(d.index);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'a') selectVerdict('agree');
  if (e.key === 'd') selectVerdict('disagree');
  if (e.key === 's') selectVerdict('skip');
  if (e.key === 'Enter') submitFeedback();
});

// Init — support ?image_idx=N from dashboard links
(async () => {
  await loadDimensionalClusters();
  await loadStats();
  const params = new URLSearchParams(window.location.search);
  const imageIdx = params.get('image_idx');
  if (imageIdx) {
    const r = await fetch(`/api/find_by_image_idx/${imageIdx}`);
    const d = await r.json();
    await loadFailure(d.index >= 0 ? d.index : 0);
  } else {
    await loadFailure(0);
  }
})();
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/divergence", response_class=HTMLResponse)
def divergence_page():
    divs = get_divergence()
    rows = ""
    for d in divs:
        rows += f"""
        <tr>
          <td>{d['image_idx']}</td>
          <td>{d['true_label']} → {d['predicted_label']}</td>
          <td style="color:#7fb3f5">{d['cluster_label']}</td>
          <td style="color:#aaa;font-size:12px">{d['llm_hypothesis'][:120]}…</td>
          <td style="color:#f9c74f;font-size:12px">{d['human_hypothesis'] or '—'}</td>
        </tr>"""
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Lens — Disagreements</title>
<style>
body {{ font-family: monospace; background: #0f0f0f; color: #e0e0e0; padding: 32px; }}
h1 {{ color: #7fb3f5; margin-bottom: 20px; font-size: 18px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; color: #666; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; padding: 8px 12px; border-bottom: 1px solid #222; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }}
tr:hover td {{ background: #1a1a1a; }}
a {{ color: #7fb3f5; }}
</style></head>
<body>
<h1>Disagreements ({len(divs)} cases where you said the LLM was wrong)</h1>
<p style="color:#666;margin-bottom:20px;font-size:13px">Use these to iterate on the hypothesis prompt in train_and_analyze.py.</p>
<table>
<tr><th>idx</th><th>confusion</th><th>cluster</th><th>LLM hypothesis</th><th>Your explanation</th></tr>
{rows if rows else '<tr><td colspan="5" style="color:#666;padding:20px">No disagreements yet.</td></tr>'}
</table>
<br><a href="/">← Back to review</a>
</body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    # Always read fresh from disk so restarts aren't needed after re-clustering
    dc = dim_clusters
    dim_path = ANALYSIS_PATH.replace("analysis.json", "dimensional_clusters.json")
    if Path(dim_path).exists():
        with open(dim_path) as _f:
            dc = json.load(_f)
    if not dc or "dimensions" not in dc:
        return HTMLResponse("<p style='color:#e0e0e0;padding:32px'>No dimensional_clusters.json found. Run cluster_dimensions.py first.</p>")

    DIM_COLORS = {"data_issue": "#f9c74f", "visual_cause": "#7fb3f5", "fix": "#4caf50"}

    def render_dimension(key, dim):
        color = dim.get("color", DIM_COLORS.get(key, "#aaa"))
        sil   = dim.get("silhouette", 0)
        cards = ""
        for c in sorted(dim["clusters"], key=lambda x: -x["size"]):
            pct = c["percentage"]
            bar_w = min(100, pct * 2)
            examples_html = ""
            for ex in c.get("representative_examples", []):
                tc = ex.get("true_class", "?")
                pc = ex.get("predicted_class", "?")
                snippet = ex.get("section_text", "")[:120].replace("<", "&lt;")
                examples_html += f'<div class="ex-row"><span class="confusion-pill">{tc}→{pc}</span><span class="ex-text">{snippet}…</span></div>'
            cards += f"""
            <div class="dim-card">
              <div class="dim-card-header">
                <span class="dim-card-label">{c['label']}</span>
                <span class="dim-card-size">{c['size']} failures &middot; {pct}%</span>
              </div>
              <div class="bar-wrap"><div class="bar-fill" style="width:{bar_w}%;background:{color}"></div></div>
              <div class="examples">{examples_html}</div>
            </div>"""
        return f"""
        <div class="dim-section">
          <div class="dim-heading" style="color:{color}">{dim['label']}
            <span class="sil-badge">silhouette {sil:.3f}</span>
          </div>
          <div class="dim-cards">{cards}</div>
        </div>"""

    dims_html = ""
    for key, dim in dc["dimensions"].items():
        dims_html += render_dimension(key, dim)

    # Data issues spotlight
    data_issues = dc.get("data_issues", {})
    spotlight_html = ""
    if data_issues.get("total_actionable", 0) > 0:
        tag_counts = data_issues.get("tag_counts", {})
        rows = ""
        for f in data_issues.get("failures", [])[:20]:
            tc  = f.get("true_class", "?")
            pc  = f.get("predicted_class", "?")
            tag = f.get("tag", "")
            v   = f.get("verdict", "").replace("<", "&lt;")
            tag_color = "#f9c74f" if tag == "annotation_issue" else "#ff9800"
            rows += f'<tr><td>{f["image_idx"]}</td><td>{tc} → {pc}</td><td style="color:{tag_color}">{tag}</td><td style="color:#aaa">{v}</td><td><a href="/?image_idx={f["image_idx"]}" style="color:#7fb3f5">review →</a></td></tr>'
        tag_pills = " ".join(f'<span style="background:#2a2a2a;border:1px solid #444;border-radius:12px;padding:3px 10px;font-size:12px;color:#f9c74f">{t}: {c}</span>' for t, c in tag_counts.items())
        spotlight_html = f"""
        <div class="spotlight">
          <div class="spotlight-header">Data Issues Worth Acting On
            <span style="color:#aaa;font-size:13px;font-weight:400">&nbsp;{data_issues['total_actionable']} failures</span>
          </div>
          <div style="margin-bottom:16px">{tag_pills}</div>
          <table>
            <tr><th>idx</th><th>confusion</th><th>tag</th><th>verdict</th><th></th></tr>
            {rows}
          </table>
        </div>"""

    def render_spotlight(dim, key, color, border_color):
        """Render a top-clusters summary table for any dimension."""
        clusters = sorted(dim.get("clusters", []), key=lambda x: -x["size"])
        rows = ""
        for c in clusters:
            exs = c.get("representative_examples", [])
            confusion = ", ".join(
                f"{e.get('true_class','?')}→{e.get('predicted_class','?')}" for e in exs[:2]
            )
            snippet = exs[0].get("section_text", "")[:100].replace("<", "&lt;") if exs else ""
            rows += (
                f'<tr>'
                f'<td style="color:{color};font-weight:600">{c["label"]}</td>'
                f'<td>{c["size"]} <span style="color:#555">({c["percentage"]}%)</span></td>'
                f'<td style="color:#888">{confusion}</td>'
                f'<td style="color:#555">{snippet}…</td>'
                f'</tr>'
            )
        return f"""
        <div class="spotlight" style="border-color:{border_color}">
          <div class="spotlight-header" style="color:{color}">{dim['label']} — Top Clusters</div>
          <table>
            <tr><th>cluster</th><th>failures</th><th>examples</th><th>detail</th></tr>
            {rows}
          </table>
        </div>"""

    visual_spotlight = render_spotlight(
        dc["dimensions"].get("visual_cause", {}), "visual_cause", "#7fb3f5", "#1a2a3a"
    ) if "visual_cause" in dc["dimensions"] else ""

    fix_spotlight = render_spotlight(
        dc["dimensions"].get("fix", {}), "fix", "#4caf50", "#1a3a1a"
    ) if "fix" in dc["dimensions"] else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Lens — Dashboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'SF Mono', 'Fira Code', monospace; background: #0f0f0f; color: #e0e0e0; padding: 32px; }}
h1 {{ color: #7fb3f5; font-size: 18px; margin-bottom: 24px; }}
.dim-section {{ margin-bottom: 40px; }}
.dim-heading {{ font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 14px; display: flex; align-items: center; gap: 12px; }}
.sil-badge {{ background: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 2px 8px; font-size: 11px; color: #666; font-weight: 400; }}
.dim-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }}
.dim-card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px; padding: 14px; }}
.dim-card-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
.dim-card-label {{ font-size: 13px; color: #e0e0e0; font-weight: 600; }}
.dim-card-size {{ font-size: 11px; color: #666; }}
.bar-wrap {{ background: #111; border-radius: 3px; height: 3px; margin-bottom: 10px; }}
.bar-fill {{ height: 3px; border-radius: 3px; }}
.examples {{ display: flex; flex-direction: column; gap: 6px; }}
.ex-row {{ display: flex; gap: 8px; align-items: flex-start; font-size: 11px; }}
.confusion-pill {{ background: #2a1a1a; border: 1px solid #4a2a2a; color: #c87e7e; padding: 1px 6px; border-radius: 4px; white-space: nowrap; flex-shrink: 0; }}
.ex-text {{ color: #666; line-height: 1.4; }}
.spotlight {{ background: #1a1a1a; border: 1px solid #3a2a1a; border-radius: 8px; padding: 20px; margin-bottom: 24px; }}
.spotlight-header {{ font-size: 14px; font-weight: 700; margin-bottom: 14px; }}
.spotlights {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 40px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 12px; }}
th {{ text-align: left; color: #555; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; padding: 6px 10px; border-bottom: 1px solid #222; }}
td {{ padding: 7px 10px; border-bottom: 1px solid #1a1a1a; vertical-align: top; color: #aaa; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
tr:hover td {{ background: #1e1e1e; }}
a {{ color: #7fb3f5; text-decoration: none; }}
.nav-links {{ margin-bottom: 28px; display: flex; gap: 16px; }}
.nav-links a {{ color: #7fb3f5; font-size: 13px; }}
.section-divider {{ border: none; border-top: 1px solid #222; margin: 32px 0; }}
</style></head>
<body>
<h1>Lens — Cluster Dashboard</h1>
<div class="nav-links">
  <a href="/">← Review UI</a>
  <a href="/divergence">Disagreements</a>
</div>

{spotlight_html}

<div class="spotlights">
  {visual_spotlight}
  {fix_spotlight}
</div>

<hr class="section-divider">
<div style="color:#555;font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:20px">Full cluster breakdown</div>
{dims_html}
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("review_ui:app", host="0.0.0.0", port=8000, reload=False)
