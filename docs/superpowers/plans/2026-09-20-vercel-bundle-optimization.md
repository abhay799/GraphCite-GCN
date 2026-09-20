# GraphCite GCN Vercel Bundle Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Vercel deploy the existing GraphCite GCN API as an ONNX-and-NumPy-only inference function while preserving all routes and Cora prediction results.

**Architecture:** Export the already-processed Cora feature matrix and edge index to a committed compressed NumPy archive using a development-only script. Production `main.py` lazily loads that archive instead of importing PyTorch Geometric. Root `requirements.txt` becomes the Vercel runtime dependency declaration; `requirements-dev.txt` retains all local/demo/training/test dependencies.

**Tech Stack:** FastAPI, ONNX Runtime, NumPy, Python 3.13 on Vercel, pytest; PyTorch and PyTorch Geometric only for local graph export/development.

**Spec:** `docs/superpowers/specs/2026-09-20-vercel-inference-only-design.md`

## Global Constraints

- Root `requirements.txt` contains only FastAPI, Uvicorn, ONNX Runtime, NumPy, and Pydantic.
- `requirements-dev.txt` retains PyTorch, PyTorch Geometric, Streamlit, Pandas, Requests, pytest, and existing training/visualization packages.
- Production `main.py` imports no Torch, PyTorch Geometric, Streamlit, or Pandas.
- Do not retrain or modify ONNX files or Cora source data.
- Preserve all current API routes, contracts, GraphCite GCN branding, and numerical Cora predictions.
- Commit `runtime/cora_graph.npz`; do not deploy `data/`, tests, notebooks, caches, virtual environments, or the graph-export script.
- Pin Vercel’s supported Python range to 3.13 in `pyproject.toml`.

## Review Focus

- The exported feature matrix and edge index exactly match the arrays previously supplied by `Planetoid`.
- Node 0’s predicted class remains the current production result after loading from NumPy.
- Vercel only receives required ONNX, NumPy graph, and static assets.
- Local development remains installable with `pip install -r requirements-dev.txt`.
- Root requirements do not reintroduce heavyweight packages transitively by direct declaration.

---

### Task 1: Capture the current Cora prediction and export graph arrays

**Files:**
- Create: `scripts/export_cora_graph.py`
- Create: `runtime/cora_graph.npz`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `torch_geometric.datasets.Planetoid` against `data/Planetoid` during development only.
- Produces: `runtime/cora_graph.npz` with `node_features: float32[2708,1433]` and `edge_indices: int64[2,10556]`.

- [ ] Add tests that assert the archive exists, has the expected dtypes/shapes, and node 0’s class equals the pre-migration class.
- [ ] Run those tests and observe failure because the archive and baseline assertion are absent.
- [ ] Implement the development-only exporter, run it once, and commit its generated archive.
- [ ] Rerun focused tests and the full suite.

### Task 2: Replace runtime PyTorch Geometric loading with NumPy loading

**Files:**
- Modify: `main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `runtime/cora_graph.npz`.
- Produces: `_load_cora_graph()` data with the same feature and edge arrays used by `/predict/cora_node`.

- [ ] Add a test that production source has no prohibited imports and that `/predict/cora_node` preserves node 0’s predicted class.
- [ ] Observe failure under the current Planetoid loader.
- [ ] Implement a NumPy-only cached archive loader and retain bounds, neighbor, and response logic.
- [ ] Run focused tests and full suite.

### Task 3: Separate Vercel and local dependencies and reduce bundle inputs

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`
- Create: `pyproject.toml`
- Modify: `vercel.json`
- Modify: `.vercelignore`
- Modify: `README.md`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: Vercel’s root requirements detection and Python 3.13 support.
- Produces: a function bundle that installs only five runtime packages and includes only model, runtime graph, and static assets.

- [ ] Add tests for exact root requirements, Python 3.13 metadata, and Vercel include/exclude configuration.
- [ ] Observe failure against the current full root requirements and data-directory bundle.
- [ ] Implement the smallest requirements/configuration changes and document local/deployment installation paths.
- [ ] Run syntax checks, `api.index:app` import, full pytest suite, and measure included artifact sizes.
