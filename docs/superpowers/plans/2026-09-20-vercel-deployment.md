# GraphCite GCN Vercel Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the existing GraphCite GCN FastAPI app as a Vercel serverless function without changing model artifacts, Cora data, API contracts, or inference behavior.

**Architecture:** Make `main.py` resolve the model, companion ONNX data file, Cora dataset, and static directory from its own absolute `Path(__file__)` location. Add a small `api/index.py` entrypoint that imports and re-exports that existing `app`; it contains no route or inference code. Use a minimal rewrite in `vercel.json` only to route all public paths through that single FastAPI function, and keep deployment inputs lean with `.vercelignore`.

**Tech Stack:** Python 3, FastAPI, ONNX Runtime, PyTorch Geometric, Vercel Python Functions, pytest.

**Spec:** User request from 2026-09-20.

## Global Constraints

- Do not modify `simple_gcn_cora.onnx`, `simple_gcn_cora.onnx.data`, Cora dataset contents, class mappings, prediction logic, request/response contracts, or verified inference behavior.
- The Vercel entrypoint is `api/index.py` and exports `app` by reusing `main.app`.
- `/`, `/health`, `/info`, `/predict`, `/predict/cora_node`, `/docs`, and `/openapi.json` must remain available.
- Resolve all runtime paths from `Path(__file__)`, never the working directory.
- Package `simple_gcn_cora.onnx`, `simple_gcn_cora.onnx.data`, `data/`, and `static/` for runtime.
- Keep `requirements.txt` runtime-only; do not add pytest, matplotlib, seaborn, or scikit-learn unless runtime code imports them.
- Exclude `.venv/`, `__pycache__/`, `.pytest_cache/`, `.git/`, `tests/`, and the training notebook from deployment.
- Preserve GraphCite GCN branding.

## Review Focus

- Importing `api.index` from the project root exposes the exact existing FastAPI application rather than a second application instance.
- Imports work when the working directory is not the repository root, because model, data, and static paths are module-relative absolute paths.
- The Vercel catch-all rule does not intercept FastAPI’s `/docs` and `/openapi.json` routes incorrectly.
- Static frontend assets are served by the existing mounted `StaticFiles` app after the catch-all rewrite.
- Deployment configuration excludes local-only artifacts while retaining both ONNX files, `data/`, and `static/`.

---

### Task 1: Add a module-relative Vercel entrypoint and regression coverage

**Files:**
- Create: `api/__init__.py`
- Create: `api/index.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `main.app`, the pre-existing FastAPI instance.
- Produces: `api.index.app`, the same FastAPI instance Vercel imports.

- [ ] **Step 1: Write the failing test**

```python
def test_vercel_entrypoint_reexports_existing_app():
    from api.index import app as vercel_app
    from main import app as local_app

    assert vercel_app is local_app
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/test_api.py::test_vercel_entrypoint_reexports_existing_app -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'api'`.

- [ ] **Step 3: Add the minimal entrypoint implementation**

```python
# api/index.py
from main import app

__all__ = ["app"]
```

Create an empty `api/__init__.py` so the directory is an explicit Python package.

- [ ] **Step 4: Run focused and full test suites**

Run: `python -m pytest tests/test_api.py::test_vercel_entrypoint_reexports_existing_app -q` then `python -m pytest -q`

Expected: the entrypoint test and all pre-existing API contract tests pass.

### Task 2: Make existing runtime paths independent of the working directory

**Files:**
- Modify: `main.py:1-36`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: the existing files adjacent to `main.py`: `simple_gcn_cora.onnx`, `simple_gcn_cora.onnx.data`, `data/Planetoid`, and `static/`.
- Produces: absolute `Path` values used by ONNX Runtime, Planetoid, `FileResponse`, and `StaticFiles`.

- [ ] **Step 1: Write the failing test**

```python
def test_runtime_paths_are_absolute_module_relative():
    from main import BASE_DIR, DATA_DIR, MODEL_PATH, STATIC_DIR

    assert BASE_DIR.is_absolute()
    assert MODEL_PATH == BASE_DIR / "simple_gcn_cora.onnx"
    assert DATA_DIR == BASE_DIR / "data" / "Planetoid"
    assert STATIC_DIR == BASE_DIR / "static"
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/test_api.py::test_runtime_paths_are_absolute_module_relative -q`

Expected: FAIL because the current `str`/`os.path` values do not provide `Path` semantics.

- [ ] **Step 3: Replace path construction only**

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "simple_gcn_cora.onnx"
DATA_DIR = BASE_DIR / "data" / "Planetoid"
STATIC_DIR = BASE_DIR / "static"
```

Use `Path.exists()`, `Path.is_dir()`, and pass `str(...)` only at library boundaries that require a string. Do not change request validation, model execution, class mappings, or endpoint output.

- [ ] **Step 4: Run focused and full test suites**

Run: `python -m pytest tests/test_api.py::test_runtime_paths_are_absolute_module_relative -q` then `python -m pytest -q`

Expected: path test and all API contract tests pass.

### Task 3: Add minimal Vercel routing and deployment exclusions

**Files:**
- Create: `vercel.json`
- Create or modify: `.vercelignore`

**Interfaces:**
- Consumes: Vercel’s Python-function convention (`api/index.py`) and `main.app` static mount.
- Produces: one catch-all rewrite to `/api/index` and a deployment file filter that retains required runtime assets.

- [ ] **Step 1: Verify the absent configuration state**

Run: `Test-Path vercel.json; Test-Path .vercelignore`

Expected: configuration files are absent or do not yet include all requested exclusions.

- [ ] **Step 2: Add minimal configuration**

```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/api/index" }]
}
```

```text
.venv/
__pycache__/
.pytest_cache/
.git/
tests/
cora_citation_network_classification_(1).ipynb
```

Do not ignore `simple_gcn_cora.onnx`, `simple_gcn_cora.onnx.data`, `data/`, or `static/`.

- [ ] **Step 3: Validate configuration contents**

Run: `python -c "import json; print(json.load(open('vercel.json'))['rewrites'])"` and inspect `.vercelignore`.

Expected: valid JSON with a single catch-all rewrite and all six requested exclusions.

### Task 4: Verify deployment readiness and document deployment commands

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Vercel entrypoint and configuration from Tasks 1–3.
- Produces: exact deployment instructions and a concise runtime-dependency note.

- [ ] **Step 1: Add a Vercel deployment section**

Document `vercel login`, `vercel`, and `vercel --prod`, state that the project root is the directory containing `vercel.json`, and identify the ONNX Runtime plus PyTorch/PyTorch Geometric footprint as deployment-size and cold-start considerations.

- [ ] **Step 2: Run the requested verification commands**

Run: `python -m py_compile main.py api/index.py`; `python -m pytest -q`; `python -c "from api.index import app; assert app.title == 'GraphCite GCN'; print(app)"`.

Expected: syntax checks and tests pass; the import command prints the existing FastAPI app instance.

- [ ] **Step 3: Verify legacy behavior has not been edited**

Run: inspect the diff of `main.py` and verify changes are limited to path handling; inspect the diff of model/data files and confirm none exist.

Expected: no ONNX, dataset, class-mapping, or prediction-logic modification.
