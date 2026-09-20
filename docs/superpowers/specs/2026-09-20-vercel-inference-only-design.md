# GraphCite GCN Vercel Inference-Only Design

## Goal

Reduce the Vercel Python function from a multi-gigabyte local-development environment to an inference-only function while preserving GraphCite GCN’s API routes, ONNX model, Cora predictions, and local development workflow.

## Current State

`main.py` imports `torch_geometric.datasets.Planetoid` only when `/predict/cora_node` loads the Cora graph. The root `requirements.txt` includes PyTorch, PyTorch Geometric, Streamlit, Pandas, and Requests for local training/demo workflows. Vercel installs that full set, which produces the oversized bundle.

## Design

### Runtime graph asset

Create `runtime/cora_graph.npz` from the existing processed Cora dataset during development. The archive contains `node_features` as a float32 array with shape `(2708, 1433)` and `edge_indices` as an int64 array with shape `(2, 10556)`. Those are the exact arrays passed to ONNX Runtime today, so no model, class mapping, or request/response behavior changes.

The checked-in development exporter is `scripts/export_cora_graph.py`. It is the only code that imports PyTorch Geometric, and it is excluded from Vercel deployment. The production app reads `runtime/cora_graph.npz` lazily through NumPy using a module-relative `Path`.

### Production dependencies

Create `requirements-vercel.txt` containing only FastAPI, Uvicorn, ONNX Runtime, NumPy, and Pydantic. Keep `requirements.txt` as the full local-development dependency set, including Streamlit and PyTorch Geometric.

Vercel’s build configuration installs `requirements-vercel.txt`, and `vercel.json` explicitly includes `runtime/cora_graph.npz`, the two ONNX files, and `static/` in the Python function bundle.

### Inference behavior

`/predict` remains unchanged. `/predict/cora_node` loads the NumPy archive and passes the same Cora node features and edges into the existing `run_model` and neighbor-discovery logic. The existing bounds validation, response fields, OpenAPI metadata, static frontend, docs, and branding remain unchanged.

## Verification

- Test that the Vercel requirements file is precisely the five required production packages and excludes PyTorch, PyTorch Geometric, Streamlit, Pandas, and development visualization/training packages.
- Test that `main.py` has no production imports of Torch, PyTorch Geometric, Streamlit, or Pandas.
- Test that the runtime archive has the expected feature and edge shapes/dtypes.
- Add a node-0 regression assertion using the class produced by the current implementation before the migration.
- Run syntax checks, the local `api.index:app` import check, and the entire pytest suite.

## Constraints

- Do not retrain or modify either ONNX artifact.
- Do not modify Cora source data contents.
- Keep all existing routes: `/`, `/health`, `/info`, `/predict`, `/predict/cora_node`, `/docs`, and `/openapi.json`.
- Do not deploy `.venv`, caches, notebooks, tests, or the development exporter.
