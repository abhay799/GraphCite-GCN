# GraphCite GCN

Graph Neural Network for Citation Classification.

An interactive Graph Convolutional Network (GCN) node-classification demo for the **Cora citation network**. The trained model is exported to ONNX and served through FastAPI, with both a browser UI and a Streamlit UI.

> **Portfolio / attribution note:** this project copy was adapted from an educational/reference repository downloaded by the current maintainer. Before publishing it publicly or presenting it as portfolio work, add the original GitHub repository and YouTube/tutorial links here, retain all required upstream attribution, and verify the upstream license. No license file was included in the downloaded ZIP, so redistribution rights should not be assumed.

## What it demonstrates

- Graph neural network inference on the Cora citation graph
- 2,708 paper nodes, 10,556 directed edge entries, 1,433-dimensional node features
- 7 research-topic classes
- ONNX Runtime inference on CPU
- FastAPI REST endpoints
- Real Cora 1-hop neighborhood visualization
- Custom graph inference with shape and edge-bound validation
- Streamlit and static HTML/JavaScript interfaces

## Architecture

```text
Cora dataset
   ↓
PyTorch Geometric GCN training notebook
   ↓
ONNX export
   ↓
ONNX Runtime
   ↓
FastAPI
   ├── Static browser UI
   └── Streamlit UI
```

## Setup

From the project root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Start the API:

```powershell
uvicorn main:app --reload
```

Open:

- App: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Model info: `http://127.0.0.1:8000/info`

Run the optional Streamlit UI in a second terminal:

```powershell
streamlit run streamlit_app.py
```

The Streamlit app defaults to `http://127.0.0.1:8000`. Override it with the `CORA_API_BASE_URL` environment variable if needed.

## Deploy on Vercel

The Vercel entrypoint at `api/index.py` reuses the existing FastAPI application. `vercel.json` routes root-level requests to that application and explicitly packages the ONNX model, its companion data file, the pre-exported Cora graph, and the static frontend. Root `requirements.txt` is intentionally the lightweight Vercel runtime dependency set; use `requirements-dev.txt` for local development.

From this project directory:

```powershell
npm install --global vercel
vercel login
vercel
vercel --prod
```

Use a recent Vercel CLI (version 48.1.8 or later). The first `vercel` command creates a preview deployment; verify `/`, `/health`, `/info`, `/docs`, and `/openapi.json` before running `vercel --prod`.

The deployment includes ONNX Runtime, PyTorch, PyTorch Geometric, the model files, and the bundled Cora dataset. Those runtime dependencies increase function bundle size and cold-start time compared with a lightweight HTTP API.

## API examples

### Predict real Cora nodes

`POST /predict/cora_node`

```json
{
  "node_indices": [0, 42]
}
```

Each prediction includes the class, softmax probabilities, raw logits, and the node's real 1-hop Cora neighbors.

### Predict a custom graph

`POST /predict`

```json
{
  "node_features": [[0.0, 0.0, "... 1433 values total ..."]],
  "edge_indices": [[0], [0]]
}
```

Every node must contain exactly 1,433 features. Edge indices are validated against the submitted node count.

## Tests

Install development dependencies:

```powershell
pip install -r requirements-dev.txt
```

Then run:

```powershell
pytest -q
```

The test suite checks health/model metadata, real Cora inference, probability normalization, invalid node handling, invalid custom edges, and custom-graph inference.

## Correctness hardening applied

This cleaned version fixes several issues from the downloaded implementation:

- corrected `probabilites` → `probabilities` API contract
- uses the `SimpleGCN` architecture
- added missing Streamlit/Pandas/Requests dependencies
- added custom edge-index bounds checking
- made the browser citation visualization use actual Cora 1-hop neighbors instead of simulated neighbors
- removed misleading hard-coded topic labels from sample-node buttons
- changed Streamlit's default API URL to the local FastAPI service
- fixed early-stopping checkpoint capture with `copy.deepcopy(model.state_dict())`
- changed TF-IDF fitting so preprocessing statistics are learned from training nodes rather than the full graph

## Important modeling note

The custom-graph endpoint accepts arbitrary 1,433-dimensional feature vectors because that is the ONNX model's input contract. Meaningful predictions still require features encoded according to the same Cora vocabulary/feature semantics used during training. Random feature vectors are useful for exercising the API, not for interpreting scientific paper topics.
