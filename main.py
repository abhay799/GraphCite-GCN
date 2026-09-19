"""FastAPI inference service for the Cora SimpleGCN ONNX model."""

import os
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

CORA_CLASSES = {
    0: "Case_Based",
    1: "Genetic_Algorithms",
    2: "Neural_Networks",
    3: "Probabilistic_Methods",
    4: "Reinforcement_Learning",
    5: "Rule_Learning",
    6: "Theory",
}

FEATURE_DIM = 1433
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "simple_gcn_cora.onnx")
DATA_DIR = os.path.join(BASE_DIR, "data", "Planetoid")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"ONNX model not found: {MODEL_PATH}")

model_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

app = FastAPI(
    title="Cora GCN Predictor",
    description="ONNX Graph Convolutional Network inference API for Cora node classification.",
    version="1.0.0",
)


class GraphPredictRequest(BaseModel):
    node_features: List[List[float]] = Field(min_length=1)
    edge_indices: Optional[List[List[int]]] = None


class CoraNodeRequest(BaseModel):
    node_indices: List[int] = Field(min_length=1)


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def run_model(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_indices_to_return: List[int],
) -> dict:
    output = model_session.run(
        ["logits"],
        {
            "node_features": node_features.astype(np.float32),
            "edge_indices": edge_index.astype(np.int64),
        },
    )
    logits = output[0]
    probabilities = softmax(logits)
    predicted_classes = logits.argmax(axis=-1)

    results = []
    for node_index in node_indices_to_return:
        class_id = int(predicted_classes[node_index])
        results.append(
            {
                "node_index": node_index,
                "predicted_class_id": class_id,
                "predicted_class_name": CORA_CLASSES[class_id],
                "probabilities": probabilities[node_index].tolist(),
                "logits": logits[node_index].tolist(),
            }
        )

    return {
        "num_nodes": int(node_features.shape[0]),
        "num_edges": int(edge_index.shape[1]),
        "predictions": results,
    }


def _load_cora_graph():
    """Load the local Cora graph. The processed dataset is bundled with this project."""
    from torch_geometric.datasets import Planetoid

    try:
        return Planetoid(root=DATA_DIR, name="Cora")[0]
    except Exception as error:  # pragma: no cover - converted into API error below
        raise HTTPException(500, f"Failed to load Cora dataset: {error}") from error


def _neighbors_for_node(edge_index: np.ndarray, node_index: int) -> List[int]:
    """Return unique 1-hop neighbors, treating the citation graph as local context."""
    src, dst = edge_index
    outgoing = dst[src == node_index]
    incoming = src[dst == node_index]
    neighbors = np.unique(np.concatenate([outgoing, incoming])).astype(int)
    neighbors = neighbors[neighbors != node_index]
    return neighbors.tolist()


@app.get("/")
def home_page():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"service": "SimpleGCN Cora API", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy", "providers": model_session.get_providers()}


@app.get("/info")
def model_info():
    return {
        "model_name": "SimpleGCN",
        "feature_dimension": FEATURE_DIM,
        "num_classes": len(CORA_CLASSES),
        "class_mapping": CORA_CLASSES,
        "inputs": [
            {"name": inp.name, "shape": inp.shape, "type": inp.type}
            for inp in model_session.get_inputs()
        ],
        "outputs": [
            {"name": out.name, "shape": out.shape, "type": out.type}
            for out in model_session.get_outputs()
        ],
    }


@app.post("/predict")
def predict_custom_graph(request: GraphPredictRequest):
    for feature_vector in request.node_features:
        if len(feature_vector) != FEATURE_DIM:
            raise HTTPException(
                422,
                f"Each node feature vector must contain exactly {FEATURE_DIM} values.",
            )

    node_features = np.asarray(request.node_features, dtype=np.float32)
    if not np.isfinite(node_features).all():
        raise HTTPException(422, "node_features must contain only finite numeric values.")

    num_nodes = node_features.shape[0]

    if request.edge_indices is not None:
        edge_index = np.asarray(request.edge_indices, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise HTTPException(422, "edge_indices must have shape [2, num_edges].")
        if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= num_nodes):
            raise HTTPException(
                422,
                f"edge_indices values must be between 0 and {num_nodes - 1}.",
            )
    else:
        node_ids = np.arange(num_nodes, dtype=np.int64)
        edge_index = np.vstack([node_ids, node_ids])

    return run_model(node_features, edge_index, list(range(num_nodes)))


@app.post("/predict/cora_node")
def predict_real_cora_nodes(request: CoraNodeRequest):
    cora_graph = _load_cora_graph()
    largest_valid_index = cora_graph.num_nodes - 1

    invalid_indices = [
        i for i in request.node_indices if i < 0 or i > largest_valid_index
    ]
    if invalid_indices:
        raise HTTPException(
            400,
            f"node_indices out of bounds: {invalid_indices}. Valid range is 0 to {largest_valid_index}.",
        )

    edge_index = cora_graph.edge_index.numpy()
    response = run_model(cora_graph.x.numpy(), edge_index, request.node_indices)

    for prediction in response["predictions"]:
        neighbors = _neighbors_for_node(edge_index, prediction["node_index"])
        prediction["neighbor_count"] = len(neighbors)
        prediction["neighbors"] = neighbors

    return response


if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
