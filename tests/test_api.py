import numpy as np
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "CPUExecutionProvider" in body["providers"]


def test_info_contract():
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "SimpleGCN"
    assert body["feature_dimension"] == 1433
    assert body["num_classes"] == 7


def test_cora_node_prediction_contract():
    response = client.post("/predict/cora_node", json={"node_indices": [0]})
    assert response.status_code == 200
    body = response.json()
    pred = body["predictions"][0]
    assert pred["node_index"] == 0
    assert len(pred["probabilities"]) == 7
    assert abs(sum(pred["probabilities"]) - 1.0) < 1e-5
    assert "probabilites" not in pred
    assert isinstance(pred["neighbors"], list)
    assert pred["neighbor_count"] == len(pred["neighbors"])


def test_rejects_out_of_range_cora_node():
    response = client.post("/predict/cora_node", json={"node_indices": [2708]})
    assert response.status_code == 400


def test_rejects_custom_edge_out_of_bounds():
    features = np.zeros((2, 1433), dtype=float).tolist()
    response = client.post(
        "/predict",
        json={"node_features": features, "edge_indices": [[0, 2], [1, 0]]},
    )
    assert response.status_code == 422


def test_custom_graph_prediction():
    features = np.zeros((2, 1433), dtype=float)
    features[0, 10] = 1.0
    features[1, 20] = 1.0
    response = client.post(
        "/predict",
        json={
            "node_features": features.tolist(),
            "edge_indices": [[0, 1], [1, 0]],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert all(len(p["probabilities"]) == 7 for p in body["predictions"])
