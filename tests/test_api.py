import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_vercel_entrypoint_reexports_existing_app():
    from api.index import app as vercel_app

    assert vercel_app is main.app


def test_runtime_paths_are_absolute_module_relative():
    assert main.BASE_DIR.is_absolute()
    assert main.MODEL_PATH == main.BASE_DIR / "simple_gcn_cora.onnx"
    assert main.DATA_DIR == main.BASE_DIR / "data" / "Planetoid"
    assert main.STATIC_DIR == main.BASE_DIR / "static"


def test_vercel_configuration_routes_root_requests_and_includes_runtime_assets():
    config_path = Path(main.__file__).resolve().parent / "vercel.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["rewrites"] == [
        {"source": "/(.*)", "destination": "/api/index"}
    ]
    included = config["functions"]["api/index.py"]["includeFiles"]
    assert set(included) == {
        "simple_gcn_cora.onnx",
        "simple_gcn_cora.onnx.data",
        "data/**",
        "static/**",
    }


def test_vercelignore_excludes_local_only_files():
    ignore_path = Path(main.__file__).resolve().parent / ".vercelignore"
    ignored = set(ignore_path.read_text(encoding="utf-8").splitlines())

    assert {
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".git/",
        "tests/",
        "cora_citation_network_classification_(1).ipynb",
    }.issubset(ignored)


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
