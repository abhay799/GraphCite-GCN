import ast
import json
from pathlib import Path
import tomllib

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
    assert main.RUNTIME_GRAPH_PATH == main.BASE_DIR / "runtime" / "cora_graph.npz"
    assert main.STATIC_DIR == main.BASE_DIR / "static"


def test_runtime_cora_graph_asset_has_expected_arrays():
    runtime_asset = Path(main.__file__).resolve().parent / "runtime" / "cora_graph.npz"

    with np.load(runtime_asset) as graph:
        assert graph["node_features"].dtype == np.float32
        assert graph["node_features"].shape == (2708, 1433)
        assert graph["edge_indices"].dtype == np.int64
        assert graph["edge_indices"].shape == (2, 10556)


def test_runtime_cora_graph_matches_processed_dataset():
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root=str(main.BASE_DIR / "data" / "Planetoid"), name="Cora")[0]
    runtime_asset = main.BASE_DIR / "runtime" / "cora_graph.npz"

    with np.load(runtime_asset) as graph:
        np.testing.assert_array_equal(graph["node_features"], dataset.x.numpy())
        np.testing.assert_array_equal(graph["edge_indices"], dataset.edge_index.numpy())


def test_production_inference_source_avoids_heavy_runtime_imports():
    source = Path(main.__file__).read_text(encoding="utf-8")
    directly_imported_modules = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    from_imported_modules = {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert {"torch", "torch_geometric", "streamlit", "pandas"}.isdisjoint(
        directly_imported_modules | from_imported_modules
    )


def test_vercel_configuration_routes_root_requests_and_includes_runtime_assets():
    config_path = Path(main.__file__).resolve().parent / "vercel.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["rewrites"] == [
        {"source": "/(.*)", "destination": "/api/index.py"}
    ]
    assert config["functions"]["api/index.py"]["includeFiles"] == (
        "{simple_gcn_cora.onnx,simple_gcn_cora.onnx.data,runtime/cora_graph.npz,static/**}"
    )


def test_vercelignore_excludes_local_only_files():
    ignore_path = Path(main.__file__).resolve().parent / ".vercelignore"
    ignored = set(ignore_path.read_text(encoding="utf-8").splitlines())

    assert {
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".git/",
        "tests/",
        "data/",
        "scripts/",
        "cora_citation_network_classification_(1).ipynb",
    }.issubset(ignored)


def test_root_requirements_are_limited_to_vercel_runtime_dependencies():
    root = Path(main.__file__).resolve().parent
    requirements = set((root / "requirements.txt").read_text(encoding="utf-8").splitlines())

    assert requirements == {
        "fastapi>=0.110.0",
        "uvicorn>=0.30.0",
        "onnxruntime>=1.18.0",
        "numpy>=1.24.0",
        "pydantic>=2.7.0",
    }


def test_development_requirements_keep_local_only_dependencies():
    root = Path(main.__file__).resolve().parent
    requirements = set((root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines())

    assert {
        "-r requirements.txt",
        "torch",
        "torch-geometric>=2.5.0",
        "streamlit>=1.30.0",
        "pandas>=2.0.0",
        "requests>=2.31.0",
        "pytest>=8.0.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "onnxscript>=0.1.0",
    }.issubset(requirements)


def test_pyproject_pins_vercel_to_python_313():
    root = Path(main.__file__).resolve().parent
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["requires-python"] == ">=3.13,<3.14"


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


def test_cora_node_zero_prediction_regression():
    response = client.post("/predict/cora_node", json={"node_indices": [0]})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["predicted_class_name"] == "Probabilistic_Methods"


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
