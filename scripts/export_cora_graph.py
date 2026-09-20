"""Export the processed Cora graph as a lightweight NumPy runtime asset."""

from pathlib import Path

import numpy as np
from torch_geometric.datasets import Planetoid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "Planetoid"
OUTPUT_PATH = PROJECT_ROOT / "runtime" / "cora_graph.npz"


def main() -> None:
    graph = Planetoid(root=str(DATA_DIR), name="Cora")[0]
    node_features = graph.x.detach().cpu().numpy().astype(np.float32, copy=False)
    edge_indices = graph.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        node_features=node_features,
        edge_indices=edge_indices,
    )
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
