import networkx as nx
import numpy as np

from esg_rag.topology import select_topology_core_chunks


def test_topology_selection_is_deterministic_and_respects_budget():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [("a", "b", 0.9), ("b", "c", 0.8), ("c", "d", 0.7), ("a", "d", 0.2)]
    )
    embeddings = np.eye(4, dtype=np.float32)
    ids = ["a", "b", "c", "d"]

    first = select_topology_core_chunks(graph, ids, embeddings, target_count=2, seed=7)
    second = select_topology_core_chunks(graph, ids, embeddings, target_count=2, seed=7)

    assert first.selected_ids == second.selected_ids
    assert len(first.selected_ids) == 2
    assert set(first.selected_ids).issubset(ids)

