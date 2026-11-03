
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Hashable, Sequence
import json
import math

import networkx as nx
import numpy as np


NodeId = Hashable


@dataclass
class SelectionResult:
    selected_ids: list[NodeId]
    representative_ids: list[NodeId]
    bridge_ids: list[NodeId]
    community_of: dict[NodeId, int]
    quotas: dict[int, int]
    diagnostics: dict

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["selected_ids"] = [str(x) for x in self.selected_ids]
        data["representative_ids"] = [str(x) for x in self.representative_ids]
        data["bridge_ids"] = [str(x) for x in self.bridge_ids]
        data["community_of"] = {
            str(k): int(v) for k, v in self.community_of.items()
        }
        data["quotas"] = {str(k): int(v) for k, v in self.quotas.items()}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {x.shape}")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo <= 1e-12:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _communities(graph: nx.Graph, seed: int) -> list[set[NodeId]]:
    try:
        groups = nx.community.louvain_communities(
            graph, weight="weight", seed=seed
        )
    except (AttributeError, ImportError):
        groups = nx.community.greedy_modularity_communities(
            graph, weight="weight"
        )
    groups = [set(g) for g in groups]
    covered = set().union(*groups) if groups else set()
    groups.extend({n} for n in graph.nodes if n not in covered)
    groups.sort(key=lambda g: (-len(g), min(str(n) for n in g)))
    return groups


def _allocate(sizes: dict[int, int], total: int) -> dict[int, int]:
    quotas = {c: 0 for c in sizes}
    if total <= 0 or not sizes:
        return quotas

    order = sorted(sizes, key=lambda c: (-sizes[c], c))
    for c in order[: min(total, len(order))]:
        quotas[c] = 1

    left = total - sum(quotas.values())
    while left > 0:
        capacities = {c: sizes[c] - quotas[c] for c in sizes}
        available = [c for c in sizes if capacities[c] > 0]
        if not available:
            break

        weight_sum = sum(capacities[c] for c in available)
        raw = {c: left * capacities[c] / weight_sum for c in available}

        added = 0
        for c in available:
            count = min(capacities[c], int(math.floor(raw[c])))
            quotas[c] += count
            added += count

        left -= added
        if left <= 0:
            break

        ranked = sorted(
            available,
            key=lambda c: (
                -(raw[c] - math.floor(raw[c])),
                -sizes[c],
                c,
            ),
        )
        for c in ranked:
            if left <= 0:
                break
            if quotas[c] < sizes[c]:
                quotas[c] += 1
                left -= 1

    return quotas


def _select_mmr(
    positions: np.ndarray,
    embeddings: np.ndarray,
    quota: int,
    mmr_lambda: float,
) -> tuple[list[int], int | None]:
    if quota <= 0 or len(positions) == 0:
        return [], None
    if quota >= len(positions):
        values = positions.tolist()
        return values, values[0]

    local = embeddings[positions]
    centroid = local.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)

    representativeness = _minmax(local @ centroid)
    first = int(np.argmax(representativeness))

    chosen = [first]
    mask = np.zeros(len(positions), dtype=bool)
    mask[first] = True

    max_sim = np.clip(
        (local @ local[first] + 1.0) / 2.0, 0.0, 1.0
    )

    while len(chosen) < quota:
        novelty = 1.0 - max_sim
        score = (
            mmr_lambda * representativeness
            + (1.0 - mmr_lambda) * novelty
        )
        score[mask] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break

        chosen.append(nxt)
        mask[nxt] = True
        sim_new = np.clip(
            (local @ local[nxt] + 1.0) / 2.0, 0.0, 1.0
        )
        max_sim = np.maximum(max_sim, sim_new)

    return positions[chosen].tolist(), int(positions[first])


def select_topology_core_chunks(
    graph: nx.Graph,
    chunk_ids: Sequence[NodeId],
    embeddings: np.ndarray,
    *,
    beta: float = 0.8,
    target_count: int | None = None,
    bridge_ratio: float = 0.10,
    mmr_lambda: float = 0.65,
    seed: int = 42,
) -> SelectionResult:
    """
    Community coverage + intra-community MMR + bridge preservation.

    ``beta`` controls the retained fraction when ``target_count`` is omitted.
    """
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if not 0 <= bridge_ratio < 1:
        raise ValueError("bridge_ratio must be in [0, 1)")
    if not 0 <= mmr_lambda <= 1:
        raise ValueError("mmr_lambda must be in [0, 1]")

    chunk_ids = list(chunk_ids)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("chunk_ids contains duplicates")

    emb = _normalize(embeddings)
    if len(chunk_ids) != emb.shape[0]:
        raise ValueError("chunk_ids and embeddings are not aligned")

    missing = set(chunk_ids) - set(graph.nodes)
    if missing:
        raise ValueError(
            f"graph misses chunk ids, examples: {list(missing)[:5]}"
        )

    graph = graph.subgraph(chunk_ids).copy()
    if graph.is_directed():
        graph = graph.to_undirected()

    pos = {node: i for i, node in enumerate(chunk_ids)}

    if target_count is None:
        target = max(
            1,
            min(
                len(chunk_ids),
                int(len(chunk_ids) * beta),
            ),
        )
    else:
        target = max(
            1,
            min(
                len(chunk_ids),
                int(target_count),
            ),
        )

    bridge_target = min(target, round(target * bridge_ratio))
    normal_target = target - bridge_target

    groups = _communities(graph, seed)
    community_of: dict[NodeId, int] = {}
    sizes: dict[int, int] = {}

    for cid, group in enumerate(groups):
        sizes[cid] = len(group)
        for node in group:
            community_of[node] = cid

    quotas = _allocate(sizes, normal_target)

    selected_positions: list[int] = []
    representative_positions: list[int] = []

    for cid, group in enumerate(groups):
        member_positions = np.asarray(
            [pos[node] for node in group], dtype=np.int64
        )
        chosen, representative = _select_mmr(
            member_positions,
            emb,
            quotas.get(cid, 0),
            mmr_lambda,
        )
        selected_positions.extend(chosen)
        if representative is not None:
            representative_positions.append(representative)

    selected = set(selected_positions)

    # Weighted cross-community ratio and strongest connected community pair.
    bridge_score = np.zeros(len(chunk_ids), dtype=np.float32)
    strongest_pair: dict[NodeId, tuple[int, int] | None] = {}

    for node in graph.nodes:
        total = 0.0
        cross = 0.0
        pair_weight: dict[tuple[int, int], float] = {}
        c1 = community_of[node]

        for neighbor, data in graph[node].items():
            weight = max(0.0, float(data.get("weight", 1.0)))
            total += weight
            c2 = community_of[neighbor]
            if c1 != c2:
                cross += weight
                pair = tuple(sorted((c1, c2)))
                pair_weight[pair] = pair_weight.get(pair, 0.0) + weight

        bridge_score[pos[node]] = cross / total if total > 0 else 0.0
        strongest_pair[node] = (
            max(pair_weight, key=pair_weight.get) if pair_weight else None
        )

    bridge_score = _minmax(bridge_score)

    by_pair: dict[tuple[int, int], list[int]] = {}
    all_bridge_candidates: list[int] = []

    for node in chunk_ids:
        p = pos[node]
        pair = strongest_pair[node]
        if p in selected or pair is None or bridge_score[p] <= 0:
            continue
        by_pair.setdefault(pair, []).append(p)
        all_bridge_candidates.append(p)

    for values in by_pair.values():
        values.sort(key=lambda p: (-float(bridge_score[p]), str(chunk_ids[p])))

    pair_best = sorted(
        (values[0] for values in by_pair.values() if values),
        key=lambda p: (-float(bridge_score[p]), str(chunk_ids[p])),
    )

    bridge_positions: list[int] = []
    for p in pair_best:
        if len(bridge_positions) >= bridge_target:
            break
        bridge_positions.append(p)
        selected.add(p)

    all_bridge_candidates.sort(
        key=lambda p: (-float(bridge_score[p]), str(chunk_ids[p]))
    )
    for p in all_bridge_candidates:
        if len(bridge_positions) >= bridge_target:
            break
        if p not in selected:
            bridge_positions.append(p)
            selected.add(p)

    # Fill any remaining slots with global MMR.
    global_centroid = emb.mean(axis=0)
    global_centroid /= max(float(np.linalg.norm(global_centroid)), 1e-12)
    global_rep = _minmax(emb @ global_centroid)

    if selected:
        chosen_array = np.asarray(sorted(selected), dtype=np.int64)
        max_sim = np.clip(
            (emb @ emb[chosen_array].T).max(axis=1) * 0.5 + 0.5,
            0.0,
            1.0,
        )
    else:
        max_sim = np.zeros(len(chunk_ids), dtype=np.float32)

    while len(selected) < target:
        novelty = 1.0 - max_sim
        score = (
            mmr_lambda * global_rep
            + (1.0 - mmr_lambda) * novelty
        )
        score[np.asarray(list(selected), dtype=np.int64)] = -np.inf
        nxt = int(np.argmax(score))
        if not np.isfinite(score[nxt]):
            break
        selected.add(nxt)
        sim_new = np.clip((emb @ emb[nxt] + 1.0) / 2.0, 0.0, 1.0)
        max_sim = np.maximum(max_sim, sim_new)

    final_positions = [i for i in range(len(chunk_ids)) if i in selected][:target]

    selected_ids = [chunk_ids[i] for i in final_positions]
    representative_ids = [
        chunk_ids[i] for i in representative_positions if i in selected
    ]
    bridge_ids = [chunk_ids[i] for i in bridge_positions if i in selected]

    selected_communities = {community_of[node] for node in selected_ids}

    return SelectionResult(
        selected_ids=selected_ids,
        representative_ids=representative_ids,
        bridge_ids=bridge_ids,
        community_of=community_of,
        quotas=quotas,
        diagnostics={
            "num_chunks": len(chunk_ids),
            "num_edges": graph.number_of_edges(),
            "num_communities": len(groups),
            "target_count": target,
            "selected_count": len(selected_ids),
            "selection_ratio": len(selected_ids) / len(chunk_ids),
            "selected_communities": len(selected_communities),
            "community_coverage": (
                len(selected_communities) / len(groups) if groups else 0.0
            ),
            "bridge_target": bridge_target,
            "actual_bridge_count": len(bridge_ids),
            "beta": beta,
            "requested_target_count": target_count,
            "bridge_ratio": bridge_ratio,
            "mmr_lambda": mmr_lambda,
        },
    )
