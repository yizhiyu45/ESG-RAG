# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from typing_extensions import NotRequired, TypedDict


class ChunkingConfigInput(TypedDict):
    """Configuration section for chunking."""

    size: NotRequired[int | str | None]
    overlap: NotRequired[int | str | None]
    group_by_columns: NotRequired[list[str] | str | None]
    strategy: NotRequired[dict | None]
    budget: NotRequired[float | str | None]
    build_skeleton_method: NotRequired[str | None]
    knn_edges: NotRequired[int | str | None]
    split_size: NotRequired[int | str | None]
    bridge_ratio: NotRequired[float | str | None]
    mmr_lambda: NotRequired[float | str | None]
    selection_seed: NotRequired[int | str | None]
