import pandas as pd
import copy
import io
import random
import re
import numpy as np
import regex
from sklearn.metrics.pairwise import cosine_similarity
from typing import Any
from collections import Counter
from tqdm import tqdm

from graphrag.index.cache import InMemoryCache
from graphrag.query.input.loaders.dfs import (
    read_text_units
)
from graphrag.model import (
    TextUnit
)
from graphrag.index.utils import gen_md5_hash
from esg_rag.embedding.embedding import OpenAIBatchAsyncEmbedding
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.index.operations.embed_text import embed_text
import tiktoken
import networkx as nx
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from graphrag.index.operations.snapshot import snapshot
from graphrag.index.storage import PipelineStorage
from datashaper import VerbCallbacks
from esg_rag.keyword_index import build_keyword_index
from esg_rag.topology import (
    select_topology_core_chunks,
)


async def filter_text_units(
    text_units_df: pd.DataFrame,
    storage: PipelineStorage,
    callbacks: VerbCallbacks,
    chunk_strategy: dict[str, Any] | None = None,
    embedding_strategy: dict[str, Any] | None = None
) -> pd.DataFrame:
    budget = chunk_strategy['budget']
    build_skeleton_method = chunk_strategy['build_skeleton_method']
    knn_edges = chunk_strategy['knn_edges']
    split_size = chunk_strategy['split_size']

    selection_seed = int(
        chunk_strategy.get(
            'selection_seed',
            42
        )
    )
    random.seed(selection_seed)

    token_encoder = tiktoken.get_encoding(
        chunk_strategy.get(
            'encoding_name',
            'cl100k_base'
        )
    )

    text_units = read_text_units(
        text_units_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col=None
    )

    text_units = split_chunks(
        text_units,
        token_encoder,
        split_size
    )

    text_embedding_bytes = await storage.get(
        'split_text_units.parquet',
        as_bytes=True
    )

    if text_embedding_bytes is not None:
        print("Text Embedding file has been read!")
        buffer = io.BytesIO(text_embedding_bytes)
        text_embedding_df = pd.read_parquet(buffer)

    else:
        print("Generating Text Embedding file")

        text_embedding_df = pd.DataFrame([
            {
                "id": unit.id,
                "chunk": unit.text,
                "chunk_id": unit.id,
                "document_ids": unit.document_ids,
                "n_tokens": unit.n_tokens
            }
            for unit in text_units
        ])

        text_embedding_df["text_embedding"] = await embed_text(
            text_embedding_df,
            callbacks=callbacks,
            cache=InMemoryCache(),
            embed_column="chunk",
            embedding_name="text_embedding",
            strategy=embedding_strategy,
        )

        await snapshot(
            text_embedding_df,
            name="split_text_units",
            storage=storage,
            formats=["parquet"],
        )

    # 关键词—文本通道保持原样，仍然使用全部文本块。
    keyword_file = await storage.get(
        'keyword_index.parquet',
        as_bytes=True
    )

    if keyword_file is None:
        print("Generating Keyword file")

        await build_keyword_index(
            text_embedding_df,
            storage,
            callbacks,
            embedding_strategy
        )

    else:
        print("Keyword file has been read!")

    # 重新读入带有 embedding 的 TextUnit。
    text_units = read_text_units(
        text_embedding_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col='text_embedding'
    )

    # 与原代码完全一致：使用 floor，而不是 ceil。
    sample_length = int(
        len(text_units) * budget
    )

    print("=" * 80)
    print(
        "Build skeleton method:",
        build_skeleton_method
    )
    print(
        "Total split text units:",
        len(text_units)
    )
    print(
        "Core sample length:",
        sample_length
    )
    print("=" * 80)

    if sample_length <= 0:
        sampled_text_units = []

    elif build_skeleton_method == "uniform":
        random.shuffle(text_units)
        sampled_text_units = text_units[:sample_length]

        # 保存随机消融的选择清单，便于核对它与 Topology
        # 使用了相同预算、相同候选集合，但选择结果不同。
        selected_ids = {
            unit.id
            for unit in sampled_text_units
        }
        selection_report_df = pd.DataFrame([
            {
                "id": unit.id,
                "chunk": unit.text,
                "n_tokens": unit.n_tokens,
                "selected": unit.id in selected_ids,
                "selection_seed": selection_seed,
            }
            for unit in text_units
        ])
        await snapshot(
            selection_report_df,
            name="uniform_core_selection",
            storage=storage,
            formats=["parquet"],
        )

    elif build_skeleton_method == "pagerank":
        # PageRank selector retained as an optional ablation mode.
        G = build_knn_chunk_graph(
            text_units,
            knn_edges,
            knn_edges
        )

        pr = nx.pagerank(
            G,
            alpha=0.85
        )

        sampled_text_units = sorted(
            text_units,
            key=lambda x: pr.get(
                x.id,
                0
            ),
            reverse=True
        )[:sample_length]

    elif build_skeleton_method in {
        "topology",
        "topocore",
        "topology_core",
    }:
        # 新方法：
        # 1. KNN图社区覆盖；
        # 2. 社区内部代表性与去冗余选择；
        # 3. 跨社区桥接文本块补充。
        G = build_knn_chunk_graph(
            text_units,
            knn_edges,
            knn_edges
        )

        chunk_ids = [
            unit.id
            for unit in text_units
        ]

        chunk_embeddings = np.asarray(
            [
                unit.text_embedding
                for unit in text_units
            ],
            dtype=np.float32,
        )

        selection_result = (
            select_topology_core_chunks(
                graph=G,
                chunk_ids=chunk_ids,
                embeddings=chunk_embeddings,

                # 精确使用原代码的 sample_length，
                # 保证新旧方法选择数量完全一致。
                target_count=sample_length,

                # 约10%的核心块预算用于桥接块。
                bridge_ratio=float(
                    chunk_strategy.get(
                        'bridge_ratio',
                        0.10
                    )
                ),

                # 0.65偏向社区代表性，
                # 0.35用于抑制重复。
                mmr_lambda=float(
                    chunk_strategy.get(
                        'mmr_lambda',
                        0.65
                    )
                ),

                seed=int(
                    chunk_strategy.get(
                        'selection_seed',
                        42
                    )
                ),
            )
        )

        selected_ids = set(
            selection_result.selected_ids
        )

        sampled_text_units = [
            unit
            for unit in text_units
            if unit.id in selected_ids
        ]

        # 保存每个文本块的选择信息，便于论文统计和消融。
        representative_ids = set(
            selection_result.representative_ids
        )

        bridge_ids = set(
            selection_result.bridge_ids
        )

        selection_report_df = pd.DataFrame([
            {
                "id": unit.id,
                "chunk": unit.text,
                "n_tokens": unit.n_tokens,
                "community_id": (
                    selection_result.community_of.get(
                        unit.id,
                        -1
                    )
                ),
                "selected": (
                    unit.id in selected_ids
                ),
                "is_representative": (
                    unit.id in representative_ids
                ),
                "is_bridge": (
                    unit.id in bridge_ids
                ),
            }
            for unit in text_units
        ])

        await snapshot(
            selection_report_df,
            name="topology_core_selection",
            storage=storage,
            formats=["parquet"],
        )

        print("=" * 80)
        print(
            "Topology core selection finished"
        )
        print(
            "Diagnostics:",
            selection_result.diagnostics
        )
        print(
            "Actual selected:",
            len(sampled_text_units)
        )
        print(
            "Representative chunks:",
            len(representative_ids)
        )
        print(
            "Bridge chunks:",
            len(bridge_ids)
        )
        print("=" * 80)

        if len(sampled_text_units) != sample_length:
            raise RuntimeError(
                "Topology selector returned an unexpected "
                "number of chunks: "
                f"{len(sampled_text_units)} != {sample_length}"
            )

    else:
        raise ValueError(
            "Unsupported build_skeleton_method: "
            f"{build_skeleton_method}. "
            "Expected one of: uniform, pagerank, topology."
        )

    sampled_text_units = sorted(
        sampled_text_units,
        key=lambda chunk: int(
            chunk.id
        )
    )

    if split_size < chunk_strategy.get(
        'chunk_size',
        1200
    ):
        sampled_text_units = merge_chunks(
            sampled_text_units,
            token_encoder,
            chunk_strategy.get(
                'chunk_size',
                1200
            )
        )

    sampled_df = pd.DataFrame([
        {
            "id": (
                hash_value := gen_md5_hash(
                    {
                        "chunk": unit.text
                    },
                    [
                        "chunk"
                    ]
                )
            ),
            "chunk": unit.text,
            "chunk_id": hash_value,
            "document_ids": unit.document_ids,
            "n_tokens": unit.n_tokens,
        }
        for unit in sampled_text_units
    ])

    return sampled_df


def split_chunks(
    chunks: list[TextUnit],
    token_encoder,
    new_size
) -> list[TextUnit]:
    results = []
    total_num = 0

    for chunk in chunks:
        if new_size < 1200:
            text_token = token_encoder.encode(
                chunk.text
            )

            split_texts = [
                token_encoder.decode(
                    text_token[
                        i:i + new_size
                    ]
                )
                for i in range(
                    0,
                    len(text_token),
                    new_size
                )
            ]

        else:
            split_texts = [
                chunk.text
            ]

        for text in split_texts:
            new_chunk = copy.deepcopy(
                chunk
            )

            new_chunk.text = text
            new_chunk.id = str(
                total_num
            )
            new_chunk.n_tokens = len(
                token_encoder.encode(
                    text
                )
            )

            results.append(
                new_chunk
            )
            total_num += 1

    return results


def merge_chunks(
    chunks: list[TextUnit],
    token_encoder,
    new_size
) -> list[TextUnit]:
    results = []
    token_result = []
    document_ids = set()
    current_id = ""

    for chunk in chunks:
        text_tokens = token_encoder.encode(
            chunk.text
        )

        if (
            len(token_result)
            + len(text_tokens)
            > new_size
        ):
            new_chunk = TextUnit(
                id=current_id,
                short_id=current_id,
                text=token_encoder.decode(
                    token_result
                ),
                document_ids=list(
                    document_ids
                ),
                n_tokens=len(
                    token_result
                )
            )

            results.append(
                new_chunk
            )

            token_result = []
            document_ids = set()
            current_id = ""

        token_result += text_tokens

        document_ids.update(
            chunk.document_ids
            or []
        )

        current_id = (
            current_id
            or chunk.id
        )

    if token_result:
        new_chunk = TextUnit(
            id=current_id,
            short_id=current_id,
            text=token_encoder.decode(
                token_result
            ),
            document_ids=list(
                document_ids
            ),
            n_tokens=len(
                token_result
            )
        )

        results.append(
            new_chunk
        )

    return results


def build_knn_chunk_graph(
    text_units: list[TextUnit],
    k1: int,
    k2: int
) -> nx.Graph:
    # Keep the two evidence channels aligned:
    # 共现边 + embedding相似边。
    co_occurrence_text_pairs = (
        find_top_k_cooccurrence_pairs(
            text_units,
            k=k1
        )
    )

    similarity_text_pairs = (
        find_top_k_similar_texts(
            text_units,
            k=k2
        )
    )

    text_pairs = (
        list(
            co_occurrence_text_pairs
        )
        + list(
            similarity_text_pairs
        )
    )

    G = nx.Graph()

    for text_unit in text_units:
        G.add_node(
            text_unit.id
        )

    for u, v in text_pairs:
        G.add_edge(
            u,
            v
        )

    return G


def find_top_k_cooccurrence_pairs(
    text_units: list[TextUnit],
    k: int,
    bow_threshold: int = 2000
) -> list[tuple[str, str]]:
    def remove_stopwords(
        text
    ):
        stop_words = set(
            stopwords.words(
                'english'
            )
        )

        word_tokens = word_tokenize(
            text.lower()
        )

        filtered_words = [
            word
            for word in word_tokens
            if (
                word.isalpha()
                and word not in stop_words
            )
        ]

        word_freq = Counter(
            filtered_words
        )

        most_common_words = [
            word
            for word, freq
            in word_freq.most_common(
                bow_threshold
            )
        ]

        return set(
            most_common_words
        )

    processed_texts = {
        text_unit.id: remove_stopwords(
            text_unit.text
        )
        for text_unit in text_units
    }

    cooccurrence_pairs = {}
    keys = list(
        processed_texts.keys()
    )

    for i in range(
        len(keys)
    ):
        for j in range(
            i + 1,
            len(keys)
        ):
            words1 = processed_texts[
                keys[i]
            ]

            words2 = processed_texts[
                keys[j]
            ]

            common_words = (
                words1.intersection(
                    words2
                )
            )

            cooccurrence_count = len(
                common_words
            )

            if keys[i] not in cooccurrence_pairs:
                cooccurrence_pairs[
                    keys[i]
                ] = []

            if keys[j] not in cooccurrence_pairs:
                cooccurrence_pairs[
                    keys[j]
                ] = []

            cooccurrence_pairs[
                keys[i]
            ].append(
                (
                    cooccurrence_count,
                    keys[j]
                )
            )

            cooccurrence_pairs[
                keys[j]
            ].append(
                (
                    cooccurrence_count,
                    keys[i]
                )
            )

    top_k_pairs = []

    for key, pairs in cooccurrence_pairs.items():
        pairs_sorted = sorted(
            pairs,
            key=lambda x: x[0],
            reverse=True
        )

        for count, other_key in pairs_sorted[:k]:
            top_k_pairs.append(
                (
                    key,
                    other_key
                )
            )

    return top_k_pairs


def find_top_k_similar_texts(
    text_units: list[TextUnit],
    k: int
) -> list[tuple[str, str]]:
    text_keys = [
        text_unit.id
        for text_unit in text_units
    ]

    embeddings = np.array([
        text.text_embedding
        for text in text_units
    ])

    cosine_similarities = (
        cosine_similarity(
            embeddings
        )
    )

    top_k_pairs = []

    for i, key in enumerate(
        text_keys
    ):
        similarities = (
            cosine_similarities[i]
        )

        top_k_indices = np.argsort(
            -similarities
        )[1:k + 1]

        for idx in top_k_indices:
            similar_key = (
                text_keys[idx]
            )

            top_k_pairs.append(
                (
                    key,
                    similar_key
                )
            )

    return top_k_pairs
