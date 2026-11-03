from __future__ import annotations

"""
ESG-RAG 双通道上下文生成脚本。

本脚本实现 ESG-RAG 的检索反馈双通道上下文生成流程：
1. 一次初始化结构通道与关键词文本通道；
2. 原问题和每个实际子查询都执行双通道检索；
3. 使用 RF-DQD 依赖式执行子查询；
4. 结构上下文和文本上下文始终独立累计；
5. 保存 RF-DQD 完整追踪文件；
6. 分别保存可直接用于后续答案生成的两个上下文文件。

运行示例：
    python -m esg_rag.retrieve <dataset_root> 10

参数：
    第1个参数：项目根目录
    第2个参数：样本数，可省略，默认10，最多500
"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

from graphrag.model import Entity, Relationship, TextUnit
from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import (
    read_text_units,
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import (
    LocalSearch,
)
from graphrag.vector_stores.lancedb import LanceDBVectorStore

from esg_rag.decompose import (
    ChannelRetrieval,
    RFDQDConfig,
    RetrievalFeedbackQueryDecomposer,
    build_dual_retriever,
)


COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2

MAX_SAMPLES = 500
STRUCTURAL_CONTEXT_TOKENS = int(
    os.getenv(
        "RF_DQD_STRUCTURAL_TOKENS",
        "12000",
    )
)
TEXTUAL_CONTEXT_TOKENS = int(
    os.getenv(
        "RF_DQD_TEXTUAL_TOKENS",
        "12000",
    )
)
FINAL_CONTEXT_TOKENS = int(
    os.getenv(
        "RF_DQD_FINAL_CONTEXT_TOKENS",
        "11000",
    )
)
RF_DQD_CONCURRENCY = int(
    os.getenv(
        "RF_DQD_CONCURRENCY",
        "2",
    )
)


# =============================================================================
# 文件与文本工具
# =============================================================================


def load_json(path: str) -> list[dict[str, Any]]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise TypeError(
            f"JSON根节点必须是列表：{path}"
        )

    return data


def save_json_atomic(
    data: Any,
    path: str,
) -> None:
    os.makedirs(
        os.path.dirname(path) or ".",
        exist_ok=True,
    )
    temporary_path = path + ".tmp"

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


def l2_normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    """将向量归一化后再用 IndexFlatIP 计算真实余弦相似度。"""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return array / norms


def truncate_to_tokens(
    text: str,
    encoder: tiktoken.Encoding,
    max_tokens: int,
) -> str:
    text = str(text or "")

    if not text or max_tokens <= 0:
        return ""

    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return encoder.decode(
        tokens[:max_tokens]
    )


def wrap_query_context(
    query: str,
    channel: str,
    context: str,
) -> str:
    return (
        f"===== {channel} retrieval query: "
        f"{query} =====\n"
        f"{str(context or '').strip()}"
    )


# =============================================================================
# ESG-RAG 索引加载
# =============================================================================


def split_chunks_and_others(
    chunks: list[TextUnit],
    entities: list[Entity],
    relations: list[Relationship],
    token_encoder: tiktoken.Encoding,
    new_size: int = 300,
) -> tuple[
    list[TextUnit],
    list[Entity],
    list[Relationship],
]:
    results: list[TextUnit] = []
    split_count: dict[str, int] = {}

    for chunk in chunks:
        text_tokens = token_encoder.encode(
            chunk.text
        )
        split_texts = [
            token_encoder.decode(
                text_tokens[index : index + new_size]
            )
            for index in range(
                0,
                len(text_tokens),
                new_size,
            )
        ]

        split_count[chunk.id] = len(
            split_texts
        )

        for index, text in enumerate(
            split_texts
        ):
            new_chunk = copy.deepcopy(chunk)
            new_chunk.text = text
            new_chunk.id = (
                f"{new_chunk.id}_{index}"
            )
            results.append(new_chunk)

    for entity in entities:
        new_ids: list[str] = []

        for chunk_id in entity.text_unit_ids:
            new_ids.extend(
                [
                    f"{chunk_id}_{index}"
                    for index in range(
                        split_count.get(
                            chunk_id,
                            0,
                        )
                    )
                ]
            )

        entity.text_unit_ids = new_ids

    for relation in relations:
        new_ids: list[str] = []

        for chunk_id in relation.text_unit_ids:
            new_ids.extend(
                [
                    f"{chunk_id}_{index}"
                    for index in range(
                        split_count.get(
                            chunk_id,
                            0,
                        )
                    )
                ]
            )

        relation.text_unit_ids = new_ids

    return results, entities, relations


def initialize_retrieval_components(
    root_path: str,
) -> dict[str, Any]:
    output_dir = os.path.join(
        root_path,
        "output",
    )
    lancedb_uri = os.path.join(
        root_path,
        "lancedb-new",
    )

    node_df = pd.read_parquet(
        os.path.join(
            output_dir,
            f"{ENTITY_TABLE}.parquet",
        )
    )
    entity_df = pd.read_parquet(
        os.path.join(
            output_dir,
            f"{ENTITY_EMBEDDING_TABLE}.parquet",
        )
    )
    entity_embedding_df = pd.read_parquet(
        os.path.join(
            output_dir,
            "embeddings.entity.description.parquet",
        )
    ).rename(
        columns={
            "embedding": (
                "description_embedding"
            )
        }
    )

    if (
        "description_embedding"
        in entity_df.columns
    ):
        entity_df = entity_df.drop(
            columns=[
                "description_embedding"
            ]
        )

    entity_df = entity_df.merge(
        entity_embedding_df,
        on="id",
        how="inner",
    )

    entities = read_indexer_entities(
        node_df,
        entity_df,
        COMMUNITY_LEVEL,
    )

    # 使用与GraphRAG实体映射相同的描述嵌入建立可审计的结构通道
    # 余弦检索器。该检索器只负责暴露真实实体排名和raw_score，
    # 结构上下文本身仍由LocalSearchMixedContext生成。
    structural_entity_ids: list[str] = []
    structural_entity_vectors: list[np.ndarray] = []
    for _, row in entity_df.iterrows():
        embedding = row.get("description_embedding")
        if embedding is None:
            continue
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            continue
        structural_entity_ids.append(str(row["id"]))
        structural_entity_vectors.append(vector)

    if not structural_entity_vectors:
        raise RuntimeError("结构通道没有可用的实体描述嵌入")

    structural_matrix = l2_normalize_rows(
        np.stack(structural_entity_vectors)
    )
    structural_entity_index = faiss.IndexFlatIP(
        structural_matrix.shape[1]
    )
    structural_entity_index.add(structural_matrix)

    description_embedding_store = (
        LanceDBVectorStore(
            collection_name=(
                "entity.description"
            )
        )
    )
    description_embedding_store.connect(
        db_uri=lancedb_uri
    )
    store_entity_semantic_embeddings(
        entities=entities,
        vectorstore=(
            description_embedding_store
        ),
    )

    relationship_df = pd.read_parquet(
        os.path.join(
            output_dir,
            f"{RELATIONSHIP_TABLE}.parquet",
        )
    )
    relationships = (
        read_indexer_relationships(
            relationship_df
        )
    )

    report_df = pd.read_parquet(
        os.path.join(
            output_dir,
            f"{COMMUNITY_REPORT_TABLE}.parquet",
        )
    )
    reports = read_indexer_reports(
        report_df,
        node_df,
        COMMUNITY_LEVEL,
    )

    text_unit_df = pd.read_parquet(
        os.path.join(
            output_dir,
            f"{TEXT_UNIT_TABLE}.parquet",
        )
    )
    text_unit_embedding_df = (
        pd.read_parquet(
            os.path.join(
                output_dir,
                "embeddings.text_unit.text.parquet",
            )
        )
    )
    text_units = read_indexer_text_units(
        text_unit_df
    )

    embedding_dict = dict(
        zip(
            text_unit_embedding_df["id"],
            text_unit_embedding_df[
                "embedding"
            ],
        )
    )

    for text_unit in text_units:
        if text_unit.id in embedding_dict:
            text_unit.text_embedding = (
                embedding_dict[text_unit.id]
            )

    token_encoder = tiktoken.get_encoding(
        "cl100k_base"
    )

    (
        graph_text_units,
        entities,
        relationships,
    ) = split_chunks_and_others(
        text_units,
        entities,
        relationships,
        token_encoder,
        new_size=300,
    )

    split_text_units_df = pd.read_parquet(
        os.path.join(
            output_dir,
            "split_text_units.parquet",
        )
    )
    keyword_df = pd.read_parquet(
        os.path.join(
            output_dir,
            "keyword_index.parquet",
        )
    )
    split_text_units = read_text_units(
        split_text_units_df,
        text_col="chunk",
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col="text_embedding",
    )

    openai_api_key = os.getenv(
        "OPENAI_API_KEY"
    )
    if not openai_api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY"
        )

    # 有独立 GRAPHRAG_API_KEY 时沿用当前 DeepSeek 配置；
    # 否则使用 OpenAI Key 和 gpt-4o-mini，避免接口与密钥不匹配。
    graphrag_api_key = os.getenv(
        "GRAPHRAG_API_KEY"
    )

    if graphrag_api_key:
        graphrag_model = os.getenv(
            "GRAPHRAG_LLM_MODEL",
            "deepseek-chat",
        )
        graphrag_base_url = os.getenv(
            "GRAPHRAG_API_BASE",
            "https://api.deepseek.com/v1",
        )
    else:
        graphrag_api_key = openai_api_key
        graphrag_model = os.getenv(
            "OPENAI_ANSWER_MODEL",
            "gpt-4o-mini",
        )
        graphrag_base_url = None

    graph_llm = ChatOpenAI(
        api_key=graphrag_api_key,
        model=graphrag_model,
        api_type=OpenaiApiType.OpenAI,
        api_base=graphrag_base_url,
        max_retries=20,
    )

    embedding_model = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    text_embedder = OpenAIEmbedding(
        api_key=openai_api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )

    context_builder = (
        LocalSearchMixedContext(
            community_reports=reports,
            text_units=graph_text_units,
            entities=entities,
            relationships=relationships,
            covariates=None,
            entity_text_embeddings=(
                description_embedding_store
            ),
            embedding_vectorstore_key=(
                EntityVectorStoreKey.ID
            ),
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )
    )

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.0,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": (
            EntityVectorStoreKey.ID
        ),
        "max_tokens": (
            STRUCTURAL_CONTEXT_TOKENS
        ),
    }

    search_engine = LocalSearch(
        llm=graph_llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params={
            "max_tokens": 2000,
            "temperature": 0.0,
        },
        context_builder_params=(
            local_context_params
        ),
        response_type=(
            "A direct, accurate, and "
            "sufficiently complete answer"
        ),
    )

    candidate_units_dict = {
        unit.id: unit
        for unit in split_text_units
    }

    word_embeddings = {
        row["word"]: row["embedding"]
        for _, row in keyword_df.iterrows()
    }
    word_chunks = {
        row["word"]: row["chunk_ids"]
        for _, row in keyword_df.iterrows()
    }
    words = list(
        word_embeddings.keys()
    )

    if not words:
        raise RuntimeError(
            "keyword_index.parquet 中没有关键词"
        )

    keyword_vectors = l2_normalize_rows(np.asarray(
        [
            word_embeddings[word]
            for word in words
        ],
        dtype=np.float32,
    ))
    keyword_index = faiss.IndexFlatIP(
        keyword_vectors.shape[1]
    )
    keyword_index.add(keyword_vectors)

    token_counts = {
        unit.id: num_tokens(
            unit.text,
            token_encoder,
        )
        for unit in split_text_units
    }

    word_chunk_data: dict[
        str,
        tuple[
            list[str],
            np.ndarray,
            list[int],
        ],
    ] = {}

    for word in words:
        chunk_ids = [
            chunk_id
            for chunk_id in list(
                word_chunks.get(
                    word,
                    [],
                )
            )
            if chunk_id
            in candidate_units_dict
        ]

        if not chunk_ids:
            continue

        chunk_embeddings = l2_normalize_rows(np.stack(
            [
                candidate_units_dict[
                    chunk_id
                ].text_embedding
                for chunk_id in chunk_ids
            ]
        ).astype(np.float32))

        chunk_token_counts = [
            token_counts[chunk_id]
            for chunk_id in chunk_ids
        ]

        word_chunk_data[word] = (
            chunk_ids,
            chunk_embeddings,
            chunk_token_counts,
        )

    return {
        "search_engine": search_engine,
        "text_embedder": text_embedder,
        "token_encoder": token_encoder,
        "keyword_index": keyword_index,
        "words": words,
        "word_chunk_data": word_chunk_data,
        "candidate_units_dict": (
            candidate_units_dict
        ),
        "structural_entity_index": (
            structural_entity_index
        ),
        "structural_entity_ids": (
            structural_entity_ids
        ),
    }


# =============================================================================
# 双通道检索
# =============================================================================


def retrieve_structural_context(
    query: str,
    search_engine: LocalSearch,
    text_embedder: OpenAIEmbedding,
    structural_entity_index: faiss.IndexFlatIP,
    structural_entity_ids: list[str],
) -> ChannelRetrieval:
    context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.0,
        "conversation_history_max_turns": 10,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "max_tokens": (
            STRUCTURAL_CONTEXT_TOKENS
        ),
    }

    context_result = (
        search_engine.context_builder.build_context(
            query,
            **context_params,
        )
    )
    raw_context = str(
        context_result.context_chunks or ""
    )

    query_embedding = l2_normalize_rows(
        np.asarray(
            text_embedder.embed(query),
            dtype=np.float32,
        )
    )
    structural_top_k = min(
        int(context_params["top_k_mapped_entities"]),
        structural_entity_index.ntotal,
    )
    final_ids: list[str] = []
    final_scores: list[float] = []
    if structural_top_k > 0:
        similarities, indices = structural_entity_index.search(
            query_embedding,
            structural_top_k,
        )
        for position, entity_index in enumerate(indices[0]):
            if (
                entity_index < 0
                or entity_index >= len(structural_entity_ids)
            ):
                continue
            final_ids.append(
                str(structural_entity_ids[entity_index])
            )
            # 保存归一化前的余弦检索结果，不做[0,1]截断。
            final_scores.append(
                float(similarities[0, position])
            )

    return ChannelRetrieval(
        context=wrap_query_context(
            query,
            "STRUCTURAL",
            raw_context,
        ),
        item_ids=final_ids,
        scores=final_scores,
        score_type="cosine_similarity",
    )


def retrieve_keyword_context(
    query: str,
    text_embedder: OpenAIEmbedding,
    token_encoder: tiktoken.Encoding,
    keyword_index: faiss.IndexFlatIP,
    words: list[str],
    word_chunk_data: dict[
        str,
        tuple[
            list[str],
            np.ndarray,
            list[int],
        ],
    ],
    candidate_units_dict: dict[
        str,
        TextUnit,
    ],
) -> ChannelRetrieval:
    query_embedding = text_embedder.embed(
        query
    )
    query_embedding = l2_normalize_rows(
        np.asarray(
            query_embedding,
            dtype=np.float32,
        )
    )

    keyword_top_k = min(
        1000,
        keyword_index.ntotal,
    )
    distances, indices = (
        keyword_index.search(
            query_embedding,
            keyword_top_k,
        )
    )

    ranked_words: list[
        tuple[str, float]
    ] = []

    for position, word_index in enumerate(
        indices[0]
    ):
        if (
            word_index < 0
            or word_index >= len(words)
        ):
            continue

        ranked_words.append(
            (
                words[word_index],
                float(
                    distances[0, position]
                ),
            )
        )

    selected_chunks: list[
        tuple[str, float]
    ] = []
    visited_chunks: set[str] = set()
    candidate_token_total = 0
    is_full = False

    for word, _ in ranked_words:
        if is_full:
            break

        data = word_chunk_data.get(word)
        if data is None:
            continue

        (
            chunk_ids,
            chunk_embeddings,
            chunk_token_counts,
        ) = data

        similarities = (
            query_embedding
            @ chunk_embeddings.T
        ).flatten()
        sorted_indices = np.argsort(
            -similarities
        )

        for chunk_position in sorted_indices:
            chunk_id = chunk_ids[
                chunk_position
            ]
            chunk_tokens = (
                chunk_token_counts[
                    chunk_position
                ]
            )

            if (
                candidate_token_total
                + chunk_tokens
                > TEXTUAL_CONTEXT_TOKENS * 2
            ):
                is_full = True
                break

            if chunk_id in visited_chunks:
                continue

            visited_chunks.add(chunk_id)
            candidate_token_total += (
                chunk_tokens
            )
            selected_chunks.append(
                (
                    chunk_id,
                    float(
                        similarities[
                            chunk_position
                        ]
                    ),
                )
            )

    selected_chunks.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    header = (
        "-----Text source that may "
        "be relevant-----\n"
        "id|text\n"
    )
    text_parts = [header]
    used_tokens = num_tokens(
        header,
        token_encoder,
    )
    final_ids: list[str] = []
    final_scores: list[float] = []

    for index, (
        chunk_id,
        score,
    ) in enumerate(
        selected_chunks,
        start=1,
    ):
        segment = (
            f"chunk_{index}|"
            f"{candidate_units_dict[chunk_id].text}"
            "\n"
        )
        segment_tokens = num_tokens(
            segment,
            token_encoder,
        )

        if (
            used_tokens + segment_tokens
            > TEXTUAL_CONTEXT_TOKENS
        ):
            break

        text_parts.append(segment)
        used_tokens += segment_tokens
        final_ids.append(str(chunk_id))
        # 与结构通道一致保存真实余弦分数，不在检索阶段截断。
        final_scores.append(float(score))

    raw_context = "".join(text_parts)

    return ChannelRetrieval(
        context=wrap_query_context(
            query,
            "TEXTUAL",
            raw_context,
        ),
        item_ids=final_ids,
        scores=final_scores,
        score_type="cosine_similarity",
    )


def create_dual_retriever(
    components: dict[str, Any],
):
    async def structural_retrieve(
        query: str,
    ) -> ChannelRetrieval:
        return await asyncio.to_thread(
            retrieve_structural_context,
            query,
            components["search_engine"],
            components["text_embedder"],
            components["structural_entity_index"],
            components["structural_entity_ids"],
        )

    async def textual_retrieve(
        query: str,
    ) -> ChannelRetrieval:
        return await asyncio.to_thread(
            retrieve_keyword_context,
            query,
            components["text_embedder"],
            components["token_encoder"],
            components["keyword_index"],
            components["words"],
            components["word_chunk_data"],
            components[
                "candidate_units_dict"
            ],
        )

    return build_dual_retriever(
        structural_retrieve,
        textual_retrieve,
    )


# =============================================================================
# 批处理、断点续跑和输出
# =============================================================================


async def run_dataset(
    root_path: str,
    sample_size: int,
) -> None:
    components = (
        initialize_retrieval_components(
            root_path
        )
    )
    token_encoder = components[
        "token_encoder"
    ]
    retrieve_dual = create_dual_retriever(
        components
    )

    config = RFDQDConfig(
        model=os.getenv(
            "RF_DQD_MODEL",
            "gpt-4o-mini",
        ),
        max_hops=int(
            os.getenv(
                "RF_DQD_MAX_HOPS",
                "3",
            )
        ),
        max_subqueries=int(
            os.getenv(
                "RF_DQD_MAX_SUBQUERIES",
                "5",
            )
        ),
        beam_width=int(
            os.getenv(
                "RF_DQD_BEAM_WIDTH",
                "2",
            )
        ),
        candidates_per_step=int(
            os.getenv(
                "RF_DQD_CANDIDATES_PER_STEP",
                "3",
            )
        ),
        novelty_threshold=float(
            os.getenv(
                "RF_DQD_NOVELTY_THRESHOLD",
                "0.15",
            )
        ),
        min_verified_candidate_score=float(
            os.getenv(
                "RF_DQD_MIN_VERIFIED_CANDIDATE_SCORE",
                "0.55",
            )
        ),
        original_context_terminal_threshold=float(
            os.getenv(
                "RF_DQD_ORIGINAL_CONTEXT_TERMINAL_THRESHOLD",
                "0.60",
            )
        ),
        min_candidate_confidence=float(
            os.getenv(
                "RF_DQD_MIN_CANDIDATE_CONFIDENCE",
                "0.55",
            )
        ),
    )
    decomposer = (
        RetrievalFeedbackQueryDecomposer(
            config=config
        )
    )

    qa_path = os.path.join(
        root_path,
        "qa-pairs",
        "qa-pairs.json",
    )
    data = load_json(qa_path)

    valid_pairs: list[
        dict[str, Any]
    ] = []

    for entry in data:
        question = (
            entry.get("question")
            or entry.get(
                "original_question"
            )
            or entry.get("query")
        )
        answer = entry.get("answer")

        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(answer, str)
        ):
            valid_pairs.append(entry)

        if (
            len(valid_pairs)
            >= min(
                sample_size,
                MAX_SAMPLES,
            )
        ):
            break

    output_dir = os.path.join(
        root_path,
        "output",
    )
    dataset_name = os.path.basename(
        os.path.normpath(root_path)
    )

    trace_path = os.path.join(
        output_dir,
        f"{dataset_name}-rf-dqd-trace.json",
    )
    structural_path = os.path.join(
        output_dir,
        (
            f"{dataset_name}-"
            "skeleton-1.0-rf-dqd.json"
        ),
    )
    textual_path = os.path.join(
        output_dir,
        (
            f"{dataset_name}-"
            "keyword-0.0-rf-dqd.json"
        ),
    )

    trace_map: dict[
        str,
        dict[str, Any],
    ] = {}

    if os.path.exists(trace_path):
        try:
            existing = load_json(
                trace_path
            )
            trace_map = {
                str(item.get("id", "")): item
                for item in existing
                if str(item.get("id", ""))
            }
        except Exception as error:
            print(
                "Warning: failed to load "
                f"existing trace: {error}"
            )

    selected_ids = {
        str(item.get("id", ""))
        for item in valid_pairs
    }
    retry_ids = {
        qid
        for qid, item in trace_map.items()
        if item.get("stop_reason") == "error"
        or bool(item.get("error"))
    }
    if retry_ids:
        print(
            "Retrying failed RF-DQD records:",
            len(retry_ids),
        )
    trace_map = {
        qid: item
        for qid, item in trace_map.items()
        if qid in selected_ids
        and qid not in retry_ids
    }

    semaphore = asyncio.Semaphore(
        max(1, RF_DQD_CONCURRENCY)
    )
    save_lock = asyncio.Lock()

    def ordered_trace() -> list[
        dict[str, Any]
    ]:
        return [
            trace_map[str(item["id"])]
            for item in valid_pairs
            if str(item["id"])
            in trace_map
        ]

    def save_all() -> None:
        traces = ordered_trace()

        structural_contexts = [
            {
                "id": item["id"],
                "question": item["question"],
                "need_decomposition": (
                    item.get(
                        "need_decomposition",
                        False,
                    )
                ),
                "retrieval_queries": [
                    trace.get("query", "")
                    for trace in item.get(
                        "traces",
                        [],
                    )
                    if trace.get(
                        "query",
                        "",
                    )
                ],
                "context": item.get(
                    "final_structural_context",
                    "",
                ),
            }
            for item in traces
        ]

        textual_contexts = [
            {
                "id": item["id"],
                "question": item["question"],
                "need_decomposition": (
                    item.get(
                        "need_decomposition",
                        False,
                    )
                ),
                "retrieval_queries": [
                    trace.get("query", "")
                    for trace in item.get(
                        "traces",
                        [],
                    )
                    if trace.get(
                        "query",
                        "",
                    )
                ],
                "context": item.get(
                    "final_textual_context",
                    "",
                ),
            }
            for item in traces
        ]

        save_json_atomic(
            traces,
            trace_path,
        )
        save_json_atomic(
            structural_contexts,
            structural_path,
        )
        save_json_atomic(
            textual_contexts,
            textual_path,
        )

    async def process_one(
        pair: dict[str, Any],
    ) -> None:
        pair_id = str(
            pair.get("id", "")
        )
        question = str(
            pair.get("question")
            or pair.get(
                "original_question"
            )
            or pair.get("query")
            or ""
        ).strip()

        if pair_id in trace_map:
            return

        async with semaphore:
            try:
                result = await decomposer.run(
                    question,
                    retrieve_dual,
                )
                output = {
                    "id": pair.get("id", ""),
                    **result.to_dict(),
                }

                # 最终答案生成时控制单通道上下文预算。
                output[
                    "final_structural_context"
                ] = truncate_to_tokens(
                    output.get(
                        "final_structural_context",
                        "",
                    ),
                    token_encoder,
                    FINAL_CONTEXT_TOKENS,
                )
                output[
                    "final_textual_context"
                ] = truncate_to_tokens(
                    output.get(
                        "final_textual_context",
                        "",
                    ),
                    token_encoder,
                    FINAL_CONTEXT_TOKENS,
                )

            except Exception as error:
                tqdm.write(
                    f"RF-DQD error | "
                    f"id={pair_id} | {error}"
                )
                output = {
                    "id": pair.get("id", ""),
                    "question": question,
                    "need_decomposition": False,
                    "stop_reason": "error",
                    "error": str(error),
                    "plan": [],
                    "traces": [],
                    "final_structural_context": "",
                    "final_textual_context": "",
                }

            async with save_lock:
                trace_map[pair_id] = output
                await asyncio.to_thread(
                    save_all
                )

    pending = [
        pair
        for pair in valid_pairs
        if str(pair.get("id", ""))
        not in trace_map
    ]

    print(
        f"RF-DQD | model={config.model} | "
        f"total={len(valid_pairs)} | "
        f"completed={len(trace_map)} | "
        f"remaining={len(pending)} | "
        f"concurrency={RF_DQD_CONCURRENCY}"
    )

    async def process_with_progress(
        pair: dict[str, Any],
        progress: tqdm,
    ) -> None:
        try:
            await process_one(pair)
        finally:
            progress.update(1)

    with tqdm(
        total=len(valid_pairs),
        initial=len(trace_map),
        desc="RF-DQD",
        unit="question",
        dynamic_ncols=True,
        leave=True,
    ) as progress:
        tasks = [
            asyncio.create_task(
                process_with_progress(
                    pair,
                    progress,
                )
            )
            for pair in pending
        ]
        await asyncio.gather(*tasks)

    await asyncio.to_thread(save_all)

    failed_ids = [
        qid
        for qid, item in trace_map.items()
        if item.get("stop_reason") == "error"
        or bool(item.get("error"))
    ]
    if failed_ids:
        raise RuntimeError(
            f"RF-DQD仍有{len(failed_ids)}条失败记录；"
            "重新运行同一命令会只重试这些记录。"
        )

    print(
        f"Completed {len(trace_map)}/"
        f"{len(valid_pairs)} questions."
    )


def main() -> None:
    if len(sys.argv) not in {2, 3}:
        print(
            "用法：\n"
            "python -m esg_rag.retrieve "
            "[root_path] [sample_size]\n\n"
            "示例：\n"
            "python -m esg_rag.retrieve "
            "ragtest-musique 10"
        )
        sys.exit(1)

    root_path = sys.argv[1].rstrip(
        "/\\"
    )
    sample_size = (
        int(sys.argv[2])
        if len(sys.argv) == 3
        else 10
    )

    if sample_size <= 0:
        raise ValueError(
            "sample_size必须大于0"
        )

    asyncio.run(
        run_dataset(
            root_path,
            min(sample_size, MAX_SAMPLES),
        )
    )


if __name__ == "__main__":
    main()
