import asyncio
import json
import os
import random
import sys

import tiktoken
from tqdm.asyncio import tqdm_asyncio

from esg_rag.search import MyLocalSearch

from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)


# ============================================================
# 基本配置
# ============================================================

LLM_MODEL = os.environ.get(
    "OPENAI_ANSWER_MODEL",
    "gpt-4o-mini"
)

# 每个上下文文件内部同时发送的请求数
ANSWER_CONCURRENCY = 2

# 单条请求最长等待时间
REQUEST_TIMEOUT = 120

# 最多处理的测试样本数
MAX_SAMPLES = 500


# ============================================================
# 读取 JSON
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 生成一个上下文文件对应的答案
# ============================================================

async def process_qa_pairs(
    search_engine,
    golden,
    context_data,
    output_directory,
    context_file_name,
):
    """
    对一个检索上下文文件生成答案。

    功能：
    1. 使用原始问题生成答案；
    2. 支持断点续跑；
    3. 已经完成的 ID 自动跳过；
    4. 每完成一条立即保存；
    5. 超时或报错时写入 Insufficient information。
    """

    # 去掉输入文件名末尾的 .json
    context_file_stem = os.path.splitext(
        context_file_name
    )[0]

    # 例如：
    # ragtest-musique-skeleton-1.0-adaptive.json
    #
    # 输出：
    # answer-gpt4omini-aligned-
    # ragtest-musique-skeleton-1.0-adaptive-raw.json
    output_file = os.path.join(
        output_directory,
        (
            "answer-gpt4omini-aligned-"
            f"{context_file_stem}-raw.json"
        ),
    )

    # 最多处理前500条；敏感性实验可以提供相同前缀的较小固定子集。
    context_data = context_data[:MAX_SAMPLES]
    sample_count = len(context_data)
    if not 1 <= sample_count <= MAX_SAMPLES:
        raise ValueError(
            f"上下文条数必须在1到{MAX_SAMPLES}之间，实际为{sample_count}"
        )
    expected_golden = golden[:sample_count]
    expected_ids = [str(item["id"]) for item in expected_golden]
    context_ids = [str(item.get("id", "")) for item in context_data]

    if len(expected_ids) != sample_count:
        raise ValueError(
            f"原始问题不足{sample_count}条，实际为{len(expected_ids)}条"
        )
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("上下文文件包含重复ID")
    if context_ids != expected_ids:
        raise ValueError(
            f"上下文ID或顺序与原始问题前{sample_count}条不完全一致；"
            "停止答案生成"
        )

    # 原始问题映射
    golden_map = {
        str(item["id"]): item.get("question", "")
        for item in golden
    }

    # 已有结果映射，用于断点续跑
    answer_map = {}

    if os.path.exists(output_file):
        try:
            existing_answers = load_json(output_file)

            for item in existing_answers:
                qid = str(item.get("id", ""))

                if qid:
                    answer_map[qid] = item

            print(
                f"Resume mode: loaded "
                f"{len(answer_map)} existing answers"
            )
            print("Existing file:", output_file)

        except Exception as error:
            print(
                "Warning: failed to load existing "
                f"answer file: {error}"
            )
            answer_map = {}

    valid_ids = {
        str(item["id"])
        for item in context_data
    }

    # 删除可能不属于当前上下文文件的旧数据
    answer_map = {
        qid: item
        for qid, item in answer_map.items()
        if qid in valid_ids
    }

    done_ids = {
        qid
        for qid, item in answer_map.items()
        if item.get("status") in {None, "ok"}
    }

    todo = [
        item
        for item in context_data
        if str(item["id"]) not in done_ids
    ]

    print("=" * 80)
    print("Input context:", context_file_name)
    print("Output answer:", output_file)
    print("Total samples:", len(context_data))
    print("Already done:", len(done_ids))
    print("Remaining:", len(todo))
    print("=" * 80)

    # 当前上下文文件内部并发为 2
    request_semaphore = asyncio.Semaphore(
        ANSWER_CONCURRENCY
    )

    def save_now():
        """
        按原上下文文件顺序保存结果，避免并发完成顺序混乱。
        """

        ordered_answers = []

        for context_item in context_data:
            qid = str(context_item["id"])

            if qid in answer_map:
                ordered_answers.append(
                    answer_map[qid]
                )

        temporary_file = output_file + ".tmp"

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                ordered_answers,
                file,
                ensure_ascii=False,
                indent=4,
            )

        os.replace(
            temporary_file,
            output_file,
        )

    async def generate_one_answer(context_item):
        """
        针对一个问题调用大模型生成答案。
        """

        async with request_semaphore:
            original_id = context_item["id"]
            qid = str(original_id)

            # 这里读取 qa-pairs.json 中的原始问题
            # 不使用 adaptive question 再次生成答案
            question = golden_map.get(qid, "")

            context = context_item.get(
                "context",
                ""
            )

            # 请求稍微错开，降低瞬时限流概率
            await asyncio.sleep(
                0.01 * random.randint(1, 9)
            )

            if not question:
                print(
                    f"Warning: question missing at id={qid}"
                )

                return {
                    "id": original_id,
                    "answer": "",
                    "status": "error",
                    "error_type": "MissingQuestion",
                }

            if not context:
                print(
                    f"Warning: context missing at id={qid}"
                )

                return {
                    "id": original_id,
                    "answer": "",
                    "status": "error",
                    "error_type": "MissingContext",
                }

            try:
                result = await asyncio.wait_for(
                    search_engine.asearch_with_context(
                        question,
                        context,
                    ),
                    timeout=REQUEST_TIMEOUT,
                )

                answer = str(
                    result.response
                ).strip()

                if not answer:
                    return {
                        "id": original_id,
                        "answer": "",
                        "status": "error",
                        "error_type": "EmptyAnswer",
                    }

                return {
                    "id": original_id,
                    "answer": answer,
                    "status": "ok",
                }

            except asyncio.TimeoutError:
                print(
                    f"Timeout at id={qid}"
                )

                return {
                    "id": original_id,
                    "answer": "",
                    "status": "error",
                    "error_type": "TimeoutError",
                }

            except Exception as error:
                print(
                    f"Error at id={qid}: {error}"
                )

                return {
                    "id": original_id,
                    "answer": "",
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }

    # 创建当前未完成样本的任务
    tasks = [
        asyncio.create_task(
            generate_one_answer(item)
        )
        for item in todo
    ]

    # 按任务完成顺序处理并立即保存
    for task in tqdm_asyncio.as_completed(
        tasks,
        desc=f"Processing {context_file_name}",
    ):
        result_item = await task

        qid = str(result_item["id"])

        answer_map[qid] = result_item
        save_now()

    # 再保存一次最终文件
    save_now()

    failed = [
        item
        for item in answer_map.values()
        if item.get("status") == "error"
    ]
    if failed:
        raise RuntimeError(
            f"答案生成仍有{len(failed)}条失败记录；"
            "重新运行同一命令会只重试这些记录。"
        )

    print("=" * 80)
    print("Finished:", output_file)
    print("Total saved:", len(answer_map))
    print("=" * 80)


# ============================================================
# 主函数
# ============================================================

async def main():
    api_key = os.environ.get(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "找不到 OPENAI_API_KEY。\n"
            "请先在 CMD 中运行：\n"
            "set OPENAI_API_KEY=你的Key"
        )

    # --------------------------------------------------------
    # 初始化 GPT-4o-mini
    # --------------------------------------------------------

    llm = ChatOpenAI(
        api_key=api_key,
        model=LLM_MODEL,
        api_type=OpenaiApiType.OpenAI,
        max_retries=5,
    )

    token_encoder = tiktoken.get_encoding(
        "cl100k_base"
    )

    # --------------------------------------------------------
    # GraphRAG 上下文参数
    # --------------------------------------------------------

    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
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
        "max_tokens": 12_000,
    }

    llm_params = {
        "max_tokens": 12_000,
        "temperature": 0.0,
    }

    context_builder = LocalSearchMixedContext(
        community_reports=None,
        text_units=None,
        entities=[],
        relationships=None,
        covariates=None,
        entity_text_embeddings=None,
        embedding_vectorstore_key=(
            EntityVectorStoreKey.ID
        ),
        text_embedder=None,
        token_encoder=token_encoder,
    )

    search_engine = MyLocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=(
            local_context_params
        ),
    )

    # --------------------------------------------------------
    # 加载原始问题
    # --------------------------------------------------------

    golden_file = os.path.join(
        ROOT_PATH,
        "qa-pairs",
        "qa-pairs.json",
    )

    if not os.path.exists(golden_file):
        raise FileNotFoundError(
            f"找不到原始问题文件：{golden_file}"
        )

    golden = load_json(golden_file)

    print("Golden question file:", golden_file)
    print("Golden question count:", len(golden))

    if golden:
        print(
            "First original question:",
            golden[0].get("question", ""),
        )

    # --------------------------------------------------------
    # 自动构造 MuSiQue 自适应上下文文件路径
    # --------------------------------------------------------

    output_directory = os.path.join(
        ROOT_PATH,
        "output",
    )

    dataset_name = os.path.basename(
        os.path.normpath(ROOT_PATH)
    )

    context_suffix = (
        "-rf-dqd"
        if CONTEXT_MODE == "rf-dqd"
        else ""
    )

    context_files = [
        os.path.join(
            output_directory,
            (
                f"{dataset_name}-"
                f"skeleton-1.0{context_suffix}.json"
            ),
        ),
        os.path.join(
            output_directory,
            (
                f"{dataset_name}-"
                f"keyword-0.0{context_suffix}.json"
            ),
        ),
    ]

    # --------------------------------------------------------
    # 检查输入上下文文件
    # --------------------------------------------------------

    missing_files = [
        file_path
        for file_path in context_files
        if not os.path.exists(file_path)
    ]

    if missing_files:
        raise FileNotFoundError(
            "以下RF-DQD上下文文件不存在：\n"
            + "\n".join(missing_files)
        )

    print("#all context files =", len(context_files))

    for path in context_files:
        print("Context file:", path)

    # --------------------------------------------------------
    # 一次只处理一个通道
    #
    # 每个通道内部并发为 2。
    # 避免 Skeleton 和 Keyword 同时运行导致总并发过高。
    # --------------------------------------------------------

    file_semaphore = asyncio.Semaphore(1)

    async def process_one_context_file(
        context_file
    ):
        async with file_semaphore:
            context_data = load_json(
                context_file
            )

            output_directory = os.path.dirname(
                context_file
            )

            context_file_name = os.path.basename(
                context_file
            )

            await process_qa_pairs(
                search_engine=search_engine,
                golden=golden,
                context_data=context_data,
                output_directory=output_directory,
                context_file_name=context_file_name,
            )

    tasks = [
        process_one_context_file(file_path)
        for file_path in context_files
    ]

    await tqdm_asyncio.gather(
        *tasks,
        desc="Processing all context files",
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print(
            "用法：\n"
            "python -m esg_rag.generate "
            "[root_path] [rf-dqd|direct]\n\n"
            "例如：\n"
            "python -m esg_rag.generate "
            "ragtest-musique-topology direct"
        )
        sys.exit(1)

    ROOT_PATH = sys.argv[1]
    CONTEXT_MODE = (
        sys.argv[2]
        if len(sys.argv) == 3
        else "rf-dqd"
    )
    if CONTEXT_MODE not in {"rf-dqd", "direct"}:
        raise ValueError(
            "context mode 必须是 rf-dqd 或 direct"
        )

    asyncio.run(main())
