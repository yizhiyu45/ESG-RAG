from __future__ import annotations

"""
RF-DQD V7.1: Cross-Dataset Retrieval-Feedback Dependency-Aware Query Decomposition

职责边界
--------
RF-DQD 只负责：
1. 从问题中抽取已知实体、目标类型和依赖关系；
2. 生成多个抽象候选计划，并使用与实体名称、数据集名称无关的通用约束选择计划；
3. 先执行原问题双通道检索，并检查原问题上下文对多跳关系链的连续覆盖前缀；
4. 完整覆盖时跳过全部子查询，部分覆盖时从首个缺失关系开始检索；
5. 未覆盖第一跳时，从第一步开始执行完整的依赖式逐跳检索；
6. 仅为后续查询提取必要的中间实体；
7. 对每一步统一执行“锚点实体—目标关系—候选实体—原始证据”验证；
8. 使用检索反馈、新颖性、查询改写和束搜索控制检索路径；
9. 返回彼此独立的结构通道与文本通道累计上下文。

通用性约束
----------
核心模块不得读取数据集名称，不得匹配具体实体名称，不得为单个问题写修复分支。
更换数据集时，只允许替换数据读取适配器、索引路径和评测脚本。

RF-DQD 不负责：
1. 生成原问题的最终答案；
2. 判断最终答案是否正确；
3. 将两个通道合并成一个最终答案。

特别说明
--------
对于查询计划中的“终端步骤”，本模块只执行检索并累计上下文，
不会抽取该步骤的最终实体。这样可以避免提前得到原问题答案，
并保持后续双通道答案验证模块的独立性。
"""

import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from openai import AsyncOpenAI


# =============================================================================
# 配置与数据结构
# =============================================================================


@dataclass
class RFDQDConfig:
    model: str = field(
        default_factory=lambda: os.getenv(
            "RF_DQD_MODEL",
            os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini"),
        )
    )

    max_hops: int = 4
    max_subqueries: int = 6
    beam_width: int = 2
    candidates_per_step: int = 3

    # V7：跨数据集通用计划生成与选择。
    candidate_plan_count: int = 3
    max_plan_generation_attempts: int = 2
    min_static_plan_score: float = 0.72
    min_semantic_plan_score: float = 0.65
    min_combined_plan_score: float = 0.70
    plan_static_weight: float = 0.45
    plan_semantic_weight: float = 0.55

    novelty_threshold: float = 0.15

    # V5.1：候选抽取只是“召回阶段”，阈值应低于最终关系验证阈值。
    # 这样可让潜在正确实体进入关系验证器，而不是在验证前被提前丢弃。
    min_raw_candidate_confidence: float = 0.30

    # 保留旧字段以兼容已有接入代码；V5.1不再用它过滤原始候选。
    min_candidate_confidence: float = 0.55
    agreement_bonus: float = 0.10

    # V5：计划验证、累计上下文抽取、多候选查询重试与关系约束验证。
    enable_plan_validation: bool = True
    max_plan_repair_attempts: int = 2
    max_alternative_queries: int = 3
    max_retry_queries_per_step: int = 3

    # 中间实体先抽取、再做关系约束验证。
    enable_relation_verification: bool = True
    min_verified_candidate_score: float = 0.55
    require_relation_support: bool = True
    require_type_match: bool = True
    verification_relation_weight: float = 0.45
    verification_type_weight: float = 0.20
    verification_evidence_weight: float = 0.20
    verification_cross_weight: float = 0.15

    # 送入“中间实体抽取/验证模型”的单通道最大字符数。
    # 代码会在原问题累计上下文与当前步骤上下文之间平衡分配。
    max_context_chars: int = 20_000

    # V6：原问题上下文覆盖检查。
    # fully_covered：不执行任何子查询；
    # partially_covered：利用已验证中间实体，从首个缺失步骤继续；
    # not_covered：从第一步开始完整逐跳检索。
    enable_original_context_coverage: bool = True
    original_context_terminal_threshold: float = 0.60
    max_original_coverage_paths: int = 2

    request_timeout: int = 120
    max_json_retries: int = 2
    max_retries: int = 5


@dataclass
class ChannelRetrieval:
    """
    单通道检索结果。

    item_ids:
        推荐传真实文本块ID、实体ID或关系ID。
        若为空，则自动根据上下文非空行生成哈希ID。

    scores:
        与 item_ids 一一对应的原始检索分数。不要在这里截断或用固定值替代。

    score_type:
        原始分数的含义，例如 cosine_similarity、inner_product、distance。
    """

    context: str = ""
    item_ids: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    score_type: str = ""


@dataclass
class DualRetrieval:
    structural: ChannelRetrieval = field(default_factory=ChannelRetrieval)
    textual: ChannelRetrieval = field(default_factory=ChannelRetrieval)


@dataclass
class QuestionStructure:
    known_entities: list[str]
    target_type: str
    relation_chain: list[str]
    dependency_depth: int
    branch_count: int
    query_type: str
    known_entity_types: dict[str, str] = field(default_factory=dict)


@dataclass
class QueryStep:
    step_id: str
    query_template: str
    relation: str
    depends_on: list[str]
    target_type: str = ""
    source_type: str = ""
    source_ref: str = ""


@dataclass
class Candidate:
    entity: str
    confidence: float
    sources: list[str]
    evidence: list[str]

    # V5关系约束验证字段。
    raw_confidence: float = 0.0
    anchor_entity: str = ""
    anchor_present: bool = False
    candidate_present: bool = False
    relation_supported: bool = False
    type_match: bool = False
    relation_confidence: float = 0.0
    evidence_confidence: float = 0.0
    verification_score: float = 0.0
    verification_evidence: str = ""
    verification_relation_phrase: str = ""
    verification_reason: str = ""

    # V7.1：关系蕴含审计。用于区分“共同出现/相关联”与
    # “证据能够直接推出目标三元关系”。
    relation_entailment: bool = False
    counterexample_possible: bool = True

    # 两个通道必须分别验证目标关系，不能用联合上下文的结论替代。
    structural_relation_supported: bool = False
    textual_relation_supported: bool = False
    structural_verification_evidence: str = ""
    textual_verification_evidence: str = ""

    hard_failure_reasons: list[str] = field(default_factory=list)
    verified: bool = False


@dataclass
class CandidateBatch:
    accepted: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)


@dataclass
class StepTrace:
    step_id: str
    query: str
    rewritten_query: str
    relation: str
    depends_on: list[str]
    is_terminal: bool
    structural_novelty: float
    textual_novelty: float
    overall_novelty: float
    selected_entity: str
    selected_score: float
    candidates: list[Candidate]

    # V5诊断字段：保留实际查询、已接受/被拒绝候选及验证原因。
    attempted_queries: list[str] = field(default_factory=list)
    retry_reason: str = ""
    extraction_context_mode: str = "accumulated"
    rejected_candidates: list[Candidate] = field(default_factory=list)
    anchor_entities: list[str] = field(default_factory=list)
    verification_enabled: bool = True

    # 当前最终选中尝试的逐条原始检索记录。
    structural_retrieval_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    textual_retrieval_records: list[dict[str, Any]] = field(
        default_factory=list
    )

    # 由关系验证器分别针对两个通道给出的步骤支持结论。
    structural_step_supported: bool = False
    textual_step_supported: bool = False


@dataclass
class PathState:
    answers: dict[str, str]
    structural_context: str
    textual_context: str
    structural_ids: set[str]
    textual_ids: set[str]
    scores: list[float]
    traces: list[StepTrace]

    @property
    def score(self) -> float:
        return (
            sum(self.scores) / len(self.scores)
            if self.scores
            else 0.0
        )


@dataclass
class RFDQDResult:
    question: str
    need_decomposition: bool
    question_structure: QuestionStructure
    plan: list[QueryStep]
    executed_subqueries: int
    stop_reason: str
    selected_path_score: float
    intermediate_answers: dict[str, str]
    traces: list[StepTrace]
    final_structural_context: str
    final_textual_context: str
    structural_evidence_ids: list[str]
    textual_evidence_ids: list[str]
    alternative_paths: list[dict[str, Any]]
    plan_validation: dict[str, Any] = field(default_factory=dict)
    original_context_coverage: dict[str, Any] = field(default_factory=dict)
    initial_structural_retrieval_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    initial_textual_retrieval_records: list[dict[str, Any]] = field(
        default_factory=list
    )
    structural_supported_step_ids: list[str] = field(default_factory=list)
    textual_supported_step_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DualRetriever(Protocol):
    async def __call__(self, query: str) -> DualRetrieval:
        ...


T = TypeVar("T")


# =============================================================================
# 通用工具
# =============================================================================


def clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(
        r"[^a-z0-9\u4e00-\u9fff ]",
        " ",
        text,
    )
    return " ".join(text.split())


def entity_similarity(a: str, b: str) -> float:
    a_norm = normalize(a)
    b_norm = normalize(b)

    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.90

    return SequenceMatcher(
        None,
        a_norm,
        b_norm,
    ).ratio()


def truncate(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit]


def merge_context(old: str, new: str) -> str:
    """
    只在同一通道内部按行去重。
    不会跨结构通道和文本通道合并。
    """
    seen: set[str] = set()
    merged: list[str] = []

    for source in (old, new):
        for line in str(source or "").splitlines():
            line = line.strip()
            key = re.sub(
                r"\s+",
                " ",
                line,
            ).lower()

            if not line or key in seen:
                continue

            seen.add(key)
            merged.append(line)

    return "\n".join(merged)



def smart_truncate(text: str, limit: int) -> str:
    """同时保留文本开头和结尾，避免累计上下文只截取最早证据。"""
    text = str(text or "")
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text

    head_size = int(limit * 0.62)
    tail_size = max(0, limit - head_size)
    return (
        text[:head_size]
        + "\n...[context truncated]...\n"
        + text[-tail_size:]
    )


def compose_extraction_context(
    accumulated_context: str,
    current_step_context: str,
    limit: int,
) -> str:
    """
    V5累计上下文：
    - 一半左右预算给原问题及前序步骤上下文；
    - 另一半给当前步骤及本步骤重试上下文。

    这样既能保留原问题和前序步骤中的关键实体，
    也不会让当前子查询证据因截断而消失。
    """
    accumulated_context = str(accumulated_context or "")
    current_step_context = str(current_step_context or "")

    if not accumulated_context:
        return smart_truncate(current_step_context, limit)
    if not current_step_context:
        return smart_truncate(accumulated_context, limit)

    accumulated_limit = int(limit * 0.50)
    current_limit = max(0, limit - accumulated_limit)

    return (
        "===== ACCUMULATED EVIDENCE FROM ORIGINAL/PREVIOUS QUERIES =====\n"
        + smart_truncate(accumulated_context, accumulated_limit)
        + "\n\n===== CURRENT STEP AND RETRY EVIDENCE =====\n"
        + smart_truncate(current_step_context, current_limit)
    )


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fallback_ids(context: str) -> set[str]:
    ids: set[str] = set()

    for line in str(context or "").splitlines():
        line = re.sub(
            r"\s+",
            " ",
            line.strip(),
        ).lower()

        if not line:
            continue

        digest = hashlib.sha1(
            line.encode("utf-8")
        ).hexdigest()
        ids.add("line:" + digest)

    return ids


def channel_ids(result: ChannelRetrieval) -> set[str]:
    ids = {
        str(item)
        for item in result.item_ids
        if str(item)
    }
    return ids or fallback_ids(result.context)


def novelty(
    new_ids: set[str],
    seen_ids: set[str],
) -> float:
    if not new_ids:
        return 0.0
    return len(new_ids - seen_ids) / len(new_ids)


def retrieval_score(
    result: ChannelRetrieval,
) -> float:
    score_type = str(result.score_type or "").lower()
    scores: list[float] = []

    for raw_score in result.scores:
        try:
            raw = float(raw_score)
        except (TypeError, ValueError):
            continue

        if score_type in {
            "cosine_similarity",
            "cosine",
            "inner_product_normalized",
        }:
            normalized_score = clamp01((raw + 1.0) / 2.0)
        elif score_type in {
            "distance",
            "l2_distance",
            "euclidean_distance",
        }:
            normalized_score = 1.0 / (1.0 + max(0.0, raw))
        else:
            normalized_score = clamp01(raw)

        scores.append(normalized_score)

    if scores:
        return sum(scores) / len(scores)

    # 缺少真实分数时不再给非空上下文固定0.5。
    return 0.0


def channel_retrieval_records(
    result: ChannelRetrieval,
    channel: str,
    query: str,
    step_id: str,
) -> list[dict[str, Any]]:
    """将单次检索结果转成可审计、可用于最终评分的逐条记录。"""
    records: list[dict[str, Any]] = []

    for rank, item_id in enumerate(result.item_ids, start=1):
        if rank > len(result.scores):
            break
        try:
            raw_score = float(result.scores[rank - 1])
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "step_id": str(step_id),
                "query": str(query),
                "channel": str(channel),
                "item_id": str(item_id),
                "rank": rank,
                "raw_score": raw_score,
                "score_type": str(result.score_type or "unknown"),
            }
        )

    return records


def coverage_channel_supported_step_ids(
    coverage_info: dict[str, Any],
    channel: str,
) -> set[str]:
    field_name = f"{channel}_supported"
    return {
        str(step.get("step_id", ""))
        for step in coverage_info.get("steps", [])
        if str(step.get("step_id", ""))
        and bool(step.get(field_name, False))
    }


def attempts_channel_retrieval_records(
    attempts: list[dict[str, Any]],
    channel: str,
    step_id: str,
) -> list[dict[str, Any]]:
    """保存该步骤全部原始查询与改写查询的检索排名。"""
    records: list[dict[str, Any]] = []
    for attempt in attempts:
        retrieval = attempt.get("retrieval")
        if not isinstance(retrieval, DualRetrieval):
            continue
        result = (
            retrieval.structural
            if channel == "structural"
            else retrieval.textual
        )
        records.extend(
            channel_retrieval_records(
                result,
                channel=channel,
                query=str(attempt.get("query", "")),
                step_id=step_id,
            )
        )
    return records


async def maybe_await(
    value: T | Awaitable[T],
) -> T:
    return (
        await value
        if inspect.isawaitable(value)
        else value
    )


def parse_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    candidates: list[str] = []

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        candidates.append(fenced.group(1).strip())

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    candidates.append(raw)
    errors: list[Exception] = []

    for candidate in dict.fromkeys(candidates):
        variants = [candidate]

        # 修复非法反斜杠，例如 \D、\m 等非法JSON转义。
        repaired = re.sub(
            r'\\(?!["\\/bfnrtu])',
            r"\\\\",
            candidate,
        )

        # 删除对象或数组结束前的多余逗号。
        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            repaired,
        )

        # 补充相邻JSON字段之间可能缺失的逗号。
        repaired = re.sub(
            r'([}\]"])\s*\n(\s*"[^"\n]+"\s*:)',
            r"\1,\n\2",
            repaired,
        )

        # 补充数组中相邻对象之间可能缺失的逗号。
        repaired = re.sub(
            r"([}\]])\s*\n(\s*[\{\[])",
            r"\1,\n\2",
            repaired,
        )

        if repaired != candidate:
            variants.append(repaired)

        for variant in variants:
            try:
                data = json.loads(variant)
                if not isinstance(data, dict):
                    raise ValueError("模型返回的JSON不是对象")
                return data
            except Exception as exc:
                errors.append(exc)

    detail = str(errors[-1]) if errors else "没有找到JSON对象"
    raise ValueError(
        f"无法解析JSON：{detail}；原始响应：{raw[:300]}"
    )

# =============================================================================
# RF-DQD 核心实现
# =============================================================================


class RetrievalFeedbackQueryDecomposer:
    def __init__(
        self,
        config: RFDQDConfig | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = config or RFDQDConfig()

        if client is not None:
            self.client = client
            return

        api_key = (
            os.getenv("OPENAI_API_KEY")
            or os.getenv("GRAPHRAG_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "缺少 OPENAI_API_KEY 或 GRAPHRAG_API_KEY"
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL"),
            max_retries=self.config.max_retries,
        )

    async def _json_call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> dict[str, Any]:
        error: Exception | None = None
        prompt = user_prompt

        for attempt in range(
            self.config.max_json_retries + 1
        ):
            raw_response = ""

            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0.0,
                        max_tokens=(
                            max_tokens
                            if attempt == 0
                            else min(max_tokens * 2, 8192)
                        ),
                        response_format={"type": "json_object"},
                    ),
                    timeout=self.config.request_timeout,
                )

                raw_response = (
                    response.choices[0].message.content
                    or ""
                )

            except Exception as exc:
                error = exc
                prompt = (
                    user_prompt
                    + "\nReturn exactly one valid JSON object. "
                    + "Do not use Markdown."
                )

            else:
                try:
                    return parse_json(raw_response)

                except Exception as exc:
                    error = exc

                    # 不再重复原任务，而是修复刚才生成的错误JSON。
                    prompt = (
                        "The following response should be one JSON "
                        "object, but its JSON syntax is invalid. "
                        "Repair JSON syntax only. Preserve all keys "
                        "and values. Correct missing commas, invalid "
                        "backslash escapes, quotation marks and "
                        "brackets. Return exactly one valid JSON "
                        "object without Markdown.\n\n"
                        "INVALID JSON RESPONSE:\n"
                        + raw_response
                    )

            if attempt < self.config.max_json_retries:
                await asyncio.sleep(
                    1.5 * (attempt + 1)
                )

        raise RuntimeError(
            f"模型JSON调用失败：{error}"
        ) from error

    # -------------------------------------------------------------------------
    # 问题结构解析与查询计划
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_structure(data: dict[str, Any]) -> QuestionStructure:
        known_entities: list[str] = []
        known_entity_types: dict[str, str] = {}

        for item in data.get("known_entities", []):
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                entity_type = str(item.get("type", "")).strip()
            else:
                name = str(item).strip()
                entity_type = ""

            if not name:
                continue
            known_entities.append(name)
            if entity_type:
                known_entity_types[name] = entity_type

        extra_types = data.get("known_entity_types", {})
        if isinstance(extra_types, dict):
            for name, entity_type in extra_types.items():
                name = str(name).strip()
                entity_type = str(entity_type).strip()
                if name and entity_type:
                    known_entity_types[name] = entity_type
                    if name not in known_entities:
                        known_entities.append(name)

        return QuestionStructure(
            known_entities=list(dict.fromkeys(known_entities)),
            target_type=str(data.get("target_type", "other")).strip(),
            relation_chain=[
                str(item).strip()
                for item in data.get("relation_chain", [])
                if str(item).strip()
            ],
            dependency_depth=max(
                1,
                safe_int(
                    data.get(
                        "dependency_depth",
                        data.get("estimated_hops", 1),
                    ),
                    1,
                ),
            ),
            branch_count=max(
                1,
                safe_int(data.get("branch_count", 1), 1),
            ),
            query_type=str(
                data.get("query_type", "single_hop")
            ).strip().lower(),
            known_entity_types=known_entity_types,
        )

    def _parse_steps(
        self,
        raw_steps: list[dict[str, Any]],
        known_entities: list[str],
    ) -> list[QueryStep]:
        """
        解析并规范化计划步骤。

        V7.1通用修复：
        - query_template 中不允许遗留抽象占位符 {{source_ref}}；
        - 第一步若错误地自引用 {{s1}}，使用其真实 source_ref 修复；
        - 依赖只从真实的步骤占位符中推导；
        - 不包含任何数据集、实体或固定问题规则。
        """
        steps: list[QueryStep] = []
        used_ids: set[str] = set()

        for index, item in enumerate(
            raw_steps[: self.config.max_subqueries],
            start=1,
        ):
            if not isinstance(item, dict):
                continue

            step_id = str(item.get("step_id", f"s{index}")).strip()
            if not step_id or step_id in used_ids:
                step_id = f"s{index}"
            used_ids.add(step_id)

            declared_dependencies = [
                str(dependency).strip()
                for dependency in item.get("depends_on", [])
                if str(dependency).strip()
            ]

            source_ref = str(item.get("source_ref", "")).strip()
            source_ref = self.resolve_known_entity_placeholders(
                source_ref,
                known_entities,
            )

            # source_ref 是模式字段，不是可执行占位符。
            if normalize(source_ref) in {"source ref", "source_ref"}:
                source_ref = ""

            if not source_ref:
                if declared_dependencies:
                    source_ref = (
                        "{{" + declared_dependencies[0] + "}}"
                    )
                elif known_entities:
                    source_ref = known_entities[0]

            template = str(item.get("query_template", "")).strip()
            template = self.resolve_known_entity_placeholders(
                template,
                known_entities,
            )
            if not template:
                continue

            # 将模型偶尔生成的抽象 {{source_ref}} 展开成真实来源。
            template = template.replace(
                "{{source_ref}}",
                source_ref,
            )
            template = template.replace(
                "{source_ref}",
                source_ref,
            )

            self_placeholder_a = "{{" + step_id + "}}"
            self_placeholder_b = "{" + step_id + "}"

            # 当前步骤的输出在执行前不可用。若模型在第一步或任一步中
            # 错误使用自身占位符，则仅在已有合法来源时替换为 source_ref。
            source_is_self = source_ref in {
                self_placeholder_a,
                self_placeholder_b,
            }
            if source_is_self:
                previous_dependencies = [
                    dep
                    for dep in declared_dependencies
                    if dep != step_id
                ]
                if previous_dependencies:
                    source_ref = (
                        "{{" + previous_dependencies[0] + "}}"
                    )
                elif known_entities:
                    source_ref = known_entities[0]

            if source_ref not in {
                self_placeholder_a,
                self_placeholder_b,
                "",
            }:
                template = template.replace(
                    self_placeholder_a,
                    source_ref,
                )
                template = template.replace(
                    self_placeholder_b,
                    source_ref,
                )

            placeholder_dependencies = re.findall(
                r"\{\{?(s\d+)\}?\}",
                template + " " + source_ref,
            )

            dependencies = list(
                dict.fromkeys(
                    [
                        dependency
                        for dependency in [
                            *declared_dependencies,
                            *placeholder_dependencies,
                        ]
                        if dependency and dependency != step_id
                    ]
                )
            )

            # 第一步不应依赖任何尚未产生的步骤。
            if index == 1:
                dependencies = []
                if (
                    source_ref.startswith("{")
                    and known_entities
                ):
                    source_ref = known_entities[0]

            steps.append(
                QueryStep(
                    step_id=step_id,
                    query_template=template,
                    relation=str(item.get("relation", "")).strip(),
                    depends_on=dependencies,
                    target_type=str(item.get("target_type", "")).strip(),
                    source_type=str(item.get("source_type", "")).strip(),
                    source_ref=source_ref,
                )
            )

        return steps

    @staticmethod
    def resolve_known_entity_placeholders(
        template: str,
        known_entities: list[str],
    ) -> str:
        """兼容并修复模型偶尔生成的 known_entities 下标占位符。"""
        pattern = re.compile(
            r"\{\{?known_entities\[(\d+)\]\}?\}"
        )

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if 0 <= index < len(known_entities):
                return known_entities[index]
            return match.group(0)

        return pattern.sub(replace, template)

    @staticmethod
    def _plan_as_dict(
        structure: QuestionStructure,
        steps: list[QueryStep],
    ) -> dict[str, Any]:
        return {
            "known_entities": [
                {
                    "name": entity,
                    "type": structure.known_entity_types.get(entity, ""),
                }
                for entity in structure.known_entities
            ],
            "target_type": structure.target_type,
            "relation_chain": structure.relation_chain,
            "dependency_depth": structure.dependency_depth,
            "branch_count": structure.branch_count,
            "query_type": structure.query_type,
            "steps": [asdict(step) for step in steps],
        }

    @staticmethod
    def _plan_signature(
        structure: QuestionStructure,
        steps: list[QueryStep],
    ) -> tuple[Any, ...]:
        return (
            normalize(structure.target_type),
            tuple(normalize(item) for item in structure.relation_chain),
            int(structure.dependency_depth),
            int(structure.branch_count),
            normalize(structure.query_type),
            tuple(
                (
                    step.step_id,
                    normalize(step.query_template),
                    normalize(step.relation),
                    tuple(step.depends_on),
                    normalize(step.target_type),
                    normalize(step.source_type),
                    normalize(step.source_ref),
                )
                for step in steps
            ),
        )

    @staticmethod
    def _canonical_type(entity_type: str) -> str:
        value = normalize(entity_type)
        if not value:
            return "other"

        aliases: dict[str, str] = {
            "human": "person",
            "people": "person",
            "individual": "person",
            "author": "person",
            "writer": "person",
            "artist": "person",
            "performer": "person",
            "company": "organization",
            "corporation": "organization",
            "institution": "organization",
            "association": "organization",
            "group": "organization",
            "party": "organization",
            "band": "organization",
            "team": "organization",
            "city": "location",
            "town": "location",
            "village": "location",
            "county": "location",
            "district": "location",
            "province": "location",
            "state": "location",
            "country": "location",
            "borough": "location",
            "region": "location",
            "place": "location",
            "work": "creative work",
            "book": "creative work",
            "film": "creative work",
            "movie": "creative work",
            "song": "creative work",
            "album": "creative work",
            "play": "creative work",
            "publication": "creative work",
            "school": "facility",
            "building": "facility",
            "venue": "facility",
            "stadium": "facility",
            "airport": "facility",
            "date": "date",
            "year": "date",
            "time": "date",
            "number": "number",
            "quantity": "number",
            "description": "description",
            "purpose": "description",
            "goal": "description",
            "entity": "entity",
            "other": "other",
        }

        if value in aliases:
            return aliases[value]

        for token, canonical in aliases.items():
            if token in value:
                return canonical

        return value

    @classmethod
    def _types_compatible(
        cls,
        produced_type: str,
        required_type: str,
    ) -> bool:
        produced = cls._canonical_type(produced_type)
        required = cls._canonical_type(required_type)

        wildcard = {"", "other", "entity"}
        if produced in wildcard or required in wildcard:
            return True
        if produced == required:
            return True

        # 地理子类型均可作为通用地点输入，组织子类型同理。
        broad_groups = [
            {"person"},
            {"organization"},
            {"location", "facility"},
            {"creative work"},
            {"date"},
            {"number"},
            {"description"},
        ]
        return any(
            produced in group and required in group
            for group in broad_groups
        )

    def _static_plan_diagnostics(
        self,
        question: str,
        structure: QuestionStructure,
        steps: list[QueryStep],
    ) -> dict[str, Any]:
        """
        与数据集和具体实体无关的计划检查。

        只检查依赖、占位符、类型连续性、终点类型、循环和重复，
        不包含任何具体问题模板或实体名称分支。
        """
        hard_issues: list[str] = []
        soft_issues: list[str] = []
        seen: dict[str, QueryStep] = {}
        normalized_queries: set[str] = set()

        if not steps:
            if self.need_decomposition(structure):
                hard_issues.append("multi-hop structure has no executable steps")
            return {
                "valid": not hard_issues,
                "hard_issues": hard_issues,
                "soft_issues": soft_issues,
                "dependency_score": 0.0 if hard_issues else 1.0,
                "type_score": 0.0 if hard_issues else 1.0,
                "target_score": 0.0 if hard_issues else 1.0,
                "conciseness_score": 1.0,
                "static_score": 0.0 if hard_issues else 1.0,
            }

        if len(steps) > self.config.max_subqueries:
            hard_issues.append("plan exceeds max_subqueries")
        if len(steps) > max(self.config.max_hops, structure.branch_count):
            soft_issues.append("plan may contain unnecessary factual steps")

        dependency_checks = 0
        dependency_passes = 0
        type_checks = 0
        type_passes = 0

        for index, step in enumerate(steps):
            if step.step_id in seen:
                hard_issues.append(f"duplicate step id: {step.step_id}")

            query_key = normalize(step.query_template)
            if not query_key:
                hard_issues.append(f"{step.step_id} has an empty query")
            elif query_key in normalized_queries:
                hard_issues.append(f"duplicate query semantics at {step.step_id}")
            normalized_queries.add(query_key)

            if not step.relation.strip():
                hard_issues.append(f"{step.step_id} has no relation")
            if not step.target_type.strip():
                hard_issues.append(f"{step.step_id} has no output type")

            if normalize(step.query_template) == normalize(question):
                soft_issues.append(
                    f"{step.step_id} copies the original question instead of isolating one relation"
                )

            for dependency in step.depends_on:
                dependency_checks += 1
                previous = seen.get(dependency)
                if previous is None:
                    hard_issues.append(
                        f"{step.step_id} depends on unavailable step {dependency}"
                    )
                    continue

                placeholder_a = "{" + dependency + "}"
                placeholder_b = "{{" + dependency + "}}"
                source_uses_dependency = (
                    placeholder_a in step.query_template
                    or placeholder_b in step.query_template
                    or placeholder_a in step.source_ref
                    or placeholder_b in step.source_ref
                )
                if not source_uses_dependency:
                    hard_issues.append(
                        f"{step.step_id} declares {dependency} but does not use its output"
                    )
                else:
                    dependency_passes += 1

                type_checks += 1
                if self._types_compatible(
                    previous.target_type,
                    step.source_type,
                ):
                    type_passes += 1
                else:
                    hard_issues.append(
                        f"type discontinuity: {dependency} outputs {previous.target_type} "
                        f"but {step.step_id} requires {step.source_type}"
                    )

            unresolved = re.findall(
                r"\{\{?([^{}]+)\}?\}",
                step.query_template + " " + step.source_ref,
            )
            for placeholder in unresolved:
                if placeholder not in step.depends_on:
                    hard_issues.append(
                        f"{step.step_id} has undeclared placeholder {placeholder}"
                    )

            if index == 0 and step.depends_on:
                hard_issues.append("first step cannot depend on a prior step")

            if index == 0 and structure.known_entities:
                first_text = normalize(
                    step.query_template + " " + step.source_ref
                )
                if not any(
                    normalize(entity) in first_text
                    for entity in structure.known_entities
                ):
                    soft_issues.append(
                        "first step does not explicitly preserve a known entity anchor"
                    )

            seen[step.step_id] = step

        target_compatible = self._types_compatible(
            steps[-1].target_type,
            structure.target_type,
        )
        if not target_compatible:
            hard_issues.append(
                "final step output type does not match the requested answer type"
            )

        dependency_score = (
            dependency_passes / dependency_checks
            if dependency_checks
            else 1.0
        )
        type_score = (
            type_passes / type_checks
            if type_checks
            else 1.0
        )
        target_score = 1.0 if target_compatible else 0.0
        conciseness_score = clamp01(
            1.0 - 0.12 * len(soft_issues)
        )

        static_score = clamp01(
            0.35 * dependency_score
            + 0.30 * type_score
            + 0.25 * target_score
            + 0.10 * conciseness_score
        )
        if hard_issues:
            static_score *= 0.35

        return {
            "valid": not hard_issues,
            "hard_issues": list(dict.fromkeys(hard_issues)),
            "soft_issues": list(dict.fromkeys(soft_issues)),
            "dependency_score": dependency_score,
            "type_score": type_score,
            "target_score": target_score,
            "conciseness_score": conciseness_score,
            "static_score": static_score,
        }

    async def _semantic_assess_plans(
        self,
        question: str,
        records: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if not records:
            return {}

        payload = [
            {
                "plan_id": record["plan_id"],
                "plan": self._plan_as_dict(
                    record["structure"],
                    record["steps"],
                ),
                "static_diagnostics": record["static"],
            }
            for record in records
        ]

        data = await self._json_call(
            (
                "You judge abstract retrieval plans. Do not answer the question. "
                "Do not prefer any dataset, entity name, or question template. "
                "Evaluate only semantic direction, dependency continuity, target alignment, "
                "and whether each step retrieves one missing fact. Return JSON only."
            ),
            f"""
Question:
{question}

Candidate plans:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Return:
{{
  "evaluations": [
    {{
      "plan_id": "p1",
      "valid": true,
      "direction_score": 0.0,
      "dependency_score": 0.0,
      "target_alignment_score": 0.0,
      "step_atomicity_score": 0.0,
      "direct_relation_score": 0.0,
      "composition_entailment_score": 0.0,
      "non_redundancy_score": 0.0,
      "issues": ["brief issue"]
    }}
  ]
}}

Rules:
- Judge whether the relation chain starts from information explicitly known in the question and moves toward the requested answer.
- A dependent step must consume the prior step's output, not ask the original question again.
- The output type of one step must be a valid input type for the next step.
- Every step relation must be a direct relation between that step's source and output.
- Simulate the variable path step by step. The composition of all direct relations must logically entail the requested answer.
- Ask the counterfactual question: could every listed step fact be true while the requested answer still does not follow? If yes, valid=false.
- The last step must retrieve the requested answer type and relation, not a merely related fact or a relation that applies to a different source.
- Reject reversed, circular, transitive-but-not-equivalent, merely correlated, or unnecessarily indirect plans.
- Branches must represent real alternatives or parallel facts, not extra sequential hops.
- Never use world knowledge to answer the question.
- Return one evaluation for every candidate plan.
""",
            1600,
        )

        result: dict[str, dict[str, Any]] = {}
        for item in data.get("evaluations", []):
            if not isinstance(item, dict):
                continue
            plan_id = str(item.get("plan_id", "")).strip()
            if not plan_id:
                continue
            scores = [
                clamp01(item.get("direction_score")),
                clamp01(item.get("dependency_score")),
                clamp01(item.get("target_alignment_score")),
                clamp01(item.get("step_atomicity_score")),
                clamp01(item.get("direct_relation_score")),
                clamp01(item.get("composition_entailment_score")),
                clamp01(item.get("non_redundancy_score")),
            ]
            result[plan_id] = {
                "valid": bool(item.get("valid", False)),
                "semantic_score": sum(scores) / len(scores),
                "direction_score": scores[0],
                "dependency_score": scores[1],
                "target_alignment_score": scores[2],
                "step_atomicity_score": scores[3],
                "direct_relation_score": scores[4],
                "composition_entailment_score": scores[5],
                "non_redundancy_score": scores[6],
                "issues": [
                    str(issue).strip()
                    for issue in item.get("issues", [])
                    if str(issue).strip()
                ],
            }
        return result

    async def _validate_decomposition_decision(
        self,
        question: str,
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """
        对“单跳/无需分解”结论执行一次通用语义复核。

        该检查只判断问题是否包含未解析的关系嵌套，不回答问题，
        也不依赖任何数据集或具体实体。
        """
        data = await self._json_call(
            (
                "Judge whether a question requires resolving an unknown "
                "intermediate entity before retrieving the final answer. "
                "Do not answer the question. Return JSON only."
            ),
            f"""
Question:
{question}

Initial analysis:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Return:
{{
  "requires_decomposition": true,
  "estimated_hops": 2,
  "reason": "brief structural reason"
}}

Rules:
- requires_decomposition=true when the requested answer is a property or relation of an entity that is itself described through another unresolved relation.
- A relational description is not a named entity merely because it is a noun phrase.
- Count direct missing relation transitions from explicit known entities to the requested answer.
- A genuinely direct one-relation lookup is single-hop.
- Do not use world knowledge and do not answer the question.
""",
            700,
        )
        return {
            "requires_decomposition": bool(
                data.get("requires_decomposition", False)
            ),
            "estimated_hops": max(
                1,
                safe_int(data.get("estimated_hops", 1), 1),
            ),
            "reason": str(data.get("reason", "")).strip(),
        }

    async def _generate_candidate_plan_data(
        self,
        question: str,
        feedback: list[str] | None = None,
    ) -> dict[str, Any]:
        feedback = feedback or []
        return await self._json_call(
            (
                "You create cross-dataset dependency-aware retrieval plans. "
                "Do not answer the question. Use no dataset-specific rules and no entity-specific shortcuts. "
                "Return JSON only."
            ),
            f"""
Question:
{question}

Previous generic validation feedback:
{json.dumps(feedback, ensure_ascii=False)}

Return:
{{
  "question_analysis": {{
    "known_entities": [{{"name": "literal entity", "type": "entity type"}}],
    "target_type": "requested answer type",
    "requires_decomposition": true,
    "estimated_hops": 2,
    "branch_count": 1,
    "query_type": "single_hop|chain|comparison|intersection|composition"
  }},
  "candidate_plans": [
    {{
      "plan_id": "p1",
      "relation_chain": ["relation_1", "relation_2"],
      "dependency_depth": 2,
      "branch_count": 1,
      "query_type": "chain",
      "steps": [
        {{
          "step_id": "s1",
          "source_ref": "literal known entity",
          "source_type": "input entity type",
          "query_template": "one atomic factual retrieval query",
          "relation": "directed relation from source to output",
          "depends_on": [],
          "target_type": "output entity type"
        }},
        {{
          "step_id": "s2",
          "source_ref": "{{{{s1}}}}",
          "source_type": "type produced by s1",
          "query_template": "one atomic query containing {{{{s1}}}}",
          "relation": "next directed relation",
          "depends_on": ["s1"],
          "target_type": "requested answer type"
        }}
      ]
    }}
  ]
}}

Rules:
- Copy every explicit named entity from the question literally; never invent a known entity.
- A relational description is not a named entity. Phrases such as “the author/performer/owner/spouse/employer/location/group of X” introduce an unresolved intermediate variable unless the whole phrase is explicitly a proper name.
- If the requested answer is a property of an entity that must first be identified through another relation, requires_decomposition must be true.
- Generate up to {self.config.candidate_plan_count} semantically distinct candidate plans.
- A direct one-relation lookup has requires_decomposition=false and candidate_plans=[].
- A multi-hop plan has one atomic retrieval step per missing direct relation transition.
- Every step relation must directly hold between that step's source and output. Do not use the overall composed relation as a substitute for a direct edge.
- Each dependent query must contain the exact prior-step placeholder.
- query_template must contain a literal known entity or an exact step placeholder such as {{{{s1}}}}. Never use {{{{source_ref}}}} or {{source_ref}} in query_template.
- source_type must describe the input to the step; target_type must describe its output.
- The final target_type must match the requested answer type.
- Start from an entity explicitly known in the question and move toward the unknown answer.
- Do not reverse relations, ask the final question prematurely, repeat a step, or use a merely correlated path.
- The composed sequence of direct relations must logically entail the requested answer. If all step facts could be true while the answer still would not follow, reject that plan.
- Do not encode a list of possible entities as sequential steps; use multiple candidates and beam search.
- Use relation descriptions that are understandable without a dataset-specific ontology.
- At most {self.config.max_hops} dependency levels and {self.config.max_subqueries} steps.
""",
            2600,
        )

    async def build_plan(
        self,
        question: str,
    ) -> tuple[QuestionStructure, list[QueryStep]]:
        feedback: list[str] = []
        best_record: dict[str, Any] | None = None

        for _ in range(max(1, self.config.max_plan_generation_attempts)):
            data = await self._generate_candidate_plan_data(
                question,
                feedback,
            )
            analysis = data.get("question_analysis", {})
            if not isinstance(analysis, dict):
                analysis = {}

            base_structure = self._parse_structure(
                {
                    **analysis,
                    "dependency_depth": analysis.get(
                        "estimated_hops",
                        analysis.get("dependency_depth", 1),
                    ),
                    "relation_chain": [],
                }
            )
            requires_decomposition = bool(
                analysis.get(
                    "requires_decomposition",
                    base_structure.dependency_depth >= 2,
                )
            )

            if not requires_decomposition:
                gate = await self._validate_decomposition_decision(
                    question,
                    analysis,
                )
                if gate["requires_decomposition"]:
                    feedback = [
                        (
                            "The initial analysis incorrectly treated a "
                            "nested relational description as a known entity. "
                            f"Regenerate a {gate['estimated_hops']}-hop plan. "
                            f"Reason: {gate['reason']}"
                        )
                    ]
                    continue

                base_structure.dependency_depth = 1
                base_structure.query_type = "single_hop"
                return base_structure, []

            records: list[dict[str, Any]] = []
            raw_plans = data.get("candidate_plans", [])
            if not isinstance(raw_plans, list):
                raw_plans = []

            for index, raw_plan in enumerate(
                raw_plans[: self.config.candidate_plan_count],
                start=1,
            ):
                if not isinstance(raw_plan, dict):
                    continue
                structure = QuestionStructure(
                    known_entities=list(base_structure.known_entities),
                    target_type=base_structure.target_type,
                    relation_chain=[
                        str(item).strip()
                        for item in raw_plan.get("relation_chain", [])
                        if str(item).strip()
                    ],
                    dependency_depth=min(
                        self.config.max_hops,
                        max(
                            1,
                            safe_int(
                                raw_plan.get(
                                    "dependency_depth",
                                    base_structure.dependency_depth,
                                ),
                                base_structure.dependency_depth,
                            ),
                        ),
                    ),
                    branch_count=max(
                        1,
                        safe_int(
                            raw_plan.get(
                                "branch_count",
                                base_structure.branch_count,
                            ),
                            base_structure.branch_count,
                        ),
                    ),
                    query_type=str(
                        raw_plan.get(
                            "query_type",
                            base_structure.query_type,
                        )
                    ).strip().lower(),
                    known_entity_types=dict(
                        base_structure.known_entity_types
                    ),
                )
                steps = self._parse_steps(
                    raw_plan.get("steps", []),
                    structure.known_entities,
                )
                structure.relation_chain = [
                    step.relation for step in steps
                ]
                structure.dependency_depth = max(
                    structure.dependency_depth,
                    min(self.config.max_hops, max(1, len(steps))),
                )
                static = self._static_plan_diagnostics(
                    question,
                    structure,
                    steps,
                )
                records.append(
                    {
                        "plan_id": str(
                            raw_plan.get("plan_id", f"p{index}")
                        ).strip() or f"p{index}",
                        "structure": structure,
                        "steps": steps,
                        "static": static,
                    }
                )

            if not records:
                feedback = [
                    "The question was classified as multi-hop but no executable candidate plan was returned."
                ]
                continue

            semantic = await self._semantic_assess_plans(
                question,
                records,
            )

            for record in records:
                semantic_info = semantic.get(
                    record["plan_id"],
                    {
                        "valid": False,
                        "semantic_score": 0.0,
                        "issues": ["semantic evaluation missing"],
                    },
                )
                record["semantic"] = semantic_info
                record["combined_score"] = clamp01(
                    self.config.plan_static_weight
                    * record["static"]["static_score"]
                    + self.config.plan_semantic_weight
                    * semantic_info["semantic_score"]
                )
                record["valid"] = bool(
                    record["static"]["valid"]
                    and semantic_info.get("valid", False)
                    and record["static"]["static_score"]
                    >= self.config.min_static_plan_score
                    and semantic_info["semantic_score"]
                    >= self.config.min_semantic_plan_score
                    and record["combined_score"]
                    >= self.config.min_combined_plan_score
                )

                if (
                    best_record is None
                    or record["combined_score"] > best_record["combined_score"]
                ):
                    best_record = record

            valid_records = [
                record for record in records if record["valid"]
            ]
            if valid_records:
                valid_records.sort(
                    key=lambda record: record["combined_score"],
                    reverse=True,
                )
                selected = valid_records[0]
                return selected["structure"], selected["steps"]

            feedback = []
            for record in records:
                feedback.extend(record["static"]["hard_issues"])
                feedback.extend(record["static"]["soft_issues"])
                feedback.extend(record["semantic"].get("issues", []))
            feedback = list(dict.fromkeys(feedback))[:12]

        if best_record is not None:
            return best_record["structure"], best_record["steps"]

        fallback_structure = QuestionStructure(
            known_entities=[],
            target_type="other",
            relation_chain=[],
            dependency_depth=1,
            branch_count=1,
            query_type="single_hop",
        )
        return fallback_structure, []

    async def _repair_plan_once(
        self,
        question: str,
        structure: QuestionStructure,
        steps: list[QueryStep],
        issues: list[str],
        strong: bool,
    ) -> tuple[
        QuestionStructure,
        list[QueryStep],
        bool,
        list[str],
    ]:
        plan = self._plan_as_dict(structure, steps)
        instruction = (
            "The previous repair was ineffective. Return a materially different plan that fixes every issue."
            if strong
            else "Repair every listed generic issue."
        )

        data = await self._json_call(
            (
                "You repair an abstract dependency-aware retrieval plan. "
                "Do not answer the question and do not use entity-specific shortcuts. Return JSON only."
            ),
            f"""
Question:
{question}

Current plan:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Generic validation issues:
{json.dumps(issues, ensure_ascii=False)}

Return:
{{
  "valid": true,
  "issues": ["remaining issue"],
  "repaired_plan": {{
    "known_entities": [{{"name": "literal known entity", "type": "type"}}],
    "target_type": "requested answer type",
    "relation_chain": ["directed relation"],
    "dependency_depth": 2,
    "branch_count": 1,
    "query_type": "chain",
    "steps": [
      {{
        "step_id": "s1",
        "source_ref": "literal known entity",
        "source_type": "input type",
        "query_template": "one atomic factual query",
        "relation": "directed relation",
        "depends_on": [],
        "target_type": "output type"
      }}
    ]
  }}
}}

Rules:
- {instruction}
- Preserve literal known entities and the requested answer type.
- Move from known information toward the unknown answer in the correct relation direction.
- Each step retrieves exactly one missing direct relation.
- Every dependency must be used through its exact step placeholder in source_ref or query_template.
- query_template must contain a literal known entity or an exact step placeholder such as {{s1}}. Never emit {{source_ref}}.
- A relational noun phrase is not a named entity; expand unresolved relational descriptions into intermediate steps.
- Adjacent source and target types must be compatible.
- The last step output type must match the requested answer type.
- The composed direct-relation path must logically entail the requested answer.
- Remove reversed, circular, redundant, transitive-but-not-equivalent, merely correlated, or answer-leaking steps.
- Do not add a rule for a particular entity or dataset.
""",
            1900,
        )

        model_issues = [
            str(item).strip()
            for item in data.get("issues", [])
            if str(item).strip()
        ]
        repaired_data = data.get("repaired_plan")
        if not isinstance(repaired_data, dict):
            return structure, steps, False, model_issues

        candidate_structure = self._parse_structure(repaired_data)
        if not candidate_structure.known_entities:
            candidate_structure.known_entities = list(
                structure.known_entities
            )
            candidate_structure.known_entity_types = dict(
                structure.known_entity_types
            )
        if not candidate_structure.target_type:
            candidate_structure.target_type = structure.target_type

        candidate_structure.dependency_depth = min(
            self.config.max_hops,
            max(1, candidate_structure.dependency_depth),
        )
        candidate_steps = self._parse_steps(
            repaired_data.get("steps", []),
            candidate_structure.known_entities,
        )
        candidate_structure.relation_chain = [
            step.relation for step in candidate_steps
        ]
        candidate_structure.dependency_depth = max(
            candidate_structure.dependency_depth,
            min(self.config.max_hops, max(1, len(candidate_steps))),
        )
        return (
            candidate_structure,
            candidate_steps,
            bool(data.get("valid", False)),
            model_issues,
        )

    async def validate_and_repair_plan(
        self,
        question: str,
        structure: QuestionStructure,
        steps: list[QueryStep],
    ) -> tuple[QuestionStructure, list[QueryStep], dict[str, Any]]:
        original_plan = self._plan_as_dict(structure, steps)
        original_signature = self._plan_signature(structure, steps)
        info: dict[str, Any] = {
            "performed": bool(steps),
            "valid": True,
            "repaired": False,
            "issues": [],
            "repair_history_issues": [],
            "original_plan": original_plan,
            "final_plan": original_plan,
            "repair_attempts": 0,
            "validation_mode": "generic_types_dependencies_direction",
        }

        if not self.config.enable_plan_validation or not steps:
            return structure, steps, info

        current_structure = structure
        current_steps = steps
        static = self._static_plan_diagnostics(
            question,
            current_structure,
            current_steps,
        )
        semantic_map = await self._semantic_assess_plans(
            question,
            [
                {
                    "plan_id": "selected",
                    "structure": current_structure,
                    "steps": current_steps,
                    "static": static,
                }
            ],
        )
        semantic = semantic_map.get(
            "selected",
            {
                "valid": False,
                "semantic_score": 0.0,
                "issues": ["semantic validation missing"],
            },
        )

        def valid_now() -> bool:
            return bool(
                static["valid"]
                and semantic.get("valid", False)
                and static["static_score"]
                >= self.config.min_static_plan_score
                and semantic["semantic_score"]
                >= self.config.min_semantic_plan_score
            )

        all_issues = [
            *static["hard_issues"],
            *static["soft_issues"],
            *semantic.get("issues", []),
        ]

        for attempt in range(
            max(0, self.config.max_plan_repair_attempts)
        ):
            if valid_now():
                break
            before = self._plan_signature(
                current_structure,
                current_steps,
            )
            (
                repaired_structure,
                repaired_steps,
                _,
                model_issues,
            ) = await self._repair_plan_once(
                question,
                current_structure,
                current_steps,
                list(dict.fromkeys(all_issues)),
                strong=attempt > 0,
            )
            info["repair_attempts"] += 1
            current_structure = repaired_structure
            current_steps = repaired_steps
            static = self._static_plan_diagnostics(
                question,
                current_structure,
                current_steps,
            )
            semantic_map = await self._semantic_assess_plans(
                question,
                [
                    {
                        "plan_id": "selected",
                        "structure": current_structure,
                        "steps": current_steps,
                        "static": static,
                    }
                ],
            )
            semantic = semantic_map.get(
                "selected",
                {
                    "valid": False,
                    "semantic_score": 0.0,
                    "issues": ["semantic validation missing"],
                },
            )
            all_issues.extend(model_issues)
            all_issues.extend(static["hard_issues"])
            all_issues.extend(static["soft_issues"])
            all_issues.extend(semantic.get("issues", []))

            if self._plan_signature(
                current_structure,
                current_steps,
            ) == before:
                all_issues.append(
                    "repair did not materially change the invalid plan"
                )

        is_valid = valid_now()
        final_signature = self._plan_signature(
            current_structure,
            current_steps,
        )
        final_issues = list(
            dict.fromkeys(
                [
                    *static["hard_issues"],
                    *static["soft_issues"],
                    *semantic.get("issues", []),
                ]
            )
        )
        info.update(
            {
                "valid": is_valid,
                "repaired": final_signature != original_signature,
                "issues": final_issues,
                "repair_history_issues": list(
                    dict.fromkeys(all_issues)
                ),
                "static_diagnostics": static,
                "semantic_diagnostics": semantic,
                "final_plan": self._plan_as_dict(
                    current_structure,
                    current_steps,
                ),
            }
        )

        return current_structure, current_steps, info

    @staticmethod
    def need_decomposition(
        structure: QuestionStructure,
    ) -> bool:
        return (
            structure.dependency_depth >= 2
            or structure.branch_count >= 2
            or structure.query_type
            in {
                "chain",
                "comparison",
                "intersection",
                "composition",
            }
        )

    @staticmethod
    def instantiate(
        step: QueryStep,
        answers: dict[str, str],
    ) -> str:
        query = step.query_template

        for dependency in step.depends_on:
            value = answers.get(
                dependency,
                "",
            )
            query = query.replace(
                "{{" + dependency + "}}",
                value,
            )
            query = query.replace(
                "{" + dependency + "}",
                value,
            )

        unresolved = re.findall(
            r"\{\{?[^{}]+\}?\}",
            query,
        )
        if unresolved:
            raise ValueError(
                f"存在未解析占位符：{unresolved}"
            )

        return query.strip()

    # -------------------------------------------------------------------------
    # 中间实体提取：仅用于后续查询实例化
    # -------------------------------------------------------------------------

    @staticmethod
    def _anchor_entities_for_step(
        step: QueryStep,
        query: str,
        known_entities: list[str],
        answers: dict[str, str],
    ) -> list[str]:
        anchors: list[str] = []

        for dependency in step.depends_on:
            value = str(answers.get(dependency, "")).strip()
            if value:
                anchors.append(value)

        query_norm = normalize(query)
        for entity in known_entities:
            entity = str(entity).strip()
            if entity and normalize(entity) in query_norm:
                anchors.append(entity)

        # 第一步通常没有 depends_on；保留显式实体作为关系锚点。
        if not anchors:
            anchors.extend(
                str(entity).strip()
                for entity in known_entities
                if str(entity).strip()
            )

        return list(dict.fromkeys(anchors))

    @staticmethod
    def _text_mentions_entity(
        text: str,
        entity: str,
    ) -> bool:
        entity_norm = normalize(entity)
        text_norm = normalize(text)
        return bool(entity_norm and entity_norm in text_norm)

    @staticmethod
    def _evidence_span_is_grounded(
        evidence: str,
        context: str,
    ) -> bool:
        """
        检查验证证据是否来自检索上下文。

        V6.2不再只做整句原样包含检查，因为检索上下文中常含有：
        - 换行、标题、编号和引用标记；
        - 人名中的附加称号；
        - 大模型对原句进行的轻微压缩或标点改写。

        但仍保持严格门槛：
        1. 优先接受标准化后的原样包含；
        2. 否则只在单个句子/文本行内做高阈值模糊匹配；
        3. 证据词项覆盖率和顺序相似度必须同时足够高。
        """
        from collections import Counter
        from difflib import SequenceMatcher

        evidence_norm = normalize(evidence)
        context_norm = normalize(context)
        if not evidence_norm or not context_norm:
            return False

        evidence_tokens = evidence_norm.split()

        # 过短证据容易发生偶然匹配，不作为严格覆盖证据。
        if len(evidence_tokens) < 5:
            return False

        # 最可靠的情况：标准化之后可以直接在上下文中找到。
        if evidence_norm in context_norm:
            return True

        evidence_counter = Counter(evidence_tokens)

        # 保留句子和原始文本行的局部边界，避免在整篇上下文中
        # 用分散词项拼出一条并不存在的证据。
        segments = [
            normalize(segment)
            for segment in re.split(
                r"[\n\r]+|(?<=[.!?。！？;；])\s+",
                str(context or ""),
            )
        ]
        segments = [
            segment
            for segment in segments
            if segment
        ]

        for segment in segments:
            segment_tokens = segment.split()
            if len(segment_tokens) < 4:
                continue

            # 候选局部文本过短或过长时，不进行宽松匹配。
            length_ratio = (
                len(segment_tokens)
                / max(1, len(evidence_tokens))
            )
            if length_ratio < 0.55 or length_ratio > 3.50:
                continue

            segment_counter = Counter(segment_tokens)
            common_count = sum(
                min(count, segment_counter.get(token, 0))
                for token, count in evidence_counter.items()
            )
            token_recall = (
                common_count
                / max(1, len(evidence_tokens))
            )

            sequence_ratio = SequenceMatcher(
                None,
                evidence_norm,
                segment,
            ).ratio()

            # 两种可靠情况：
            # A. 词项几乎全部来自同一句，且顺序相似度较高；
            # B. 顺序相似度非常高，允许少量称号或修饰语差异。
            if (
                token_recall >= 0.88
                and sequence_ratio >= 0.72
            ):
                return True

            if sequence_ratio >= 0.86:
                return True

        return False

    @staticmethod
    def _relation_phrase_is_grounded(
        relation_phrase: str,
        evidence: str,
        context: str,
    ) -> bool:
        """
        通用关系落地检查。

        不维护数据集专属关系词表。验证模型必须从证据中复制一个明确表达
        目标关系的短语，代码只检查该短语是否真实出现在证据和检索上下文中。
        """
        phrase = normalize(relation_phrase)
        evidence_norm = normalize(evidence)
        if not phrase or len(phrase.split()) > 12:
            return False
        if phrase not in evidence_norm:
            return False
        return RetrievalFeedbackQueryDecomposer._evidence_span_is_grounded(
            relation_phrase,
            context,
        ) or phrase in normalize(context)

    async def verify_candidates(
        self,
        original_question: str,
        query: str,
        step: QueryStep,
        anchor_entities: list[str],
        raw_candidates: list[Candidate],
        retrieval: DualRetrieval,
        strict_grounding: bool = False,
    ) -> CandidateBatch:
        if not raw_candidates:
            return CandidateBatch()

        if not self.config.enable_relation_verification:
            accepted: list[Candidate] = []
            for candidate in raw_candidates:
                candidate.raw_confidence = candidate.confidence
                candidate.anchor_entity = (
                    anchor_entities[0] if anchor_entities else ""
                )
                candidate.anchor_present = True
                candidate.candidate_present = True
                candidate.relation_supported = True
                candidate.type_match = True
                candidate.relation_confidence = candidate.confidence
                candidate.evidence_confidence = candidate.confidence
                candidate.verification_score = candidate.confidence
                candidate.relation_entailment = True
                candidate.counterexample_possible = False
                candidate.structural_relation_supported = (
                    "structural" in candidate.sources
                )
                candidate.textual_relation_supported = (
                    "textual" in candidate.sources
                )
                candidate.verified = True
                accepted.append(candidate)
            return CandidateBatch(accepted=accepted, rejected=[])

        candidates_payload = [
            {
                "entity": candidate.entity,
                "raw_confidence": candidate.confidence,
                "sources": candidate.sources,
                "extraction_evidence": candidate.evidence,
            }
            for candidate in raw_candidates
        ]

        data = await self._json_call(
            (
                "Verify intermediate entities against one exact relation. "
                "Reject lexical overlap, mere co-occurrence, wrong relation direction, "
                "and unsupported inference. Do not answer the original question. "
                "Return JSON only."
            ),
            f"""
Original question:
{original_question}

Current intermediate sub-query:
{query}

Allowed anchor entities:
{json.dumps(anchor_entities, ensure_ascii=False)}

Target relation:
{step.relation}

Expected candidate type:
{step.target_type}

Candidates to verify:
{json.dumps(candidates_payload, ensure_ascii=False, indent=2)}

===== STRUCTURAL EVIDENCE =====
{smart_truncate(retrieval.structural.context, self.config.max_context_chars)}

===== TEXTUAL EVIDENCE =====
{smart_truncate(retrieval.textual.context, self.config.max_context_chars)}

Return:
{{
  "verifications": [
    {{
      "entity": "candidate exactly as supplied",
      "anchor_entity": "one supplied anchor",
      "anchor_present": true,
      "candidate_present": true,
      "relation_supported": true,
      "structural_relation_supported": true,
      "textual_relation_supported": true,
      "structural_evidence": "direct span copied only from structural evidence",
      "textual_evidence": "direct span copied only from textual evidence",
      "type_match": true,
      "relation_confidence": 0.0,
      "evidence_confidence": 0.0,
      "evidence": "short direct evidence span containing anchor, relation and candidate",
      "relation_phrase": "exact short phrase copied from evidence that expresses the target relation",
      "relation_entailment": true,
      "counterexample_possible": false,
      "reason": "brief verification reason"
    }}
  ]
}}

Hard rules:
- Verify exactly one directed triple: anchor entity -- target relation --> candidate entity.
- Relation direction must exactly match the sub-query.
- Mere name similarity, co-occurrence, topic relevance, transitive inference, or a related relation is insufficient.
- Select anchor_entity only from Allowed anchor entities.
- Copy a short evidence span from the supplied context; do not invent or merge distant sentences.
- relation_phrase must be copied from that evidence and must explicitly express the target relation.
- relation_entailment=true only when the evidence alone logically entails the exact directed triple.
- Set counterexample_possible=true when the evidence could be true while the target triple is false. Mere shared employment, office holding, co-occurrence, support, affiliation, residence, production involvement, or topical association is insufficient unless it directly entails the requested relation.
- relation_supported=false when the relation is absent, reversed, ambiguous, inferred, transitive, merely correlated, or only approximately related.
- Evaluate structural_relation_supported using only STRUCTURAL EVIDENCE.
- Evaluate textual_relation_supported using only TEXTUAL EVIDENCE.
- structural_evidence and textual_evidence must each be a direct local span copied from its own channel and must contain the anchor, relation and candidate; otherwise use an empty string and mark that channel unsupported.
- relation_supported may be true when at least one channel independently entails the exact directed relation; do not mark both channels true from combined co-occurrence.
- type_match concerns the candidate's semantic type, not the wording of its name.
- Return one verification for every supplied candidate.
""",
            1300,
        )

        verification_items = [
            item
            for item in data.get("verifications", [])
            if isinstance(item, dict)
        ]

        combined_context = (
            retrieval.structural.context
            + "\n"
            + retrieval.textual.context
        )
        accepted: list[Candidate] = []
        rejected: list[Candidate] = []

        for candidate in raw_candidates:
            verification = next(
                (
                    item
                    for item in verification_items
                    if entity_similarity(
                        str(item.get("entity", "")),
                        candidate.entity,
                    )
                    >= 0.90
                ),
                None,
            )

            if verification is None:
                candidate.raw_confidence = candidate.confidence
                candidate.verification_reason = (
                    "verification result missing for candidate"
                )
                candidate.hard_failure_reasons = [
                    "verification_result_missing"
                ]
                candidate.verified = False
                candidate.confidence = 0.0
                rejected.append(candidate)
                continue

            anchor_entity = str(
                verification.get("anchor_entity", "")
            ).strip()
            if anchor_entities and not any(
                entity_similarity(anchor_entity, anchor) >= 0.90
                for anchor in anchor_entities
            ):
                anchor_entity = ""

            anchor_present_model = bool(
                verification.get("anchor_present", False)
            )
            candidate_present_model = bool(
                verification.get("candidate_present", False)
            )
            relation_supported = bool(
                verification.get("relation_supported", False)
            )
            structural_relation_supported = bool(
                verification.get(
                    "structural_relation_supported",
                    False,
                )
            )
            textual_relation_supported = bool(
                verification.get(
                    "textual_relation_supported",
                    False,
                )
            )
            structural_verification_evidence = str(
                verification.get("structural_evidence", "")
            ).strip()
            textual_verification_evidence = str(
                verification.get("textual_evidence", "")
            ).strip()
            type_match = bool(
                verification.get("type_match", False)
            )
            relation_confidence = clamp01(
                verification.get(
                    "relation_confidence",
                    verification.get("confidence", 0.0),
                )
            )
            evidence_confidence = clamp01(
                verification.get("evidence_confidence", 0.0)
            )
            verification_evidence = str(
                verification.get("evidence", "")
            ).strip()
            verification_relation_phrase = str(
                verification.get("relation_phrase", "")
            ).strip()
            verification_reason = str(
                verification.get("reason", "")
            ).strip()
            relation_entailment = bool(
                verification.get("relation_entailment", False)
            )
            counterexample_possible = bool(
                verification.get("counterexample_possible", True)
            )

            candidate_in_context = self._text_mentions_entity(
                combined_context,
                candidate.entity,
            )
            anchor_in_context = (
                any(
                    self._text_mentions_entity(
                        combined_context,
                        anchor,
                    )
                    for anchor in anchor_entities
                )
                if anchor_entities
                else bool(anchor_entity)
            )

            evidence_mentions_candidate = self._text_mentions_entity(
                verification_evidence,
                candidate.entity,
            )
            evidence_mentions_anchor = (
                any(
                    self._text_mentions_entity(
                        verification_evidence,
                        anchor,
                    )
                    for anchor in anchor_entities
                )
                if anchor_entities
                else bool(verification_evidence)
            )

            def channel_support_is_grounded(
                model_supported: bool,
                channel_evidence: str,
                channel_context: str,
            ) -> bool:
                if not model_supported or not channel_evidence:
                    return False
                candidate_mentioned = self._text_mentions_entity(
                    channel_evidence,
                    candidate.entity,
                )
                anchor_mentioned = (
                    any(
                        self._text_mentions_entity(
                            channel_evidence,
                            anchor,
                        )
                        for anchor in anchor_entities
                    )
                    if anchor_entities
                    else bool(channel_evidence)
                )
                return bool(
                    candidate_mentioned
                    and anchor_mentioned
                    and self._evidence_span_is_grounded(
                        channel_evidence,
                        channel_context,
                    )
                )

            structural_relation_supported = (
                channel_support_is_grounded(
                    structural_relation_supported,
                    structural_verification_evidence,
                    retrieval.structural.context,
                )
            )
            textual_relation_supported = (
                channel_support_is_grounded(
                    textual_relation_supported,
                    textual_verification_evidence,
                    retrieval.textual.context,
                )
            )

            # 短证据至少应明确点名候选；锚点可以由标题/表格行上下文给出，
            # 但缺少锚点时降低证据置信度，不能靠共现绕过关系硬约束。
            direct_evidence = bool(
                verification_evidence
                and evidence_mentions_candidate
                and (
                    evidence_mentions_anchor
                    or relation_confidence >= 0.90
                )
            )
            if not direct_evidence:
                evidence_confidence *= 0.50

            # 原问题上下文覆盖判断采用更严格的证据门槛：
            # 1. 证据短句必须同时包含锚点和候选实体；
            # 2. 证据短句必须能在检索上下文中原样找到；
            # 3. 高风险关系必须出现对应关系触发词。
            strict_direct_evidence = bool(
                verification_evidence
                and evidence_mentions_candidate
                and evidence_mentions_anchor
            )
            evidence_grounded = self._evidence_span_is_grounded(
                verification_evidence,
                combined_context,
            )
            relation_phrase_grounded = (
                self._relation_phrase_is_grounded(
                    verification_relation_phrase,
                    verification_evidence,
                    combined_context,
                )
            )
            strict_grounding_pass = (
                not strict_grounding
                or (
                    strict_direct_evidence
                    and evidence_grounded
                    and relation_phrase_grounded
                )
            )

            if strict_grounding and not strict_grounding_pass:
                failure_parts: list[str] = []
                if not strict_direct_evidence:
                    failure_parts.append(
                        "evidence does not mention both anchor and candidate"
                    )
                if not evidence_grounded:
                    failure_parts.append(
                        "evidence span is not grounded in retrieved context"
                    )
                if not relation_phrase_grounded:
                    failure_parts.append(
                        "relation phrase is missing or not grounded in the retrieved evidence"
                    )
                verification_reason = (
                    "; ".join(failure_parts)
                    or verification_reason
                )

            cross_support = (
                1.0 if len(candidate.sources) > 1 else 0.0
            )
            verification_score = clamp01(
                self.config.verification_relation_weight
                * relation_confidence
                + self.config.verification_type_weight
                * (1.0 if type_match else 0.0)
                + self.config.verification_evidence_weight
                * evidence_confidence
                + self.config.verification_cross_weight
                * cross_support
            )

            anchor_present = (
                anchor_present_model
                and anchor_in_context
            )
            candidate_present = (
                candidate_present_model
                and candidate_in_context
            )

            hard_failure_reasons: list[str] = []
            if not anchor_present:
                hard_failure_reasons.append("anchor_not_present_in_context")
            if not candidate_present:
                hard_failure_reasons.append("candidate_not_present_in_context")
            if not verification_evidence:
                hard_failure_reasons.append("verification_evidence_missing")
            if self.config.require_relation_support and not relation_supported:
                hard_failure_reasons.append("relation_not_supported")
            if relation_supported and not (
                structural_relation_supported
                or textual_relation_supported
            ):
                hard_failure_reasons.append(
                    "relation_not_supported_by_either_channel"
                )
            if not relation_entailment:
                hard_failure_reasons.append("relation_not_entailed_by_evidence")
            if counterexample_possible:
                hard_failure_reasons.append("relation_counterexample_possible")
            if self.config.require_type_match and not type_match:
                hard_failure_reasons.append("candidate_type_mismatch")
            if verification_score < self.config.min_verified_candidate_score:
                hard_failure_reasons.append("verification_score_below_threshold")
            if not strict_grounding_pass:
                hard_failure_reasons.append("strict_evidence_grounding_failed")

            hard_accept = not hard_failure_reasons

            candidate.raw_confidence = candidate.confidence
            candidate.anchor_entity = anchor_entity
            candidate.anchor_present = anchor_present
            candidate.candidate_present = candidate_present
            candidate.relation_supported = relation_supported
            candidate.type_match = type_match
            candidate.relation_confidence = relation_confidence
            candidate.evidence_confidence = evidence_confidence
            candidate.verification_score = verification_score
            candidate.verification_evidence = verification_evidence
            candidate.verification_relation_phrase = (
                verification_relation_phrase
            )
            candidate.verification_reason = verification_reason
            candidate.relation_entailment = relation_entailment
            candidate.counterexample_possible = counterexample_possible
            candidate.structural_relation_supported = (
                structural_relation_supported
            )
            candidate.textual_relation_supported = (
                textual_relation_supported
            )
            candidate.structural_verification_evidence = (
                structural_verification_evidence
            )
            candidate.textual_verification_evidence = (
                textual_verification_evidence
            )
            candidate.hard_failure_reasons = hard_failure_reasons
            candidate.verified = hard_accept
            candidate.confidence = (
                verification_score if hard_accept else 0.0
            )

            if hard_accept:
                accepted.append(candidate)
            else:
                rejected.append(candidate)

        accepted.sort(
            key=lambda candidate: candidate.verification_score,
            reverse=True,
        )
        rejected.sort(
            key=lambda candidate: candidate.verification_score,
            reverse=True,
        )
        return CandidateBatch(
            accepted=accepted[
                : self.config.candidates_per_step
            ],
            rejected=rejected,
        )

    async def extract_candidates(
        self,
        original_question: str,
        query: str,
        step: QueryStep,
        retrieval: DualRetrieval,
        anchor_entities: list[str],
        strict_grounding: bool = False,
    ) -> CandidateBatch:
        data = await self._json_call(
            (
                "Extract candidate intermediate entities for one exact sub-query. "
                "Keep channels separate. Extraction is provisional: a later verifier "
                "will reject wrong relations. Do not answer the original question. "
                "Return JSON only."
            ),
            f"""
Original question:
{original_question}

Current intermediate sub-query:
{query}

Anchor entities:
{json.dumps(anchor_entities, ensure_ascii=False)}

Target relation:
{step.relation}

Expected type:
{step.target_type}

===== STRUCTURAL CHANNEL: ACCUMULATED + CURRENT EVIDENCE =====
{smart_truncate(retrieval.structural.context, self.config.max_context_chars)}

===== TEXTUAL CHANNEL: ACCUMULATED + CURRENT EVIDENCE =====
{smart_truncate(retrieval.textual.context, self.config.max_context_chars)}

Return:
{{
  "structural_candidates": [
    {{
      "entity": "candidate",
      "confidence": 0.0,
      "evidence": "supporting relation or sentence"
    }}
  ],
  "textual_candidates": [
    {{
      "entity": "candidate",
      "confidence": 0.0,
      "evidence": "supporting sentence"
    }}
  ]
}}

Rules:
- Extract only the object/result requested by the current directed relation.
- Every candidate string must occur as a contiguous entity span in that channel's supplied context.
- Do not invent a canonical name from an abbreviation, description, or world knowledge.
- Do not select entities merely because their names overlap with the anchor/query.
- The candidate type must match Expected type.
- Search the whole accumulated context, including original-question evidence.
- For plural targets, return each candidate separately so beam search can preserve alternatives.
- Keep structural and textual candidates separate.
- Return an empty list for a channel that lacks explicit supporting evidence.
- Never answer beyond the current step.
""",
            950,
        )

        mentions: list[tuple[str, str, float, str]] = []

        for source, field_name in (
            ("structural", "structural_candidates"),
            ("textual", "textual_candidates"),
        ):
            for item in data.get(field_name, []):
                if not isinstance(item, dict):
                    continue
                entity = str(item.get("entity", "")).strip()
                confidence = clamp01(item.get("confidence"))

                if (
                    entity
                    and confidence
                    >= self.config.min_raw_candidate_confidence
                ):
                    mentions.append(
                        (
                            entity,
                            source,
                            confidence,
                            str(item.get("evidence", "")).strip(),
                        )
                    )

        groups: list[dict[str, Any]] = []
        for entity, source, confidence, evidence in mentions:
            group = next(
                (
                    existing
                    for existing in groups
                    if entity_similarity(
                        entity,
                        existing["entity"],
                    )
                    >= 0.90
                ),
                None,
            )

            if group is None:
                groups.append(
                    {
                        "entity": entity,
                        "sources": {source},
                        "confidences": [confidence],
                        "evidence": [evidence] if evidence else [],
                    }
                )
            else:
                group["sources"].add(source)
                group["confidences"].append(confidence)
                if evidence:
                    group["evidence"].append(evidence)

        raw_candidates: list[Candidate] = []
        for group in groups:
            raw_confidence = max(group["confidences"])
            if len(group["sources"]) > 1:
                raw_confidence = min(
                    1.0,
                    raw_confidence
                    + self.config.agreement_bonus,
                )

            raw_candidates.append(
                Candidate(
                    entity=group["entity"],
                    confidence=raw_confidence,
                    raw_confidence=raw_confidence,
                    sources=sorted(group["sources"]),
                    evidence=list(
                        dict.fromkeys(group["evidence"])
                    ),
                )
            )

        raw_candidates.sort(
            key=lambda candidate: candidate.raw_confidence,
            reverse=True,
        )
        raw_candidates = raw_candidates[
            : max(
                self.config.candidates_per_step * 2,
                self.config.candidates_per_step,
            )
        ]

        return await self.verify_candidates(
            original_question=original_question,
            query=query,
            step=step,
            anchor_entities=anchor_entities,
            raw_candidates=raw_candidates,
            retrieval=retrieval,
            strict_grounding=strict_grounding,
        )


    # -------------------------------------------------------------------------
    # 原问题上下文的多跳关系链覆盖检查
    # -------------------------------------------------------------------------

    async def verify_terminal_relation_coverage(
        self,
        original_question: str,
        query: str,
        step: QueryStep,
        retrieval: DualRetrieval,
        anchor_entities: list[str],
    ) -> dict[str, Any]:
        """
        使用与中间步骤相同的三元证据验证判断终端关系是否已覆盖。

        方法内部会抽取“覆盖见证候选”，但候选实体不会写入中间答案，
        也不会传给最终答案模块；它只用于证明原问题上下文确实包含：
        锚点实体 --目标关系--> 目标类型对象。
        """
        candidate_batch = await self.extract_candidates(
            original_question=original_question,
            query=query,
            step=step,
            retrieval=retrieval,
            anchor_entities=anchor_entities,
            strict_grounding=True,
        )
        accepted = [
            candidate
            for candidate in candidate_batch.accepted
            if candidate.verification_score
            >= self.config.original_context_terminal_threshold
        ]
        supported = bool(accepted)
        structural_supported = any(
            candidate.structural_relation_supported
            for candidate in accepted
        )
        textual_supported = any(
            candidate.textual_relation_supported
            for candidate in accepted
        )
        confidence = max(
            (candidate.verification_score for candidate in accepted),
            default=0.0,
        )

        return {
            "supported": supported,
            "structural_supported": structural_supported,
            "textual_supported": textual_supported,
            "model_supported": bool(candidate_batch.accepted),
            "confidence": confidence,
            "witness_count": len(accepted),
            "evidence_summary": [
                candidate.verification_evidence
                for candidate in accepted[:3]
            ],
            "reason": (
                "verified terminal relation witness found in original-question context"
                if supported
                else (
                    "terminal candidates were extracted but failed strict relation grounding"
                    if candidate_batch.rejected
                    else "no terminal relation witness found"
                )
            ),
            # 不暴露内部目标实体，只保留拒绝诊断。
            "rejected_diagnostics": [
                {
                    "verification_score": candidate.verification_score,
                    "hard_failure_reasons": candidate.hard_failure_reasons,
                    "verification_reason": candidate.verification_reason,
                }
                for candidate in candidate_batch.rejected[:5]
            ],
        }

    async def assess_original_context_coverage(
        self,
        question: str,
        structure: QuestionStructure,
        plan: list[QueryStep],
        initial: DualRetrieval,
        base: PathState,
    ) -> tuple[
        dict[str, Any],
        list[PathState],
        list[QueryStep],
        set[str],
    ]:
        """
        检查原问题检索结果覆盖了关系链的多少个前序步骤。

        返回：
        1. coverage_info：完整诊断信息；
        2. seed_paths：后续检索的起始路径，包含原问题上下文中已验证的中间实体；
        3. remaining_plan：从首个缺失关系开始的实际子查询计划；
        4. covered_step_ids：已由原问题上下文覆盖的步骤ID。

        三种决策：
        - fully_covered：
          原问题上下文覆盖全部关系链，不执行任何子查询；
        - partially_covered：
          已覆盖一个或多个前序关系，从首个缺失步骤继续；
        - not_covered：
          第一跳无法得到可靠中间实体，从第一步开始完整逐跳检索。
        """
        info: dict[str, Any] = {
            "performed": False,
            "mode": "not_applicable",
            "decision": "execute_full_plan",
            "total_plan_steps": len(plan),
            "covered_prefix_steps": 0,
            "covered_step_ids": [],
            "first_missing_step_id": (
                plan[0].step_id if plan else ""
            ),
            "skipped_subqueries": 0,
            "seed_paths": [],
            "steps": [],
        }

        if (
            not self.config.enable_original_context_coverage
            or not plan
            or not self.need_decomposition(structure)
        ):
            return info, [base], plan, set()

        info["performed"] = True

        # 没有原问题检索上下文时，直接从第一跳开始。
        if not (
            initial.structural.context.strip()
            or initial.textual.context.strip()
        ):
            info.update(
                {
                    "mode": "not_covered",
                    "decision": "execute_full_plan",
                    "first_missing_step_id": plan[0].step_id,
                }
            )
            return info, [base], plan, set()

        required_outputs = {
            dependency
            for later_step in plan
            for dependency in later_step.depends_on
        }

        coverage_paths: list[PathState] = [base]
        covered_step_ids: list[str] = []

        for step_index, step in enumerate(plan):
            is_terminal = (
                step.step_id not in required_outputs
            )
            next_paths: list[PathState] = []
            step_path_records: list[dict[str, Any]] = []

            for path in coverage_paths:
                try:
                    query = self.instantiate(
                        step,
                        path.answers,
                    )
                except ValueError as error:
                    step_path_records.append(
                        {
                            "query": "",
                            "anchor_entities": [],
                            "covered": False,
                            "reason": str(error),
                            "accepted_candidates": [],
                            "rejected_candidates": [],
                        }
                    )
                    continue

                anchor_entities = self._anchor_entities_for_step(
                    step=step,
                    query=query,
                    known_entities=structure.known_entities,
                    answers=path.answers,
                )

                if is_terminal:
                    verdict = (
                        await self.verify_terminal_relation_coverage(
                            original_question=question,
                            query=query,
                            step=step,
                            retrieval=initial,
                            anchor_entities=anchor_entities,
                        )
                    )
                    step_path_records.append(
                        {
                            "query": query,
                            "anchor_entities": anchor_entities,
                            "covered": verdict["supported"],
                            "structural_supported": verdict.get(
                                "structural_supported", False
                            ),
                            "textual_supported": verdict.get(
                                "textual_supported", False
                            ),
                            "confidence": verdict["confidence"],
                            "evidence_summary": verdict[
                                "evidence_summary"
                            ],
                            "witness_count": verdict.get(
                                "witness_count", 0
                            ),
                            "reason": verdict["reason"],
                            "accepted_candidates": [],
                            "rejected_candidates": verdict.get(
                                "rejected_diagnostics", []
                            ),
                        }
                    )

                    if verdict["supported"]:
                        next_paths.append(
                            PathState(
                                answers=dict(path.answers),
                                structural_context=(
                                    path.structural_context
                                ),
                                textual_context=(
                                    path.textual_context
                                ),
                                structural_ids=set(
                                    path.structural_ids
                                ),
                                textual_ids=set(
                                    path.textual_ids
                                ),
                                scores=[
                                    *path.scores,
                                    verdict["confidence"],
                                ],
                                traces=list(path.traces),
                            )
                        )
                    continue

                candidate_batch = await self.extract_candidates(
                    original_question=question,
                    query=query,
                    step=step,
                    retrieval=initial,
                    anchor_entities=anchor_entities,
                    strict_grounding=True,
                )

                accepted = candidate_batch.accepted
                rejected = candidate_batch.rejected

                step_path_records.append(
                    {
                        "query": query,
                        "anchor_entities": anchor_entities,
                        "covered": bool(accepted),
                        "structural_supported": any(
                            candidate.structural_relation_supported
                            for candidate in accepted
                        ),
                        "textual_supported": any(
                            candidate.textual_relation_supported
                            for candidate in accepted
                        ),
                        "confidence": max(
                            (
                                candidate.confidence
                                for candidate in accepted
                            ),
                            default=0.0,
                        ),
                        "reason": (
                            "verified intermediate entity found "
                            "in original-question context"
                            if accepted
                            else (
                                "all extracted candidates were rejected"
                                if rejected
                                else "no intermediate entity found"
                            )
                        ),
                        "accepted_candidates": [
                            asdict(candidate)
                            for candidate in accepted
                        ],
                        "rejected_candidates": [
                            asdict(candidate)
                            for candidate in rejected[:5]
                        ],
                    }
                )

                for candidate in accepted:
                    answers = dict(path.answers)
                    answers[step.step_id] = (
                        candidate.entity
                    )
                    next_paths.append(
                        PathState(
                            answers=answers,
                            structural_context=(
                                path.structural_context
                            ),
                            textual_context=(
                                path.textual_context
                            ),
                            structural_ids=set(
                                path.structural_ids
                            ),
                            textual_ids=set(
                                path.textual_ids
                            ),
                            scores=[
                                *path.scores,
                                candidate.confidence,
                            ],
                            traces=list(path.traces),
                        )
                    )

            step_covered = bool(next_paths)
            structural_step_supported = any(
                bool(record.get("structural_supported", False))
                for record in step_path_records
                if record.get("covered", False)
            )
            textual_step_supported = any(
                bool(record.get("textual_supported", False))
                for record in step_path_records
                if record.get("covered", False)
            )
            info["steps"].append(
                {
                    "step_id": step.step_id,
                    "query_template": (
                        step.query_template
                    ),
                    "relation": step.relation,
                    "target_type": step.target_type,
                    "depends_on": list(step.depends_on),
                    "is_terminal": is_terminal,
                    "covered": step_covered,
                    "structural_supported": structural_step_supported,
                    "textual_supported": textual_step_supported,
                    "path_results": step_path_records,
                }
            )

            if not step_covered:
                remaining_plan = plan[step_index:]
                mode = (
                    "not_covered"
                    if not covered_step_ids
                    else "partially_covered"
                )
                decision = (
                    "execute_full_plan"
                    if mode == "not_covered"
                    else "start_from_first_missing_step"
                )
                info.update(
                    {
                        "mode": mode,
                        "decision": decision,
                        "covered_prefix_steps": len(
                            covered_step_ids
                        ),
                        "covered_step_ids": list(
                            covered_step_ids
                        ),
                        "first_missing_step_id": (
                            step.step_id
                        ),
                        "skipped_subqueries": len(
                            covered_step_ids
                        ),
                        "seed_paths": [
                            {
                                "score": path.score,
                                "intermediate_answers": (
                                    path.answers
                                ),
                            }
                            for path in coverage_paths
                        ],
                    }
                )
                return (
                    info,
                    coverage_paths,
                    remaining_plan,
                    set(covered_step_ids),
                )

            next_paths.sort(
                key=lambda path: path.score,
                reverse=True,
            )
            coverage_paths = next_paths[
                : max(
                    1,
                    min(
                        self.config.beam_width,
                        self.config.max_original_coverage_paths,
                    ),
                )
            ]
            covered_step_ids.append(step.step_id)

        info.update(
            {
                "mode": "fully_covered",
                "decision": "skip_all_subqueries",
                "covered_prefix_steps": len(
                    covered_step_ids
                ),
                "covered_step_ids": list(
                    covered_step_ids
                ),
                "first_missing_step_id": "",
                "skipped_subqueries": len(
                    covered_step_ids
                ),
                "seed_paths": [
                    {
                        "score": path.score,
                        "intermediate_answers": (
                            path.answers
                        ),
                    }
                    for path in coverage_paths
                ],
            }
        )
        return (
            info,
            coverage_paths,
            [],
            set(covered_step_ids),
        )

    # -------------------------------------------------------------------------
    # 查询改写与替代查询生成
    # -------------------------------------------------------------------------

    @staticmethod
    def _deduplicate_queries(
        original_query: str,
        queries: list[str],
    ) -> list[str]:
        seen = {normalize(original_query)}
        result: list[str] = []

        for query in queries:
            query = str(query or "").strip()
            key = normalize(query)
            if not query or not key or key in seen:
                continue
            seen.add(key)
            result.append(query)

        return result

    @staticmethod
    def _deterministic_alternative_query(
        failed_query: str,
        step: QueryStep,
        known_entities: list[str],
        answers: dict[str, str],
    ) -> str:
        relation = normalize(step.relation)
        anchor = ""
        if answers:
            anchor = list(answers.values())[-1]
        elif known_entities:
            anchor = known_entities[0]

        if not anchor:
            return ""

        if "employer" in relation or "employ" in relation:
            return f"Which organization did {anchor} work for?"
        if "owner" in relation or "owned" in relation:
            return f"Who owns {anchor}?"
        if "parent" in relation:
            return f"Who are the parents of {anchor}?"
        if "mother" in relation:
            return f"Who is the mother of {anchor}?"
        if "father" in relation:
            return f"Who is the father of {anchor}?"
        if "member" in relation or "membership" in relation:
            return f"Which organization is {anchor} a member of?"
        if "headquarter" in relation:
            return f"Where is {anchor} headquartered?"
        if "located" in relation or "location" in relation:
            return f"Where is {anchor} located?"
        if "perform" in relation:
            return f"Who performed the work titled {anchor}?"

        return ""

    async def generate_alternative_queries(
        self,
        original_question: str,
        failed_query: str,
        step: QueryStep,
        known_entities: list[str],
        answers: dict[str, str],
        retry_reason: str,
    ) -> list[str]:
        data = await self._json_call(
            (
                "Generate materially different retrieval-query alternatives while "
                "preserving one exact target relation. Do not answer the original "
                "question. Return JSON only."
            ),
            f"""
Original question:
{original_question}

Current query:
{failed_query}

Retry reason:
{retry_reason}

Target relation:
{step.relation}

Expected entity type:
{step.target_type}

Explicit entities from the original question:
{json.dumps(known_entities, ensure_ascii=False)}

Verified prior-step entities:
{json.dumps(answers, ensure_ascii=False)}

Return:
{{
  "alternative_queries": [
    "first materially different self-contained query",
    "second materially different self-contained query",
    "third materially different self-contained query"
  ]
}}

Rules:
- Preserve the exact target relation and the same reasoning hop.
- Keep all necessary names, titles, roles and disambiguating phrases.
- Use verified prior-step entities literally.
- Produce syntactically and lexically different alternatives; do not merely
  repeat or lightly punctuate the current query.
- Do not introduce an unknown entity or a new reasoning hop.
- Do not answer the original question.
- Do not use placeholders.
- Return at most {self.config.max_alternative_queries} alternatives.
""",
            500,
        )

        alternatives = [
            str(item).strip()
            for item in data.get("alternative_queries", [])
            if str(item).strip()
        ]
        alternatives = self._deduplicate_queries(
            failed_query,
            alternatives,
        )

        if not alternatives:
            fallback = self._deterministic_alternative_query(
                failed_query,
                step,
                known_entities,
                answers,
            )
            alternatives = self._deduplicate_queries(
                failed_query,
                [fallback],
            )

        return alternatives[: self.config.max_alternative_queries]

    def path_step_score(
        self,
        candidate_confidence: float,
        retrieval: DualRetrieval,
        novelty_score: float,
    ) -> float:
        ret_score = max(
            retrieval_score(
                retrieval.structural
            ),
            retrieval_score(
                retrieval.textual
            ),
        )

        return clamp01(
            0.55 * candidate_confidence
            + 0.25 * ret_score
            + 0.20 * novelty_score
        )

    # -------------------------------------------------------------------------
    # 完整执行流程
    # -------------------------------------------------------------------------

    async def run(
        self,
        question: str,
        retrieve_dual: (
            DualRetriever
            | Callable[
                [str],
                DualRetrieval | Awaitable[DualRetrieval],
            ]
        ),
    ) -> RFDQDResult:
        question = str(question).strip()
        if not question:
            raise ValueError("question不能为空")

        # 原问题检索与抽象查询计划并行生成。
        # 查询计划只用于识别关系链和检查覆盖情况；
        # 若原问题上下文已完整覆盖，计划中的子查询不会实际执行。
        ((structure, plan), initial) = await asyncio.gather(
            self.build_plan(question),
            maybe_await(retrieve_dual(question)),
        )

        if not isinstance(initial, DualRetrieval):
            raise TypeError("retrieve_dual必须返回DualRetrieval")

        initial_structural_records = channel_retrieval_records(
            initial.structural,
            channel="structural",
            query=question,
            step_id="original",
        )
        initial_textual_records = channel_retrieval_records(
            initial.textual,
            channel="textual",
            query=question,
            step_id="original",
        )

        plan_validation: dict[str, Any] = {
            "performed": False,
            "valid": True,
            "repaired": False,
            "issues": [],
        }
        coverage_info: dict[str, Any] = {
            "performed": False,
            "mode": "not_applicable",
            "decision": "execute_full_plan",
            "total_plan_steps": len(plan),
            "covered_prefix_steps": 0,
            "covered_step_ids": [],
            "first_missing_step_id": (
                plan[0].step_id if plan else ""
            ),
            "skipped_subqueries": 0,
            "seed_paths": [],
            "steps": [],
        }

        if self.need_decomposition(structure) and plan:
            structure, plan, plan_validation = (
                await self.validate_and_repair_plan(
                    question,
                    structure,
                    plan,
                )
            )

        base = PathState(
            answers={},
            structural_context=initial.structural.context,
            textual_context=initial.textual.context,
            structural_ids=channel_ids(initial.structural),
            textual_ids=channel_ids(initial.textual),
            scores=[],
            traces=[],
        )

        if (
            plan_validation.get("performed")
            and not plan_validation.get("valid", True)
        ):
            return RFDQDResult(
                question=question,
                need_decomposition=True,
                question_structure=structure,
                plan=plan,
                executed_subqueries=0,
                stop_reason="plan_validation_failed",
                selected_path_score=0.0,
                intermediate_answers={},
                traces=[],
                final_structural_context=base.structural_context,
                final_textual_context=base.textual_context,
                structural_evidence_ids=sorted(base.structural_ids),
                textual_evidence_ids=sorted(base.textual_ids),
                alternative_paths=[],
                plan_validation=plan_validation,
                original_context_coverage=coverage_info,
                initial_structural_retrieval_records=(
                    initial_structural_records
                ),
                initial_textual_retrieval_records=(
                    initial_textual_records
                ),
            )

        if not self.need_decomposition(structure) or not plan:
            return RFDQDResult(
                question=question,
                need_decomposition=False,
                question_structure=structure,
                plan=[],
                executed_subqueries=0,
                stop_reason="single_hop_or_no_plan",
                selected_path_score=0.0,
                intermediate_answers={},
                traces=[],
                final_structural_context=base.structural_context,
                final_textual_context=base.textual_context,
                structural_evidence_ids=sorted(base.structural_ids),
                textual_evidence_ids=sorted(base.textual_ids),
                alternative_paths=[],
                plan_validation=plan_validation,
                original_context_coverage=coverage_info,
                initial_structural_retrieval_records=(
                    initial_structural_records
                ),
                initial_textual_retrieval_records=(
                    initial_textual_records
                ),
            )

        # ------------------------------------------------------------------
        # V6：先检查原问题上下文覆盖了关系链的多少个前序步骤。
        # ------------------------------------------------------------------
        paths: list[PathState] = [base]
        execution_plan: list[QueryStep] = list(plan)
        covered_step_ids: set[str] = set()

        try:
            (
                coverage_info,
                paths,
                execution_plan,
                covered_step_ids,
            ) = await self.assess_original_context_coverage(
                question=question,
                structure=structure,
                plan=plan,
                initial=initial,
                base=base,
            )
        except Exception as error:
            # 覆盖检查失败时不丢失样本，回退到原有完整逐跳检索。
            coverage_info = {
                "performed": True,
                "mode": "coverage_check_failed",
                "decision": "execute_full_plan",
                "total_plan_steps": len(plan),
                "covered_prefix_steps": 0,
                "covered_step_ids": [],
                "first_missing_step_id": (
                    plan[0].step_id
                    if plan
                    else ""
                ),
                "skipped_subqueries": 0,
                "seed_paths": [],
                "steps": [],
                "error": str(error),
            }
            paths = [base]
            execution_plan = list(plan)
            covered_step_ids = set()

        # 原问题上下文已经覆盖完整关系链：不执行任何子查询。
        if coverage_info.get("mode") == "fully_covered":
            paths.sort(
                key=lambda path: path.score,
                reverse=True,
            )
            best = paths[0]
            return RFDQDResult(
                question=question,
                need_decomposition=False,
                question_structure=structure,
                plan=plan,
                executed_subqueries=0,
                stop_reason="original_context_fully_covered",
                selected_path_score=best.score,
                intermediate_answers=best.answers,
                traces=[],
                final_structural_context=best.structural_context,
                final_textual_context=best.textual_context,
                structural_evidence_ids=sorted(
                    best.structural_ids
                ),
                textual_evidence_ids=sorted(
                    best.textual_ids
                ),
                alternative_paths=[
                    {
                        "score": path.score,
                        "intermediate_answers": (
                            path.answers
                        ),
                    }
                    for path in paths[1:]
                ],
                plan_validation=plan_validation,
                original_context_coverage=coverage_info,
                initial_structural_retrieval_records=(
                    initial_structural_records
                ),
                initial_textual_retrieval_records=(
                    initial_textual_records
                ),
                structural_supported_step_ids=sorted(
                    coverage_channel_supported_step_ids(
                        coverage_info,
                        "structural",
                    )
                ),
                textual_supported_step_ids=sorted(
                    coverage_channel_supported_step_ids(
                        coverage_info,
                        "textual",
                    )
                ),
            )

        required_outputs = {
            dependency
            for later_step in plan
            for dependency in later_step.depends_on
        }

        query_count = 0
        stop_reason = "plan_completed"

        async def execute_attempt(
            query: str,
            step: QueryStep,
            path: PathState,
            structural_bundle: str,
            textual_bundle: str,
            seen_structural_ids: set[str],
            seen_textual_ids: set[str],
            extract_intermediate: bool,
        ) -> dict[str, Any] | None:
            nonlocal query_count

            if query_count >= self.config.max_subqueries:
                return None

            retrieval = await maybe_await(
                retrieve_dual(query)
            )
            query_count += 1

            if not isinstance(retrieval, DualRetrieval):
                raise TypeError(
                    "retrieve_dual必须返回DualRetrieval"
                )

            structural_ids = channel_ids(
                retrieval.structural
            )
            textual_ids = channel_ids(
                retrieval.textual
            )
            structural_novelty = novelty(
                structural_ids,
                seen_structural_ids,
            )
            textual_novelty = novelty(
                textual_ids,
                seen_textual_ids,
            )
            overall_novelty = max(
                structural_novelty,
                textual_novelty,
            )

            new_structural_bundle = merge_context(
                structural_bundle,
                retrieval.structural.context,
            )
            new_textual_bundle = merge_context(
                textual_bundle,
                retrieval.textual.context,
            )

            extraction_retrieval = DualRetrieval(
                structural=ChannelRetrieval(
                    context=compose_extraction_context(
                        path.structural_context,
                        new_structural_bundle,
                        self.config.max_context_chars,
                    )
                ),
                textual=ChannelRetrieval(
                    context=compose_extraction_context(
                        path.textual_context,
                        new_textual_bundle,
                        self.config.max_context_chars,
                    )
                ),
            )

            anchor_entities = (
                self._anchor_entities_for_step(
                    step=step,
                    query=query,
                    known_entities=(
                        structure.known_entities
                    ),
                    answers=path.answers,
                )
            )
            candidate_batch = CandidateBatch()
            if extract_intermediate:
                candidate_batch = (
                    await self.extract_candidates(
                        original_question=question,
                        query=query,
                        step=step,
                        retrieval=extraction_retrieval,
                        anchor_entities=anchor_entities,
                    )
                )
            candidates = candidate_batch.accepted
            rejected_candidates = (
                candidate_batch.rejected
            )

            top_confidence = max(
                (
                    candidate.confidence
                    for candidate in candidates
                ),
                default=0.0,
            )
            ret_score = max(
                retrieval_score(
                    retrieval.structural
                ),
                retrieval_score(
                    retrieval.textual
                ),
            )
            utility = clamp01(
                0.60 * top_confidence
                + 0.25 * ret_score
                + 0.15 * overall_novelty
            )

            return {
                "query": query,
                "retrieval": retrieval,
                "structural_ids": structural_ids,
                "textual_ids": textual_ids,
                "structural_novelty": (
                    structural_novelty
                ),
                "textual_novelty": textual_novelty,
                "overall_novelty": overall_novelty,
                "candidates": candidates,
                "rejected_candidates": (
                    rejected_candidates
                ),
                "anchor_entities": anchor_entities,
                "utility": utility,
                "structural_bundle": (
                    new_structural_bundle
                ),
                "textual_bundle": (
                    new_textual_bundle
                ),
                "seen_structural_ids": (
                    seen_structural_ids
                    | structural_ids
                ),
                "seen_textual_ids": (
                    seen_textual_ids
                    | textual_ids
                ),
            }

        # execution_plan 已经从首个缺失步骤开始。
        for step in execution_plan:
            if query_count >= self.config.max_subqueries:
                stop_reason = "query_budget_reached"
                break

            is_terminal = (
                step.step_id not in required_outputs
            )
            expanded: list[PathState] = []
            failed_paths: list[PathState] = []

            for path in paths:
                if query_count >= self.config.max_subqueries:
                    break

                try:
                    original_query = self.instantiate(
                        step,
                        path.answers,
                    )
                except ValueError:
                    continue

                structural_bundle = ""
                textual_bundle = ""
                seen_structural_ids = set(
                    path.structural_ids
                )
                seen_textual_ids = set(
                    path.textual_ids
                )
                attempts: list[dict[str, Any]] = []

                initial_attempt = await execute_attempt(
                    query=original_query,
                    step=step,
                    path=path,
                    structural_bundle=structural_bundle,
                    textual_bundle=textual_bundle,
                    seen_structural_ids=(
                        seen_structural_ids
                    ),
                    seen_textual_ids=(
                        seen_textual_ids
                    ),
                    extract_intermediate=(
                        not is_terminal
                    ),
                )

                if initial_attempt is None:
                    stop_reason = (
                        "query_budget_reached"
                    )
                    break

                attempts.append(initial_attempt)
                structural_bundle = initial_attempt[
                    "structural_bundle"
                ]
                textual_bundle = initial_attempt[
                    "textual_bundle"
                ]
                seen_structural_ids = initial_attempt[
                    "seen_structural_ids"
                ]
                seen_textual_ids = initial_attempt[
                    "seen_textual_ids"
                ]

                initial_candidates = (
                    initial_attempt["candidates"]
                )
                low_novelty = (
                    initial_attempt[
                        "overall_novelty"
                    ]
                    < self.config.novelty_threshold
                )

                retry_reasons: list[str] = []
                if (
                    not is_terminal
                    and not initial_candidates
                ):
                    if initial_attempt[
                        "rejected_candidates"
                    ]:
                        retry_reasons.append(
                            "all_candidates_rejected"
                        )
                    else:
                        retry_reasons.append(
                            "no_candidate"
                        )
                if low_novelty:
                    retry_reasons.append(
                        "low_novelty"
                    )
                retry_reason = "+".join(
                    retry_reasons
                )

                should_retry = bool(retry_reasons)
                if (
                    should_retry
                    and query_count
                    < self.config.max_subqueries
                ):
                    try:
                        alternatives = (
                            await self.generate_alternative_queries(
                                original_question=question,
                                failed_query=original_query,
                                step=step,
                                known_entities=(
                                    structure.known_entities
                                ),
                                answers=path.answers,
                                retry_reason=retry_reason,
                            )
                        )
                    except Exception:
                        fallback = (
                            self._deterministic_alternative_query(
                                original_query,
                                step,
                                structure.known_entities,
                                path.answers,
                            )
                        )
                        alternatives = (
                            self._deduplicate_queries(
                                original_query,
                                [fallback],
                            )
                        )

                    retry_limit = min(
                        self.config.max_retry_queries_per_step,
                        len(alternatives),
                    )

                    for alternative_query in (
                        alternatives[:retry_limit]
                    ):
                        if (
                            query_count
                            >= self.config.max_subqueries
                        ):
                            break

                        attempt = await execute_attempt(
                            query=alternative_query,
                            step=step,
                            path=path,
                            structural_bundle=(
                                structural_bundle
                            ),
                            textual_bundle=(
                                textual_bundle
                            ),
                            seen_structural_ids=(
                                seen_structural_ids
                            ),
                            seen_textual_ids=(
                                seen_textual_ids
                            ),
                            extract_intermediate=(
                                not is_terminal
                            ),
                        )
                        if attempt is None:
                            break

                        attempts.append(attempt)
                        structural_bundle = attempt[
                            "structural_bundle"
                        ]
                        textual_bundle = attempt[
                            "textual_bundle"
                        ]
                        seen_structural_ids = attempt[
                            "seen_structural_ids"
                        ]
                        seen_textual_ids = attempt[
                            "seen_textual_ids"
                        ]

                        if (
                            not is_terminal
                            and attempt["candidates"]
                            and (
                                "no_candidate"
                                in retry_reasons
                                or "all_candidates_rejected"
                                in retry_reasons
                            )
                        ):
                            break

                        if is_terminal:
                            break

                merged_structural_context = (
                    merge_context(
                        path.structural_context,
                        structural_bundle,
                    )
                )
                merged_textual_context = merge_context(
                    path.textual_context,
                    textual_bundle,
                )
                merged_structural_ids = set(
                    seen_structural_ids
                )
                merged_textual_ids = set(
                    seen_textual_ids
                )
                attempted_queries = [
                    attempt["query"]
                    for attempt in attempts
                ]
                all_rejected_candidates: list[
                    Candidate
                ] = []
                for attempt in attempts:
                    all_rejected_candidates.extend(
                        attempt.get(
                            "rejected_candidates",
                            [],
                        )
                    )
                all_anchor_entities = list(
                    dict.fromkeys(
                        anchor
                        for attempt in attempts
                        for anchor in attempt.get(
                            "anchor_entities",
                            [],
                        )
                    )
                )

                if is_terminal:
                    best_attempt = max(
                        attempts,
                        key=lambda attempt: (
                            attempt[
                                "overall_novelty"
                            ],
                            attempt["utility"],
                        ),
                    )
                    terminal_score = (
                        self.path_step_score(
                            candidate_confidence=0.50,
                            retrieval=best_attempt[
                                "retrieval"
                            ],
                            novelty_score=best_attempt[
                                "overall_novelty"
                            ],
                        )
                    )
                    selected_query = best_attempt[
                        "query"
                    ]
                    try:
                        terminal_verdict = (
                            await self.verify_terminal_relation_coverage(
                                original_question=question,
                                query=selected_query,
                                step=step,
                                retrieval=best_attempt["retrieval"],
                                anchor_entities=best_attempt.get(
                                    "anchor_entities", []
                                ),
                            )
                        )
                    except Exception:
                        terminal_verdict = {
                            "structural_supported": False,
                            "textual_supported": False,
                        }
                    trace = StepTrace(
                        step_id=step.step_id,
                        query=selected_query,
                        rewritten_query=(
                            selected_query
                            if normalize(
                                selected_query
                            )
                            != normalize(
                                original_query
                            )
                            else ""
                        ),
                        relation=step.relation,
                        depends_on=step.depends_on,
                        is_terminal=True,
                        structural_novelty=(
                            best_attempt[
                                "structural_novelty"
                            ]
                        ),
                        textual_novelty=(
                            best_attempt[
                                "textual_novelty"
                            ]
                        ),
                        overall_novelty=(
                            best_attempt[
                                "overall_novelty"
                            ]
                        ),
                        selected_entity="",
                        selected_score=(
                            terminal_score
                        ),
                        candidates=[],
                        attempted_queries=(
                            attempted_queries
                        ),
                        retry_reason=retry_reason,
                        rejected_candidates=[],
                        anchor_entities=(
                            best_attempt.get(
                                "anchor_entities",
                                [],
                            )
                        ),
                        verification_enabled=(
                            self.config.enable_relation_verification
                        ),
                        structural_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="structural",
                                step_id=step.step_id,
                            )
                        ),
                        textual_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="textual",
                                step_id=step.step_id,
                            )
                        ),
                        structural_step_supported=bool(
                            terminal_verdict.get(
                                "structural_supported", False
                            )
                        ),
                        textual_step_supported=bool(
                            terminal_verdict.get(
                                "textual_supported", False
                            )
                        ),
                    )
                    expanded.append(
                        PathState(
                            answers=dict(
                                path.answers
                            ),
                            structural_context=(
                                merged_structural_context
                            ),
                            textual_context=(
                                merged_textual_context
                            ),
                            structural_ids=(
                                merged_structural_ids
                            ),
                            textual_ids=(
                                merged_textual_ids
                            ),
                            scores=[
                                *path.scores,
                                terminal_score,
                            ],
                            traces=[
                                *path.traces,
                                trace,
                            ],
                        )
                    )
                    continue

                valid_attempts = [
                    attempt
                    for attempt in attempts
                    if attempt["candidates"]
                ]

                if not valid_attempts:
                    last_attempt = attempts[-1]
                    failed_trace = StepTrace(
                        step_id=step.step_id,
                        query=last_attempt["query"],
                        rewritten_query=(
                            last_attempt["query"]
                            if normalize(
                                last_attempt["query"]
                            )
                            != normalize(
                                original_query
                            )
                            else ""
                        ),
                        relation=step.relation,
                        depends_on=step.depends_on,
                        is_terminal=False,
                        structural_novelty=(
                            last_attempt[
                                "structural_novelty"
                            ]
                        ),
                        textual_novelty=(
                            last_attempt[
                                "textual_novelty"
                            ]
                        ),
                        overall_novelty=(
                            last_attempt[
                                "overall_novelty"
                            ]
                        ),
                        selected_entity="",
                        selected_score=0.0,
                        candidates=[],
                        attempted_queries=(
                            attempted_queries
                        ),
                        retry_reason=(
                            retry_reason
                            or "no_verified_candidate"
                        ),
                        rejected_candidates=(
                            all_rejected_candidates
                        ),
                        anchor_entities=(
                            all_anchor_entities
                        ),
                        verification_enabled=(
                            self.config.enable_relation_verification
                        ),
                        structural_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="structural",
                                step_id=step.step_id,
                            )
                        ),
                        textual_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="textual",
                                step_id=step.step_id,
                            )
                        ),
                        structural_step_supported=False,
                        textual_step_supported=False,
                    )
                    failed_paths.append(
                        PathState(
                            answers=dict(
                                path.answers
                            ),
                            structural_context=(
                                merged_structural_context
                            ),
                            textual_context=(
                                merged_textual_context
                            ),
                            structural_ids=(
                                merged_structural_ids
                            ),
                            textual_ids=(
                                merged_textual_ids
                            ),
                            scores=[
                                *path.scores,
                                0.0,
                            ],
                            traces=[
                                *path.traces,
                                failed_trace,
                            ],
                        )
                    )
                    continue

                best_attempt = max(
                    valid_attempts,
                    key=lambda attempt: (
                        attempt["utility"]
                    ),
                )
                candidates = best_attempt[
                    "candidates"
                ]
                selected_query = best_attempt[
                    "query"
                ]

                for candidate in candidates:
                    score = self.path_step_score(
                        candidate_confidence=(
                            candidate.confidence
                        ),
                        retrieval=best_attempt[
                            "retrieval"
                        ],
                        novelty_score=best_attempt[
                            "overall_novelty"
                        ],
                    )
                    answers = dict(path.answers)
                    answers[step.step_id] = (
                        candidate.entity
                    )

                    trace = StepTrace(
                        step_id=step.step_id,
                        query=selected_query,
                        rewritten_query=(
                            selected_query
                            if normalize(
                                selected_query
                            )
                            != normalize(
                                original_query
                            )
                            else ""
                        ),
                        relation=step.relation,
                        depends_on=step.depends_on,
                        is_terminal=False,
                        structural_novelty=(
                            best_attempt[
                                "structural_novelty"
                            ]
                        ),
                        textual_novelty=(
                            best_attempt[
                                "textual_novelty"
                            ]
                        ),
                        overall_novelty=(
                            best_attempt[
                                "overall_novelty"
                            ]
                        ),
                        selected_entity=(
                            candidate.entity
                        ),
                        selected_score=score,
                        candidates=candidates,
                        attempted_queries=(
                            attempted_queries
                        ),
                        retry_reason=retry_reason,
                        rejected_candidates=(
                            best_attempt.get(
                                "rejected_candidates",
                                [],
                            )
                        ),
                        anchor_entities=(
                            best_attempt.get(
                                "anchor_entities",
                                [],
                            )
                        ),
                        verification_enabled=(
                            self.config.enable_relation_verification
                        ),
                        structural_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="structural",
                                step_id=step.step_id,
                            )
                        ),
                        textual_retrieval_records=(
                            attempts_channel_retrieval_records(
                                attempts,
                                channel="textual",
                                step_id=step.step_id,
                            )
                        ),
                        structural_step_supported=(
                            candidate.structural_relation_supported
                        ),
                        textual_step_supported=(
                            candidate.textual_relation_supported
                        ),
                    )
                    expanded.append(
                        PathState(
                            answers=answers,
                            structural_context=(
                                merged_structural_context
                            ),
                            textual_context=(
                                merged_textual_context
                            ),
                            structural_ids=(
                                merged_structural_ids
                            ),
                            textual_ids=(
                                merged_textual_ids
                            ),
                            scores=[
                                *path.scores,
                                score,
                            ],
                            traces=[
                                *path.traces,
                                trace,
                            ],
                        )
                    )

            if not expanded:
                if (
                    stop_reason
                    != "query_budget_reached"
                ):
                    stop_reason = (
                        "no_valid_intermediate_candidate"
                    )
                if failed_paths:
                    failed_paths.sort(
                        key=lambda path: path.score,
                        reverse=True,
                    )
                    paths = failed_paths[
                        : self.config.beam_width
                    ]
                break

            expanded.sort(
                key=lambda path: path.score,
                reverse=True,
            )
            paths = expanded[
                : self.config.beam_width
            ]

        paths.sort(
            key=lambda path: path.score,
            reverse=True,
        )
        best = paths[0]

        # 完成检查同时认可“由原问题上下文覆盖”的步骤
        # 和“实际执行子查询”的步骤。
        if stop_reason == "plan_completed":
            expected_step_ids = {
                step.step_id
                for step in plan
            }
            traced_step_ids = {
                trace.step_id
                for trace in best.traces
            }
            resolved_step_ids = (
                traced_step_ids
                | covered_step_ids
            )
            missing_steps = sorted(
                expected_step_ids
                - resolved_step_ids
            )
            missing_required_outputs = sorted(
                step_id
                for step_id in required_outputs
                if not str(
                    best.answers.get(
                        step_id,
                        "",
                    )
                ).strip()
            )
            if (
                missing_steps
                or missing_required_outputs
            ):
                stop_reason = (
                    "incomplete_plan_execution"
                )

        return RFDQDResult(
            question=question,
            need_decomposition=True,
            question_structure=structure,
            plan=plan,
            executed_subqueries=query_count,
            stop_reason=stop_reason,
            selected_path_score=best.score,
            intermediate_answers=best.answers,
            traces=best.traces,
            final_structural_context=(
                best.structural_context
            ),
            final_textual_context=(
                best.textual_context
            ),
            structural_evidence_ids=sorted(
                best.structural_ids
            ),
            textual_evidence_ids=sorted(
                best.textual_ids
            ),
            alternative_paths=[
                {
                    "score": path.score,
                    "intermediate_answers": (
                        path.answers
                    ),
                }
                for path in paths[1:]
            ],
            plan_validation=plan_validation,
            original_context_coverage=(
                coverage_info
            ),
            initial_structural_retrieval_records=(
                initial_structural_records
            ),
            initial_textual_retrieval_records=(
                initial_textual_records
            ),
            structural_supported_step_ids=sorted(
                coverage_channel_supported_step_ids(
                    coverage_info,
                    "structural",
                )
                | {
                    trace.step_id
                    for trace in best.traces
                    if trace.structural_step_supported
                }
            ),
            textual_supported_step_ids=sorted(
                coverage_channel_supported_step_ids(
                    coverage_info,
                    "textual",
                )
                | {
                    trace.step_id
                    for trace in best.traces
                    if trace.textual_step_supported
                }
            ),
        )


# =============================================================================
# 检索适配器
# =============================================================================


def build_dual_retriever(
    structural_retrieve: Callable[
        [str],
        ChannelRetrieval
        | str
        | Awaitable[
            ChannelRetrieval | str
        ],
    ],
    textual_retrieve: Callable[
        [str],
        ChannelRetrieval
        | str
        | Awaitable[
            ChannelRetrieval | str
        ],
    ],
) -> DualRetriever:
    async def retrieve_dual(
        query: str,
    ) -> DualRetrieval:
        structural, textual = (
            await asyncio.gather(
                maybe_await(
                    structural_retrieve(query)
                ),
                maybe_await(
                    textual_retrieve(query)
                ),
            )
        )

        if isinstance(structural, str):
            structural = ChannelRetrieval(
                context=structural
            )

        if isinstance(textual, str):
            textual = ChannelRetrieval(
                context=textual
            )

        return DualRetrieval(
            structural=structural,
            textual=textual,
        )

    return retrieve_dual
