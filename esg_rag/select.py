import argparse
import json
import os
import re
from difflib import SequenceMatcher
from collections import Counter

from openai import OpenAI


# ============================================================
# 文件路径与命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-channel cross-evidence selection with optional conditional fusion."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("context_mode", nargs="?", default="rf-dqd", choices=["rf-dqd", "direct"])
    parser.add_argument("decision_mode", nargs="?", default="standard", choices=["standard", "score-only"])
    parser.add_argument("--lambda-answer", type=float, default=1.0)
    parser.add_argument("--lambda-own", type=float, default=0.5)
    parser.add_argument("--lambda-cross", type=float, default=0.2)
    parser.add_argument("--lambda-retrieval", type=float, default=0.3)
    parser.add_argument("--score-gap-threshold", type=float, default=0.15)
    parser.add_argument("--complement-threshold", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output", default="")
    return parser.parse_args()

ARGS = parse_args()

ROOT_PATH = os.path.normpath(ARGS.dataset_root)
CONTEXT_MODE = ARGS.context_mode
DECISION_MODE = ARGS.decision_mode

DATASET_NAME = os.path.basename(ROOT_PATH)
OUTPUT_DIR = os.path.join(ROOT_PATH, "output")

CONTEXT_SUFFIX = "-rf-dqd" if CONTEXT_MODE == "rf-dqd" else ""
ANSWER_VARIANT = "rf-dqd" if CONTEXT_MODE == "rf-dqd" else "direct"
if DECISION_MODE == "score-only":
    ANSWER_VARIANT = f"score-only-selection-{ANSWER_VARIANT}"

STRUCTURED_ANSWER_FILE = os.path.join(
    OUTPUT_DIR,
    f"answer-gpt4omini-aligned-{DATASET_NAME}-"
    f"skeleton-1.0{CONTEXT_SUFFIX}-raw.json",
)
KEYWORD_ANSWER_FILE = os.path.join(
    OUTPUT_DIR,
    f"answer-gpt4omini-aligned-{DATASET_NAME}-"
    f"keyword-0.0{CONTEXT_SUFFIX}-raw.json",
)
STRUCTURED_CONTEXT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{DATASET_NAME}-skeleton-1.0{CONTEXT_SUFFIX}.json",
)
KEYWORD_CONTEXT_FILE = os.path.join(
    OUTPUT_DIR,
    f"{DATASET_NAME}-keyword-0.0{CONTEXT_SUFFIX}.json",
)
GOLD_FILE = os.path.join(
    ROOT_PATH,
    "qa-pairs",
    "qa-pairs.json",
)
OUT_FILE = (
    os.path.normpath(ARGS.output)
    if ARGS.output
    else os.path.join(
        OUTPUT_DIR,
        "answer-gpt4omini-aligned-dual-channel-"
        f"cross-evidence-fusion-{ANSWER_VARIANT}-raw.json",
    )
)
# ============================================================
# 超参数
# ============================================================

# 候选答案评分权重
LAMBDA_ANSWER = ARGS.lambda_answer
LAMBDA_OWN = ARGS.lambda_own
LAMBDA_CROSS = ARGS.lambda_cross
LAMBDA_RETRIEVAL = ARGS.lambda_retrieval

# 两个候选答案得分差距阈值
SCORE_GAP_THRESHOLD = ARGS.score_gap_threshold

# 证据互补度阈值
COMPLEMENT_THRESHOLD = ARGS.complement_threshold

# 是否启用条件证据融合
ENABLE_EVIDENCE_FUSION = True

# 融合生成模型
FUSION_MODEL = "gpt-4o-mini"

# 每个通道送给融合模型的最大字符数
MAX_CONTEXT_CHARS = 6000

# 每个通道最多保留的证据片段数
MAX_SNIPPETS = 8


# OpenAI 客户端延迟初始化
_openai_client = None


def get_openai_client():
    global _openai_client

    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "找不到 OPENAI_API_KEY，请先在 CMD 中设置："
                "set OPENAI_API_KEY=你的key"
            )

        _openai_client = OpenAI(api_key=api_key)

    return _openai_client


# ============================================================
# 基础工具函数
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split())
    return text


def tokenize(text):
    return [
        word
        for word in normalize(text).split()
        if len(word) > 2
    ]


def is_bad_answer(answer):
    answer = str(answer).strip().lower()

    bad_phrases = [
        "",
        "insufficient information",
        "not enough information",
        "cannot determine",
        "cannot be determined",
        "unknown",
        "i don't know",
        "no answer",
        "not provided",
        "not mentioned",
        "unable to determine",
    ]

    return (
        answer in bad_phrases
        or "insufficient information" in answer
    )


def answer_similarity(answer_a, answer_b):
    """
    仅作为结果分析指标，不再参与两个候选答案的排序。
    """
    answer_a = normalize(answer_a)
    answer_b = normalize(answer_b)

    if not answer_a or not answer_b:
        return 0.0

    if answer_a == answer_b:
        return 1.0

    if answer_a in answer_b or answer_b in answer_a:
        return 0.85

    return SequenceMatcher(
        None,
        answer_a,
        answer_b
    ).ratio()


def answers_agree(answer_s, answer_t):
    """
    判断两个通道的答案在规范化后是否完全一致。
    """
    norm_s = normalize(answer_s)
    norm_t = normalize(answer_t)

    return bool(norm_s) and norm_s == norm_t


def choose_cleaner_answer(answer_s, answer_t):
    """
    两个答案规范化后一致时，选择更短、更规范的原始形式。
    """
    candidates = [
        str(answer).strip()
        for answer in [answer_s, answer_t]
        if answer and not is_bad_answer(answer)
    ]

    if not candidates:
        return "Insufficient information."

    return min(
        candidates,
        key=lambda answer: (
            len(normalize(answer).split()),
            len(answer)
        )
    )


# ============================================================
# 检索置信度
# ============================================================

def query_context_similarity(question, context):
    question_tokens = tokenize(question)
    context_tokens = set(tokenize(context))

    if not question_tokens:
        return 0.0

    matched = sum(
        1
        for token in question_tokens
        if token in context_tokens
    )

    return matched / len(question_tokens)


def evidence_density(context):
    """
    根据上下文中重复出现的关键词估计证据密度。
    """
    tokens = tokenize(context)

    if not tokens:
        return 0.0

    token_counter = Counter(tokens)

    evidence_words = [
        word
        for word, count in token_counter.items()
        if count >= 2
    ]

    return min(
        len(evidence_words) / 100.0,
        1.0
    )


def context_agreement(context_a, context_b):
    tokens_a = set(tokenize(context_a))
    tokens_b = set(tokenize(context_b))

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)

    return intersection / union


def retrieval_confidence(question, context, other_context):
    """
    通道检索置信度：

    R_i = 0.5 × QueryContextSimilarity
        + 0.3 × EvidenceDensity
        + 0.2 × ContextAgreement
    """
    query_similarity = query_context_similarity(
        question,
        context
    )

    density = evidence_density(context)

    agreement = context_agreement(
        context,
        other_context
    )

    return (
        0.5 * query_similarity
        + 0.3 * density
        + 0.2 * agreement
    )


# ============================================================
# 答案与证据支持
# ============================================================

def evidence_score(answer, context):
    """
    E(a, C)：候选答案是否被上下文支持。
    """
    normalized_answer = normalize(answer)
    normalized_context = normalize(context)

    if not normalized_answer or is_bad_answer(answer):
        return 0.0

    # 完整答案直接出现在上下文中
    if normalized_answer in normalized_context:
        return 1.0

    answer_tokens = normalized_answer.split()
    context_tokens = set(normalized_context.split())

    if not answer_tokens:
        return 0.0

    matched = sum(
        1
        for token in answer_tokens
        if token in context_tokens
    )

    if len(answer_tokens) == 1:
        return 1.0 if matched == 1 else 0.0

    return matched / len(answer_tokens)


def answer_base_score(answer):
    """
    B(a)：候选答案的基础质量分。

    主要用于：
    1. 淘汰无法回答类输出；
    2. 鼓励简洁答案；
    3. 惩罚解释性长答案。
    """
    if is_bad_answer(answer):
        return 0.0

    answer = str(answer).strip()
    answer_lower = answer.lower()
    words = answer.split()

    score = 1.0

    # HotpotQA 是短答案任务，1～6 个词比较合理
    if len(words) <= 6:
        score += 0.10

    if len(words) > 15:
        score -= 0.35

    if answer_lower.startswith("based on"):
        score -= 0.30

    if "." in answer and len(words) > 8:
        score -= 0.20

    return max(
        0.0,
        min(score, 1.2)
    )


def evidence_complementarity(
    own_support_s,
    cross_support_s,
    own_support_t,
    cross_support_t
):
    """
    衡量结构通道和文本通道的证据互补性。

    如果结构答案主要被结构上下文支持，
    文本答案主要被文本上下文支持，
    说明两个通道可能具有互补证据。
    """
    structural_unique = max(
        0.0,
        own_support_s - cross_support_s
    )

    textual_unique = max(
        0.0,
        own_support_t - cross_support_t
    )

    return 0.5 * (
        structural_unique + textual_unique
    )


# ============================================================
# 证据片段筛选
# ============================================================

def sentence_relevance(sentence, question):
    question_tokens = set(tokenize(question))
    sentence_tokens = set(tokenize(sentence))

    if not question_tokens:
        return 0.0

    return len(
        question_tokens & sentence_tokens
    ) / len(question_tokens)


def select_relevant_snippets(
    question,
    context,
    max_snippets=MAX_SNIPPETS,
    max_chars=MAX_CONTEXT_CHARS
):
    """
    从长上下文中选取与问题更相关的证据片段，
    避免把完整的数万字符上下文送给融合模型。
    """
    if not context:
        return ""

    segments = re.split(
        r"\n+|(?<=[.!?])\s+",
        str(context)
    )

    scored_segments = []

    for segment in segments:
        segment = segment.strip()

        if len(segment) < 20:
            continue

        relevance = sentence_relevance(
            segment,
            question
        )

        scored_segments.append(
            (relevance, segment)
        )

    scored_segments.sort(
        key=lambda item: item[0],
        reverse=True
    )

    selected = []
    total_chars = 0

    for relevance, segment in scored_segments:
        if len(selected) >= max_snippets:
            break

        if total_chars + len(segment) > max_chars:
            continue

        selected.append(segment)
        total_chars += len(segment)

    if not selected:
        return str(context)[:max_chars]

    return "\n".join(selected)


# ============================================================
# 条件证据融合
# ============================================================

def generate_fused_answer(
    question,
    context_s,
    context_t,
    answer_s,
    answer_t
):
    """
    当两个通道得分接近且证据互补时，
    融合两个通道的高质量证据重新生成答案。
    """
    structured_evidence = select_relevant_snippets(
        question,
        context_s
    )

    textual_evidence = select_relevant_snippets(
        question,
        context_t
    )

    prompt = f"""
You are answering a multi-hop question using two complementary evidence channels.

Question:
{question}

Candidate answer from the structured evidence channel:
{answer_s}

Candidate answer from the textual evidence channel:
{answer_t}

Structured evidence:
{structured_evidence}

Textual evidence:
{textual_evidence}

Resolve the disagreement using only the supplied evidence.

Output only the final short answer as a single phrase.
Do not provide explanations or reasoning.
If the evidence is insufficient, output exactly:
Insufficient information.
"""

    try:
        client = get_openai_client()

        response = client.chat.completions.create(
            model=FUSION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You resolve conflicting candidate "
                        "answers using structured and textual evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=100,
        )

        fused_answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        if not fused_answer:
            return ""

        return fused_answer

    except Exception as error:
        print("Fusion error:", error)
        return ""


# ============================================================
# 加载数据
# ============================================================

structured_answers = load_json(STRUCTURED_ANSWER_FILE)
keyword_answers = load_json(KEYWORD_ANSWER_FILE)

structured_contexts = load_json(STRUCTURED_CONTEXT_FILE)
keyword_contexts = load_json(KEYWORD_CONTEXT_FILE)

if not 1 <= ARGS.limit <= 500:
    raise ValueError("--limit必须在1到500之间")

gold_data = load_json(GOLD_FILE)[:ARGS.limit]


def validate_complete(label, records, expected_ids):
    ids = [str(item.get("id", "")) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}包含重复ID")
    if ids != expected_ids:
        raise ValueError(
            f"{label}的ID或顺序与前{ARGS.limit}个原始问题不完全一致"
        )
    failed = [
        item for item in records
        if item.get("status") == "error"
    ]
    if failed:
        raise ValueError(f"{label}仍有{len(failed)}条失败记录")


expected_ids = [str(item.get("id", "")) for item in gold_data]
if (
    len(expected_ids) != ARGS.limit
    or len(expected_ids) != len(set(expected_ids))
):
    raise ValueError(
        f"原始问题必须是无重复ID的前{ARGS.limit}条"
    )

validate_complete("Structured答案", structured_answers, expected_ids)
validate_complete("Textual答案", keyword_answers, expected_ids)
validate_complete("Structured上下文", structured_contexts, expected_ids)
validate_complete("Textual上下文", keyword_contexts, expected_ids)


# 统一将 id 转成字符串，防止整数和字符串 id 不匹配
question_map = {
    str(item["id"]): item["question"]
    for item in gold_data
}

keyword_answer_map = {
    str(item["id"]): item
    for item in keyword_answers
}

ket_context_map = {
    str(item["id"]): item.get("context", "")
    for item in structured_contexts
}

keyword_context_map = {
    str(item["id"]): item.get("context", "")
    for item in keyword_contexts
}


# ============================================================
# 双通道决策
# ============================================================

final_data = []
mode_counter = Counter()

for item in structured_answers:
    original_qid = item["id"]
    qid = str(original_qid)

    question = question_map.get(qid, "")

    answer_s = item.get("answer", "")
    answer_t = keyword_answer_map.get(
        qid,
        {}
    ).get("answer", "")

    context_s = ket_context_map.get(qid, "")
    context_t = keyword_context_map.get(qid, "")

    # --------------------------------------------------------
    # 1. 本通道证据支持
    # --------------------------------------------------------

    own_support_s = evidence_score(
        answer_s,
        context_s
    )

    own_support_t = evidence_score(
        answer_t,
        context_t
    )

    # --------------------------------------------------------
    # 2. 跨通道证据支持
    # --------------------------------------------------------

    cross_support_s = evidence_score(
        answer_s,
        context_t
    )

    cross_support_t = evidence_score(
        answer_t,
        context_s
    )

    # --------------------------------------------------------
    # 3. 检索置信度
    # --------------------------------------------------------

    reliability_s = retrieval_confidence(
        question,
        context_s,
        context_t
    )

    reliability_t = retrieval_confidence(
        question,
        context_t,
        context_s
    )

    # --------------------------------------------------------
    # 4. 基础答案质量
    # --------------------------------------------------------

    base_s = answer_base_score(answer_s)
    base_t = answer_base_score(answer_t)

    # --------------------------------------------------------
    # 5. 新评分公式
    # --------------------------------------------------------

    score_s = max(
        0.0,
        min(
            1.0,
            LAMBDA_ANSWER * base_s
            + LAMBDA_OWN * own_support_s
            + LAMBDA_CROSS * cross_support_s
            + LAMBDA_RETRIEVAL * reliability_s
        ),
    )

    score_t = max(
        0.0,
        min(
            1.0,
            LAMBDA_ANSWER * base_t
            + LAMBDA_OWN * own_support_t
            + LAMBDA_CROSS * cross_support_t
            + LAMBDA_RETRIEVAL * reliability_t
        ),
    )

    score_gap = abs(score_s - score_t)

    complementarity = evidence_complementarity(
        own_support_s,
        cross_support_s,
        own_support_t,
        cross_support_t
    )

    answer_sim = answer_similarity(
        answer_s,
        answer_t
    )

    # --------------------------------------------------------
    # 6. 三级决策
    # --------------------------------------------------------

    if DECISION_MODE == "score-only":
        if score_s >= score_t:
            final_answer = answer_s
            selection_mode = "score_only_select_structured"
        else:
            final_answer = answer_t
            selection_mode = "score_only_select_textual"

    # 情况一：两个答案一致
    elif answers_agree(answer_s, answer_t):
        final_answer = choose_cleaner_answer(
            answer_s,
            answer_t
        )

        selection_mode = "agreement"

    # 情况二：得分差距明显，选择高分答案
    elif score_gap >= SCORE_GAP_THRESHOLD:
        if score_s >= score_t:
            final_answer = answer_s
            selection_mode = "select_structured"
        else:
            final_answer = answer_t
            selection_mode = "select_textual"

    # 情况三：得分接近且证据互补，融合证据
    elif (
        ENABLE_EVIDENCE_FUSION
        and complementarity >= COMPLEMENT_THRESHOLD
    ):
        fused_answer = generate_fused_answer(
            question=question,
            context_s=context_s,
            context_t=context_t,
            answer_s=answer_s,
            answer_t=answer_t,
        )

        if (
            fused_answer
            and not is_bad_answer(fused_answer)
        ):
            final_answer = fused_answer
            selection_mode = "evidence_fusion"

        # 融合失败，回退到高分候选答案
        elif score_s >= score_t:
            final_answer = answer_s
            selection_mode = (
                "fusion_failed_select_structured"
            )

        else:
            final_answer = answer_t
            selection_mode = (
                "fusion_failed_select_textual"
            )

    # 得分接近，但证据互补性不明显
    else:
        if score_s >= score_t:
            final_answer = answer_s
            selection_mode = (
                "close_score_select_structured"
            )
        else:
            final_answer = answer_t
            selection_mode = (
                "close_score_select_textual"
            )

    mode_counter[selection_mode] += 1

    # --------------------------------------------------------
    # 7. 保存详细结果
    # --------------------------------------------------------

    new_item = {
        "id": original_qid,
        "answer": final_answer,
        "selected": selection_mode,

        "structured_answer": answer_s,
        "textual_answer": answer_t,

        "structured_base_score": round(base_s, 4),
        "textual_base_score": round(base_t, 4),

        "structured_own_support": round(
            own_support_s,
            4
        ),
        "structured_cross_support": round(
            cross_support_s,
            4
        ),

        "textual_own_support": round(
            own_support_t,
            4
        ),
        "textual_cross_support": round(
            cross_support_t,
            4
        ),

        "structured_retrieval_confidence": round(
            reliability_s,
            4
        ),
        "textual_retrieval_confidence": round(
            reliability_t,
            4
        ),

        "weights": {
            "answer": LAMBDA_ANSWER,
            "own": LAMBDA_OWN,
            "cross": LAMBDA_CROSS,
            "retrieval": LAMBDA_RETRIEVAL,
        },

        "structured_score": round(score_s, 4),
        "textual_score": round(score_t, 4),

        "score_gap": round(score_gap, 4),
        "answer_similarity": round(answer_sim, 4),
        "complementarity": round(
            complementarity,
            4
        ),
    }

    final_data.append(new_item)


# ============================================================
# 保存与输出统计
# ============================================================

temporary_output = OUT_FILE + ".tmp"
with open(
    temporary_output,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        final_data,
        f,
        ensure_ascii=False,
        indent=4
    )

os.replace(temporary_output, OUT_FILE)


print("=" * 80)
print("Cross-evidence selection and conditional fusion 完成")
print("权重:", {
    "answer": LAMBDA_ANSWER,
    "own": LAMBDA_OWN,
    "cross": LAMBDA_CROSS,
    "retrieval": LAMBDA_RETRIEVAL,
})
print("输出:", OUT_FILE)
print("Total:", len(final_data))

print("=" * 80)
print("决策模式统计")
print("=" * 80)

for mode, count in mode_counter.most_common():
    print(f"{mode}: {count}")
