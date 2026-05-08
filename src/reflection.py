"""
生成后反思：答案与上下文一致性检查（Phase 1）
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_FIXED_TAIL_PHRASES = (
    "根据当前知识库内容，无法回答该问题",
    "无法回答该问题",
)


def answer_context_term_overlap(answer: str, context: str) -> float:
    """基于中英数字片段的重合比例（相对于答案侧词项）。"""
    def terms(text: str):
        return {
            t.lower()
            for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text)
            if len(t) >= 2
        }

    a = answer.strip()
    for ph in _FIXED_TAIL_PHRASES:
        if ph in a:
            return 1.0

    qa = terms(a)
    qc = terms(context)
    if not qa:
        return 1.0
    return len(qa & qc) / len(qa)


def llm_supports_answer(invoke_llm: Callable[[str], Any], answer: str, context: str) -> bool:
    """单次 LLM 校验：答案是否可由上下文支持。"""
    prompt = (
        "你是审计员。判断下列「答案」中的事实陈述是否都能由「上下文」直接支持，"
        "不允许推断或常识补全。若存在无法在上下文中找到依据的具体陈述，输出 supported:false。\n"
        '只输出 JSON：{"supported":true|false}\n\n'
        f"上下文：\n{context[:6000]}\n\n答案：\n{answer[:4000]}"
    )
    try:
        raw = invoke_llm(prompt)
        text = raw if isinstance(raw, str) else str(raw)
        if '"supported"' not in text.lower():
            return True
        return bool(re.search(r'"supported"\s*:\s*true\b', text, re.I))
    except Exception as e:
        logger.warning(f"反思 LLM 校验异常，跳过重写：{e}")
        return True


def should_regenerate(
    invoke_llm: Callable[[str], Any],
    answer: str,
    context: str,
    mode: str,
    min_overlap: float,
) -> bool:
    """
    是否需要重写生成。mode: heuristic | llm | hybrid
    """
    mode = (mode or "hybrid").lower()
    overlap = answer_context_term_overlap(answer, context)

    if mode == "heuristic":
        return overlap < min_overlap

    if mode == "llm":
        return not llm_supports_answer(invoke_llm, answer, context)

    # hybrid
    if overlap < min_overlap:
        return True
    return not llm_supports_answer(invoke_llm, answer, context)
