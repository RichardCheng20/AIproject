"""
快慢路径路由器（Phase 1）
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    route: str  # "fast" | "slow"
    reason: str
    confidence: float


_SLOW_HINT_PATTERNS = (
    "对比",
    "区别",
    "差异",
    "相比",
    " versus ",
    " vs ",
    "总结",
    "归纳",
    "梳理",
    "列出",
    "有哪些",
    "哪几个",
    "分别",
    "分点",
    "所有项目",
    "每一段",
    "多个经历",
    "详细说说",
    "分几步",
    "一步步",
    "STAR",
    "star",
)


def _rule_based_route(question: str) -> RouteDecision:
    q = question.strip()
    lower = q.lower()
    for p in _SLOW_HINT_PATTERNS:
        if p.lower() in lower or p in q:
            return RouteDecision(route="slow", reason=f"rule_keyword:{p}", confidence=1.0)
    return RouteDecision(route="fast", reason="rule_default", confidence=1.0)


def _parse_llm_route(text: str) -> Optional[tuple[str, float]]:
    match = re.search(
        r'"route"\s*:\s*"(?P<route>fast|slow)".*?"confidence"\s*:\s*(?P<conf>0(?:\.\d+)?|1(?:\.0+)?)',
        text,
        re.S | re.I,
    )
    if not match:
        return None
    return match.group("route").lower(), float(match.group("conf"))


def decide_route(
    question: str,
    invoke_llm: Callable[[str], Any],
    route_mode: str,
    llm_confidence_threshold: float,
    agent_enabled: bool,
) -> RouteDecision:
    """
    判定检索路径。agent_enabled=False 时固定 fast。
    """
    if not agent_enabled:
        return RouteDecision(route="fast", reason="agent_disabled", confidence=1.0)

    rule = _rule_based_route(question)
    mode = (route_mode or "hybrid").lower()

    if mode == "rule_only":
        return rule

    if mode == "llm":
        raw = invoke_llm(_router_prompt(question))
        text = raw if isinstance(raw, str) else str(raw)
        parsed = _parse_llm_route(text)
        if not parsed:
            logger.debug("LLM 路由解析失败，回退 fast")
            return RouteDecision(route="fast", reason="llm_parse_fallback", confidence=0.0)
        route, conf = parsed
        if conf < llm_confidence_threshold:
            return RouteDecision(route="fast", reason="llm_low_confidence", confidence=conf)
        return RouteDecision(route=route, reason="llm", confidence=conf)

    # hybrid
    if rule.route == "slow":
        return rule

    raw = invoke_llm(_router_prompt(question))
    text = raw if isinstance(raw, str) else str(raw)
    parsed = _parse_llm_route(text)
    if not parsed:
        return rule
    route, conf = parsed
    if route == "slow" and conf >= llm_confidence_threshold:
        return RouteDecision(route="slow", reason="llm_override_fast_rule", confidence=conf)
    return rule


def _router_prompt(question: str) -> str:
    return (
        "你是检索路径路由器。fast 表示单跳、常规的简历/知识问答；"
        "slow 表示需要多角度检索合并的问题（对比、列举多条经历、总结多篇材料、复杂拆解）。\n"
        '只输出 JSON：{"route":"fast|slow","confidence":0-1}\n'
        f"用户问题：{question}"
    )
