"""
RAG管道模块
负责检索增强生成的全流程
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from langchain_ollama import OllamaLLM
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import config
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG管道类，用于构建和管理完整的检索增强生成流程。

    负责整合语言模型、向量数据库检索器以及提示模板，形成一个可执行的问答系统。
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        初始化RAG管道

        Args:
            vector_store_manager: 向量存储管理器实例，用于获取向量数据库及创建检索器
        """
        self.vector_store_manager = vector_store_manager
        self.llm = None
        self.rag_chain = None

        self._initialize_llm()
        self._initialize_rag_chain()

    def _initialize_llm(self):
        """
        初始化大语言模型（LLM）

        根据配置加载Ollama LLM，并设置相关参数如温度、最大输出长度等。
        若初始化失败则记录日志并抛出异常。
        """
        try:
            provider = config.model.provider.lower()
            if provider == "dashscope":
                if not config.dashscope.api_key:
                    raise ValueError("未配置DASHSCOPE_API_KEY，无法使用百炼模型")
                self.llm = ChatTongyi(
                    model=config.model.llm,
                    temperature=config.model.temperature,
                    api_key=config.dashscope.api_key
                )
            else:
                self.llm = OllamaLLM(
                    model=config.model.llm,
                    base_url=config.ollama.base_url,
                    temperature=config.model.temperature,
                    num_predict=config.model.max_tokens,
                    timeout=config.ollama.timeout
                )
            logger.info(f"LLM初始化成功: {config.model.llm}")
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise

    def _initialize_rag_chain(self):
        """
        初始化RAG处理链

        构建从输入问题到最终答案输出的完整处理链条，包括文档检索、上下文格式化、提示构造与模型调用。
        """

        # 定义面试场景专用提示模板
        prompt_template = self._create_prompt_template()

        # 构建RAG链
        self.rag_chain = (
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                    "current_time": RunnablePassthrough()
                }
                | prompt_template
                | self.llm
                | StrOutputParser()
        )

        logger.info("RAG链初始化完成")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        创建适用于面试问答场景的提示模板

        返回一个预设好的聊天提示模板，指导模型如何基于上下文进行专业回答。

        Returns:
            配置好的ChatPromptTemplate对象
        """
        template = """你是一个专业的个人面试助手，正在模拟真实面试场景。

请基于以下提供的上下文信息，准确、专业地回答面试问题。如果上下文信息不足或无法回答，请如实说明，不要编造经历或数据。
你必须优先依据候选人简历内容作答，不允许使用常识或经验进行补全。

上下文信息（来自候选人知识库）：
{context}

当前时间：{current_time}

用户问题：{question}

请按照以下要求提供回答：
1. **准确性**：仅基于上下文作答，不补充未经提供的信息
2. **面试表达**：回答应简洁、结构化，优先给出“背景-行动-结果”或分点表达
3. **专业性**：体现岗位能力与项目思维，重点突出可量化成果
4. **实战性**：必要时给出可执行的改进建议或下一步方案
5. **完整性**：涉及多个维度的问题要分点覆盖
6. **诚实性**：信息不足时明确说明，并指出还需要哪些信息
7. **强约束**：如果上下文中没有与问题直接相关的事实，必须明确回复“根据当前知识库内容，无法回答该问题”，不得进行推测、泛化或常识补全

你的回答："""

        return ChatPromptTemplate.from_template(template)

    def query(self, question: str, session_context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        执行一次完整的RAG查询流程

        包括文档检索、答案生成和结果封装。支持错误处理机制，在发生异常时返回错误信息。

        Args:
            question: 用户提出的问题文本

        Returns:
            包含问题、答案及相关元数据的结果字典
        """
        try:
            logger.info(f"执行查询: {question}")
            effective_question = self._build_effective_question(question, session_context)

            # 先做相似度检索并按阈值过滤，确保仅在有依据时回答
            retrieval_results = self.vector_store_manager.similarity_search(
                query=effective_question,
                k=config.vector_store.recall_k if config.vector_store.enable_reranker else config.vector_store.search_k
            )
            source_policy = self._detect_source_policy(question)
            if config.vector_store.enable_reranker:
                retrieval_results = self._rerank_results(
                    question=question,
                    results=retrieval_results,
                    source_policy=source_policy
                )
            filtered_results = self._filter_results_by_threshold(retrieval_results)
            prioritized_results = self._prioritize_results(filtered_results, source_policy=source_policy)
            prioritized_results = self._apply_source_policy(question, prioritized_results, source_policy=source_policy)
            relevant_docs = [doc for doc, _ in prioritized_results]

            if not relevant_docs:
                answer = "根据当前知识库内容，无法回答该问题。请先补充相关资料后再提问。"
                return {
                    "question": question,
                    "answer": answer,
                    "relevant_sources": [],
                    "source_count": 0,
                    "status": "insufficient_context",
                    "timestamp": datetime.now().isoformat()
                }

            context = self._format_docs(relevant_docs)

            # 生成答案
            answer = self.rag_chain.invoke(
                {
                    "context": context,
                    "question": effective_question,
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            )

            # 构建结果
            result = {
                "question": question,
                "answer": answer,
                "relevant_sources": [
                    {
                        "content": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
                        "source": doc.metadata.get("source", "未知来源"),
                        "source_type": doc.metadata.get("source_type", "other"),
                        "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                        "relevance_score": score
                    }
                    for doc, score in prioritized_results
                ],
                "source_count": len(relevant_docs),
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"查询完成, 找到 {len(relevant_docs)} 个相关文档")
            return result

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return {
                "question": question,
                "answer": f"抱歉，查询过程中出现错误：{str(e)}",
                "relevant_sources": [],
                "source_count": 0,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def stream_query(self, question: str, session_context: Optional[List[Dict[str, str]]] = None):
        """
        流式执行查询，逐段返回响应内容

        适合在Web界面中实现实时显示效果，逐步展示生成的答案片段。

        Args:
            question: 用户提出的问题文本

        Yields:
            响应中的每一个文本块
        """
        try:
            effective_question = self._build_effective_question(question, session_context)
            retrieval_results = self.vector_store_manager.similarity_search(
                query=effective_question,
                k=config.vector_store.recall_k if config.vector_store.enable_reranker else config.vector_store.search_k
            )
            source_policy = self._detect_source_policy(question)
            if config.vector_store.enable_reranker:
                retrieval_results = self._rerank_results(
                    question=question,
                    results=retrieval_results,
                    source_policy=source_policy
                )
            prioritized_results = self._prioritize_results(
                self._filter_results_by_threshold(retrieval_results),
                source_policy=source_policy
            )
            prioritized_results = self._apply_source_policy(
                question,
                prioritized_results,
                source_policy=source_policy
            )
            relevant_docs = [doc for doc, score in prioritized_results]

            if not relevant_docs:
                yield "根据当前知识库内容，无法回答该问题。请先补充相关资料后再提问。"
                return

            context = self._format_docs(relevant_docs)
            for chunk in self.rag_chain.stream(
                {
                    "context": context,
                    "question": effective_question,
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ):
                yield chunk
        except Exception as e:
            yield f"错误：{str(e)}"

    @staticmethod
    def _format_docs(docs: List[Any]) -> str:
        """将检索文档转换为可注入提示词的上下文字符串"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content
            source = doc.metadata.get("source", "未知来源")
            chunk_id = doc.metadata.get("chunk_id", "N/A")
            formatted.append(f"[文档 {i}] 来源: {source} (块ID: {chunk_id})\n{content}\n")
        return "\n".join(formatted)

    @staticmethod
    def _prioritize_results(results: List[Any], source_policy: str = "default") -> List[Any]:
        """
        检索优先级：
        1) resume
        2) additional
        3) 其他
        同优先级内按相关度降序
        """
        if source_policy == "additional":
            priority_order = {"additional": 1, "resume": 2, "other": 99}
        else:
            priority_order = {"resume": 1, "additional": 2, "other": 99}

        def sort_key(item):
            doc, score = item
            source_type = doc.metadata.get("source_type", "other")
            priority = priority_order.get(source_type, 99)
            safe_score = score if score is not None else -1
            return (priority, -safe_score)

        sorted_results = sorted(results, key=sort_key)
        return sorted_results

    @staticmethod
    def _filter_results_by_threshold(results: List[Any]) -> List[Any]:
        """
        双阈值过滤：
        - 简历文档：使用更宽松阈值，避免简历问答被误拒
        - 其他文档：使用默认阈值
        """
        default_threshold = config.vector_store.relevance_threshold
        # Chroma 在部分模型下返回分数整体偏低，启用重排时放宽基础阈值
        if config.vector_store.enable_reranker:
            default_threshold = min(default_threshold, 0.12)
        resume_threshold = min(default_threshold, 0.08)
        min_rerank_score = config.vector_store.min_rerank_score if config.vector_store.enable_reranker else 0.0
        if config.vector_store.enable_reranker:
            min_rerank_score = min(min_rerank_score, 0.12)
        filtered = []
        for doc, score in results:
            if score is None:
                continue
            source_type = doc.metadata.get("source_type", "other")
            if source_type == "resume":
                # 简历问题优先召回，不受重排阈值二次压制
                final_threshold = resume_threshold
            else:
                final_threshold = max(default_threshold, min_rerank_score)
            if score >= final_threshold:
                filtered.append((doc, score))
        return filtered

    @staticmethod
    def _apply_source_policy(question: str, results: List[Any], source_policy: str = "default") -> List[Any]:
        """
        根据语义策略约束来源：
        - resume: 只使用简历证据
        - additional: 优先使用补充资料，不强制剔除简历
        - default: 不额外约束
        """
        if source_policy != "resume":
            return results

        resume_only = [
            (doc, score) for doc, score in results
            if doc.metadata.get("source_type") == "resume"
        ]
        return resume_only if resume_only else results

    def _detect_source_policy(self, question: str) -> str:
        """
        识别用户希望基于哪类资料回答：
        - resume: 明确要求基于简历
        - additional: 明确不按简历，或问题是通用岗位/方法论问法
        - default: 混合
        """
        q = question.strip()
        fallback_policy = self._rule_based_source_policy(q)
        if not config.conversation.enable_intent_classifier:
            return fallback_policy

        if config.conversation.intent_classifier_mode != "hybrid":
            return fallback_policy

        hybrid_policy = self._hybrid_source_policy(q)
        return hybrid_policy if hybrid_policy in {"resume", "additional", "default"} else fallback_policy

    @staticmethod
    def _rule_based_source_policy(question: str) -> str:
        q = question.strip()

        negative_resume_patterns = (
            "不需要根据简历",
            "不要根据简历",
            "不按简历",
            "不基于简历",
            "不要基于简历",
            "不看简历",
        )
        if any(p in q for p in negative_resume_patterns):
            return "additional"

        positive_resume_patterns = (
            "根据简历",
            "基于简历",
            "按简历",
            "简历中",
        )
        if any(p in q for p in positive_resume_patterns):
            return "resume"

        pm_general_patterns = (
            "产品经理",
            "角色",
            "职责",
            "能力框架",
            "方法论",
        )
        if any(p in q for p in pm_general_patterns):
            return "additional"

        return "default"

    def _hybrid_source_policy(self, question: str) -> str:
        """
        轻量混合意图识别：
        - 先走LLM分类
        - 若输出不可解析或置信度低，返回rule结果
        """
        fallback = self._rule_based_source_policy(question)
        prompt = (
            "你是一个意图分类器。请将用户问题分类为以下三类之一：\n"
            "resume: 明确要求基于简历事实回答\n"
            "additional: 明确要求不基于简历，或在问通用岗位职责/方法论\n"
            "default: 其他情况\n"
            "只输出JSON，格式为 {\"policy\":\"resume|additional|default\",\"confidence\":0-1}。\n"
            f"用户问题：{question}"
        )
        try:
            raw = self.llm.invoke(prompt)
            text = raw if isinstance(raw, str) else str(raw)
            match = re.search(r'"policy"\s*:\s*"(?P<policy>resume|additional|default)".*?"confidence"\s*:\s*(?P<conf>0(?:\.\d+)?|1(?:\.0+)?)', text, re.S)
            if not match:
                return fallback
            policy = match.group("policy")
            confidence = float(match.group("conf"))
            return policy if confidence >= 0.65 else fallback
        except Exception:
            return fallback

    @staticmethod
    def _build_effective_question(question: str, session_context: Optional[List[Dict[str, str]]]) -> str:
        """将多轮会话压缩为检索友好的增强问题"""
        if not config.conversation.enable_memory or not session_context:
            return question

        window = max(1, config.conversation.memory_window)
        recent_turns = session_context[-window:]
        condensed = []
        for turn in recent_turns:
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "").strip()
            if not q or not a:
                continue
            condensed.append(f"Q:{q}\nA:{a[:180]}")

        if not condensed:
            return question

        history_block = "\n".join(condensed)
        return f"请结合以下历史对话语义理解当前问题。\n{history_block}\n当前问题：{question}"

    @staticmethod
    def _rerank_results(question: str, results: List[Any], source_policy: str = "default") -> List[Any]:
        """
        轻量重排：结合原始相似分、词汇重合度、来源偏好分
        说明：不引入额外依赖，优先提升可用性与时延稳定性
        """
        if not results:
            return []

        q_terms = {t for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", question.lower()) if len(t) >= 2}
        ranked = []
        for doc, base_score in results:
            text = doc.page_content.lower()
            d_terms = {t for t in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text) if len(t) >= 2}
            overlap = 0.0
            if q_terms and d_terms:
                overlap = len(q_terms & d_terms) / len(q_terms)

            source_type = doc.metadata.get("source_type", "other")
            source_bonus = 0.0
            if source_policy == "resume" and source_type == "resume":
                source_bonus = 0.08
            elif source_policy == "additional" and source_type == "additional":
                source_bonus = 0.08

            safe_base = base_score if base_score is not None else 0.0
            fused_score = 0.75 * safe_base + 0.25 * overlap + source_bonus
            ranked.append((doc, min(1.0, fused_score)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        top_n = max(1, config.vector_store.rerank_top_n)
        return ranked[:top_n]
