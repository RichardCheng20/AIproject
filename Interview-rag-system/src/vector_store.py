"""
向量存储模块
负责向量数据库的创建、更新和检索
"""
import os
import logging
import shutil
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from chromadb.api.client import SharedSystemClient

from src.config import config

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """初始化嵌入模型"""
        try:
            provider = config.model.provider.lower()
            if provider == "dashscope":
                if not config.dashscope.api_key:
                    raise ValueError("未配置DASHSCOPE_API_KEY，无法使用百炼向量模型")
                self.embeddings = DashScopeEmbeddings(
                    model=config.model.embedding,
                    dashscope_api_key=config.dashscope.api_key
                )
            else:
                self.embeddings = OllamaEmbeddings(
                    model=config.model.embedding,
                    base_url=config.ollama.base_url,
                    # timeout=config.ollama.timeout
                )
            logger.info(f"嵌入模型初始化成功: {config.model.embedding}")
        except Exception as e:
            logger.error(f"嵌入模型初始化失败: {e}")
            raise

    @staticmethod
    def _is_schema_incompatible_error(error: Exception) -> bool:
        """判断是否为Chroma持久化Schema不兼容错误"""
        message = str(error).lower()
        return (
            "no such column: collections.topic" in message
            or "sqlite3.operationalerror" in message and "collections.topic" in message
        )

    @staticmethod
    def _reset_persist_directory(persist_dir: Path):
        """删除并重建向量库目录，用于恢复不兼容Schema"""
        # 清空Chroma共享客户端缓存，避免仍持有旧SQLite连接
        SharedSystemClient.clear_system_cache()
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)
        persist_dir.mkdir(parents=True, exist_ok=True)

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        创建向量存储

        Args:
            documents: 文档列表

        Returns:
            向量存储实例
        """
        try:
            # 确保存储目录存在
            persist_dir = Path(config.vector_store.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)

            # 创建向量存储
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(persist_dir),
                collection_name=config.vector_store.collection_name,
                collection_metadata={"hnsw:space": config.vector_store.similarity_metric}
            )

            logger.info(f"向量存储创建成功: {len(documents)} 个文档块")
            return self.vectorstore

        except Exception as e:
            logger.error(f"向量存储创建失败: {e}")
            raise

    def load_vectorstore(self) -> Optional[Chroma]:
        """
        加载现有向量存储

        Returns:
            向量存储实例或None
        """
        persist_dir = Path(config.vector_store.persist_directory)

        if not persist_dir.exists():
            logger.warning(f"向量存储目录不存在: {persist_dir}")
            return None

        try:
            self.vectorstore = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings,
                collection_name=config.vector_store.collection_name
            )

            # 检查集合是否存在
            collection_count = self.vectorstore._collection.count()
            logger.info(f"向量存储加载成功: {collection_count} 个文档块")

            return self.vectorstore

        except Exception as e:
            logger.error(f"向量存储加载失败: {e}")
            if self._is_schema_incompatible_error(e):
                logger.warning(
                    "检测到向量库Schema不兼容，已清理旧索引目录并准备重建"
                )
                self._reset_persist_directory(persist_dir)
            return None

    def get_or_create_vectorstore(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        获取或创建向量存储

        Args:
            documents: 可选的文档列表，用于创建新存储

        Returns:
            向量存储实例
        """
        # 尝试加载现有存储
        vectorstore = self.load_vectorstore()

        if vectorstore is not None and documents is not None:
            existing_count = vectorstore._collection.count()
            expected_count = len(documents)
            if existing_count != expected_count:
                logger.warning(
                    "检测到向量库文档数不一致(existing=%s, expected=%s)，执行重建",
                    existing_count,
                    expected_count,
                )
                persist_dir = Path(config.vector_store.persist_directory)
                self._reset_persist_directory(persist_dir)
                vectorstore = None

        if vectorstore is None and documents is not None:
            # 创建新存储
            try:
                vectorstore = self.create_vectorstore(documents)
            except Exception as e:
                persist_dir = Path(config.vector_store.persist_directory)
                if self._is_schema_incompatible_error(e):
                    logger.warning("创建向量库时检测到Schema冲突，执行强制重建")
                    self._reset_persist_directory(persist_dir)
                    vectorstore = self.create_vectorstore(documents)
                else:
                    raise

        self.vectorstore = vectorstore
        return vectorstore

    def create_retriever(self, search_k: Optional[int] = None) -> VectorStoreRetriever:
        """
        创建检索器

        Args:
            search_k: 检索结果数量

        Returns:
            检索器实例
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        if search_k is None:
            search_k = config.vector_store.search_k

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": search_k}
        )

        return self.retriever

    def add_documents(self, documents: List[Document]):
        """
        向向量存储添加文档

        Args:
            documents: 文档列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        try:
            self.vectorstore.add_documents(documents)
            logger.info(f"成功添加 {len(documents)} 个文档块到向量存储")
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def similarity_search(
            self,
            query: str,
            k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        相似性搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            文档和相似度分数列表
        """
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")

        if k is None:
            k = config.vector_store.search_k

        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
            return results
        except Exception as e:
            logger.error(f"相似性搜索失败: {e}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息

        Returns:
            集合信息字典
        """
        if self.vectorstore is None:
            return {"error": "向量存储未初始化"}

        try:
            collection = self.vectorstore._collection
            count = collection.count()

            return {
                "collection_name": config.vector_store.collection_name,
                "document_count": count,
                "persist_directory": config.vector_store.persist_directory,
                "similarity_metric": config.vector_store.similarity_metric
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}