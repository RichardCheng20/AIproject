"""
配置管理模块
"""
import yaml
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ModelConfig(BaseModel):
    """
    模型相关配置

    Attributes:
        provider (str): 模型服务提供方，支持 "ollama" 或 "dashscope"
        llm (str): 大语言模型名称，默认为 "qwen2.5:3b"
        embedding (str): 嵌入模型名称，默认为 "nomic-embed-text"
        temperature (float): 模型温度参数，控制输出随机性，默认为 0.3
        max_tokens (int): 最大生成token数，默认为 1024
    """
    provider: str = "ollama"
    llm: str = "qwen2.5:3b"
    embedding: str = "nomic-embed-text"
    temperature: float = 0.3
    max_tokens: int = 1024


class OllamaConfig(BaseModel):
    """
    Ollama服务相关配置

    Attributes:
        base_url (str): Ollama服务基础URL，默认为 "http://localhost:11434"
        timeout (int): 请求超时时间（秒），默认为 120
    """
    base_url: str = "http://localhost:11434"
    timeout: int = 120


class DashScopeConfig(BaseModel):
    """
    阿里百炼(DashScope)服务相关配置

    Attributes:
        api_key (str): DashScope API Key
        base_url (str): DashScope服务地址
    """
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class VectorStoreConfig(BaseModel):
    """
    向量存储相关配置

    Attributes:
        collection_name (str): 集合名称，默认为 "interview_knowledge"
        persist_directory (str): 数据持久化目录，默认为 "./chroma_db"
        similarity_metric (str): 相似度计算方式，默认为 "cosine"
        search_k (int): 搜索返回的最相似条目数量，默认为 5
        relevance_threshold (float): 最低相关性阈值，低于该值则判定为无有效依据
    """
    collection_name: str = "interview_knowledge"
    persist_directory: str = "./chroma_db"
    similarity_metric: str = "cosine"
    search_k: int = 5
    relevance_threshold: float = 0.35
    recall_k: int = 30
    enable_reranker: bool = True
    rerank_top_n: int = 6
    min_rerank_score: float = 0.2


class TextProcessingConfig(BaseModel):
    """
    文本处理相关配置

    Attributes:
        chunk_size (int): 文本分块大小，默认为 800
        chunk_overlap (int): 分块重叠长度，默认为 150
        separators (list): 分割符列表，默认包括换行、句号等标点符号
    """
    chunk_size: int = 800
    chunk_overlap: int = 150
    separators: list = ["\n\n", "\n", "。", "；", "，", " ", ""]


class ServerConfig(BaseModel):
    """
    服务器运行相关配置

    Attributes:
        host (str): 服务器监听地址，默认为 "0.0.0.0"
        port (int): 服务器端口，默认为 8000
        reload (bool): 是否启用热重载模式，默认为 False
        workers (int): 工作进程数，默认为 4
    """
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 4


class KnowledgeBaseConfig(BaseModel):
    """
    知识库相关配置

    Attributes:
        data_dir (str): 知识库数据目录，默认为 "./data"
        main_file (str): 主要知识文件名，默认为 "interview_knowledge.txt"
        resume_file (Optional[str]): 候选人简历文件名（可选），优先用于检索
        supported_extensions (list): 支持的文件扩展名列表
    """
    data_dir: str = "./data"
    main_file: str = "interview_knowledge.txt"
    resume_file: Optional[str] = None
    supported_extensions: list = Field(default_factory=lambda: [".txt", ".md", ".pdf", ".docx"])


class LoggingConfig(BaseModel):
    """
    日志记录相关配置

    Attributes:
        level (str): 日志级别，默认为 "INFO"
        format (str): 日志格式字符串，默认包含时间、模块名、级别和消息内容
        file (str): 日志文件路径，默认为 "interview_assistant.log"
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "interview_assistant.log"


class AppSystemConfig(BaseModel):
    """
    系统信息配置

    Attributes:
        name (str): 系统名称
        version (str): 系统版本
    """
    name: str = "个人面试助手"
    version: str = "1.0.0"


class ConversationConfig(BaseModel):
    """
    会话与意图识别配置

    Attributes:
        enable_memory (bool): 是否启用多轮会话记忆
        memory_window (int): 保留最近N轮问答
        enable_intent_classifier (bool): 是否启用意图分类增强
        intent_classifier_mode (str): rule 或 hybrid
    """
    enable_memory: bool = True
    memory_window: int = 8
    enable_intent_classifier: bool = True
    intent_classifier_mode: str = "hybrid"


class SystemConfig(BaseModel):
    """
    整体系统配置容器类，整合各个子配置项

    Attributes:
        model (ModelConfig): 模型配置实例
        ollama (OllamaConfig): Ollama服务配置实例
        dashscope (DashScopeConfig): DashScope服务配置实例
        vector_store (VectorStoreConfig): 向量存储配置实例
        text_processing (TextProcessingConfig): 文本处理配置实例
        server (ServerConfig): 服务器配置实例
        knowledge_base (KnowledgeBaseConfig): 知识库配置实例
        logging (LoggingConfig): 日志配置实例
        system (AppSystemConfig): 系统信息配置实例
    """
    system: AppSystemConfig = Field(default_factory=AppSystemConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    dashscope: DashScopeConfig = Field(default_factory=DashScopeConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)

    def get(self, key: str, default=None):
        """
        提供类似字典式的键值访问功能

        Args:
            key (str): 使用点号分隔的属性路径，例如 "model.llm"
            default: 当找不到对应属性时返回的默认值

        Returns:
            返回指定路径下的属性值或默认值
        """
        keys = key.split('.')
        current = self
        try:
            for k in keys:
                if hasattr(current, k):
                    current = getattr(current, k)
                else:
                    return default
            return current
        except:
            return default

class ConfigManager:
    """
    配置管理器，负责加载、保存和提供全局配置访问接口
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器

        Args:
            config_path (str): 配置文件路径，默认为 "config.yaml"
        """
        self.config_path = config_path
        self.config = self.load_config()
        self._normalize_config_file_if_needed()

    def load_config(self) -> SystemConfig:
        """
        加载并合并配置文件与默认配置

        Returns:
            SystemConfig: 包含所有配置信息的系统配置对象
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                # 兼容异常情况下的重复拼接配置，只保留首段有效配置
                if "\n# config.yaml" in raw_text:
                    raw_text = raw_text.split("\n# config.yaml", 1)[0]
                first_system = raw_text.find("system:\n")
                if first_system != -1:
                    second_system = raw_text.find("\nsystem:\n", first_system + 1)
                    if second_system != -1:
                        raw_text = raw_text[:second_system]
                config_data = yaml.safe_load(raw_text)
        else:
            config_data = {}

        # 合并默认配置和文件配置
        system_config = SystemConfig()

        # 递归更新配置
        self._update_config(system_config, config_data)

        # 环境变量覆盖（优先级高于配置文件）
        if os.getenv("MODEL_PROVIDER"):
            system_config.model.provider = os.getenv("MODEL_PROVIDER", "ollama")
        if os.getenv("MODEL_LLM"):
            system_config.model.llm = os.getenv("MODEL_LLM", system_config.model.llm)
        if os.getenv("MODEL_EMBEDDING"):
            system_config.model.embedding = os.getenv("MODEL_EMBEDDING", system_config.model.embedding)
        if os.getenv("APP_NAME"):
            system_config.system.name = os.getenv("APP_NAME", system_config.system.name)
        if os.getenv("RESUME_FILE"):
            system_config.knowledge_base.resume_file = os.getenv("RESUME_FILE", system_config.knowledge_base.resume_file)
        if os.getenv("DASHSCOPE_API_KEY"):
            system_config.dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if os.getenv("DASHSCOPE_BASE_URL"):
            system_config.dashscope.base_url = os.getenv("DASHSCOPE_BASE_URL", system_config.dashscope.base_url)
        if os.getenv("ENABLE_RERANKER"):
            system_config.vector_store.enable_reranker = os.getenv("ENABLE_RERANKER", "true").lower() == "true"
        if os.getenv("RERANK_TOP_N"):
            system_config.vector_store.rerank_top_n = int(os.getenv("RERANK_TOP_N", system_config.vector_store.rerank_top_n))
        if os.getenv("ENABLE_MEMORY"):
            system_config.conversation.enable_memory = os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        if os.getenv("MEMORY_WINDOW"):
            system_config.conversation.memory_window = int(os.getenv("MEMORY_WINDOW", system_config.conversation.memory_window))
        if os.getenv("INTENT_CLASSIFIER_MODE"):
            system_config.conversation.intent_classifier_mode = os.getenv("INTENT_CLASSIFIER_MODE", system_config.conversation.intent_classifier_mode)

        return system_config

    def _update_config(self, config_obj, config_data: Dict[str, Any]):
        """
        递归地将配置数据更新到配置对象中

        Args:
            config_obj: 要被更新的配置对象
            config_data (Dict[str, Any]): 来自配置文件的数据字典
        """
        for key, value in config_data.items():
            if hasattr(config_obj, key):
                attr = getattr(config_obj, key)
                if isinstance(attr, BaseModel):
                    self._update_config(attr, value)
                else:
                    setattr(config_obj, key, value)

    def get_config(self) -> SystemConfig:
        """
        获取当前系统的完整配置对象

        Returns:
            SystemConfig: 当前系统的所有配置信息
        """
        return self.config

    def save_config(self):
        """
        将当前内存中的配置保存回配置文件
        """
        config_dict = self.config.model_dump()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)

    def _normalize_config_file_if_needed(self):
        """
        检测并净化重复拼接的配置文件。
        当检测到多段system配置或历史拼接标记时，回写为单份标准配置。
        """
        if not os.path.exists(self.config_path):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            needs_normalize = (
                "# config.yaml" in raw_text
                or raw_text.count("\nsystem:") > 1
            )
            if not needs_normalize:
                return

            config_dict = self.config.model_dump()
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2, sort_keys=False)

        except Exception:
            # 净化失败不阻塞主流程，保持加载可用
            pass


# 全局配置实例
config_manager = ConfigManager()
config = config_manager.get_config()

if __name__ == '__main__':
    print(config.get("system.name", "个人面试助手"))
