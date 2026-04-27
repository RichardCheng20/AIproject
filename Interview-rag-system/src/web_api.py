"""
Web API模块
提供RESTful API接口
"""
import logging
from typing import List, Optional
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# 定义数据模型
class QueryRequest(BaseModel):
    """
    查询请求模型

    Attributes:
        question (str): 用户提出的问题，长度在1到1000字符之间
        session_id (Optional[str]): 会话ID，用于标识用户会话
        stream (bool): 是否启用流式响应，默认为False
    """
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    """
    查询响应模型

    Attributes:
        question (str): 用户提出的问题
        answer (str): 系统生成的回答
        relevant_sources (List[dict]): 相关来源信息列表
        source_count (int): 来源数量统计
        status (str): 响应状态描述
        timestamp (str): 时间戳字符串
        session_id (Optional[str]): 对应回话ID
    """
    question: str
    answer: str
    relevant_sources: List[dict]
    source_count: int
    status: str
    timestamp: str
    session_id: Optional[str] = None


class KnowledgeBaseInfo(BaseModel):
    """
    知识库信息模型

    Attributes:
        document_count (int): 文档总数
        last_updated (Optional[str]): 最后更新时间（可选）
        vector_store_info (dict): 向量存储相关信息
    """
    document_count: int
    last_updated: Optional[str]
    vector_store_info: dict


class SystemStatus(BaseModel):
    """
    系统状态模型

    Attributes:
        status (str): 当前运行状态
        version (str): 版本号
        model (str): 使用的大语言模型名称
        embedding_model (str): 使用的嵌入模型名称
        uptime (str): 系统运行时长
    """
    status: str
    version: str
    model: str
    embedding_model: str
    uptime: str


# 初始化应用
app = FastAPI(
    title=config.get("system.name", "个人面试助手"),
    description="基于LangChain的个人面试问答系统，结合RAG对个人知识文档进行检索增强回答",
    version=config.get("system.version", "1.0.0"),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# 配置CORS跨域资源共享策略
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局组件实例
document_processor = None
vector_store_manager = None
rag_pipeline = None
start_time = datetime.now()
session_memory = defaultdict(list)


def initialize_system():
    """
    初始化系统核心组件：文档处理器、向量存储管理器和RAG管道。

    此函数负责加载或构建知识库，并初始化整个系统的推理流程。
    若过程中出现异常将抛出错误日志并终止程序。
    """
    global document_processor, vector_store_manager, rag_pipeline

    try:
        # 初始化组件
        document_processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager()

        # 处理知识库并创建向量存储
        logger.info("开始初始化知识库...")
        documents = document_processor.process_knowledge_base()

        if documents:
            vector_store_manager.get_or_create_vectorstore(documents)
            logger.info(f"知识库初始化完成，共 {len(documents)} 个文档块")
        else:
            # 尝试加载现有向量存储
            vector_store = vector_store_manager.load_vectorstore()
            if vector_store is None:
                logger.warning("知识库为空，系统将以空知识库运行")

        # 初始化RAG管道
        rag_pipeline = RAGPipeline(vector_store_manager)
        logger.info("系统初始化完成")

    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """
    应用启动事件回调函数，在服务启动时自动调用initialize_system方法来完成系统初始化工作。
    """
    initialize_system()


# API端点定义区域
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    根路径访问入口，返回前端聊天页面HTML内容。

    Returns:
        str: 渲染后的HTML页面内容
    """
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """
    提供一个简单的健康检查接口，确认服务是否正常运行。

    Returns:
        dict: 包含当前服务状态的对象
    """
    return {"status": "healthy"}


@app.get("/api/status")
async def get_status():
    """
    获取系统详细运行状态信息，包括版本号、模型配置及运行时间等关键指标。

    Returns:
        SystemStatus: 系统状态对象
    """
    uptime = str(datetime.now() - start_time)

    return SystemStatus(
        status="running",
        version=config.get("system.version", "1.0.0"),
        model=f"{config.model.provider}:{config.model.llm}",
        embedding_model=config.model.embedding,
        uptime=uptime
    )


@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    接收用户的提问并通过RAG管道进行回答。该接口不支持流式输出，请使用专门的流式接口。

    Args:
        request (QueryRequest): 包含问题文本及其他控制参数的请求体

    Returns:
        QueryResponse: 结构化的查询结果响应对象

    Raises:
        HTTPException: 当系统未初始化或输入无效时抛出相应HTTP异常
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 检查是否请求流式响应
    if request.stream:
        raise HTTPException(status_code=400, detail="请使用 /api/query/stream 端点进行流式查询")

    session_id = request.session_id or "default"
    history = session_memory.get(session_id, [])
    result = rag_pipeline.query(request.question, session_context=history)
    result["session_id"] = request.session_id
    if config.conversation.enable_memory:
        history.append({"question": request.question, "answer": result.get("answer", "")})
        session_memory[session_id] = history[-max(1, config.conversation.memory_window):]

    return QueryResponse(**result)


@app.post("/api/query/stream")
async def stream_query(request: QueryRequest):
    """
    支持SSE(Server-Sent Events)方式实时推送答案片段给客户端。

    Args:
        request (QueryRequest): 包含问题文本及其他控制参数的请求体

    Returns:
        StreamingResponse: 流式响应对象

    Raises:
        HTTPException: 当系统未初始化或输入无效时抛出相应HTTP异常
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    session_id = request.session_id or "default"
    history = session_memory.get(session_id, [])
    answer_chunks = []

    def generate():
        try:
            for chunk in rag_pipeline.stream_query(request.question, session_context=history):
                answer_chunks.append(chunk)
                yield f"data: {chunk}\n\n"
        finally:
            if config.conversation.enable_memory:
                history.append({"question": request.question, "answer": "".join(answer_chunks)})
                session_memory[session_id] = history[-max(1, config.conversation.memory_window):]

    response = StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
    return response


@app.get("/api/knowledge-base/info")
async def get_knowledge_base_info():
    """
    获取当前知识库的基本信息，如文档数量、集合详情等元数据。

    Returns:
        KnowledgeBaseInfo: 知识库信息结构化对象

    Raises:
        HTTPException: 当系统未初始化时抛出相应HTTP异常
    """
    if vector_store_manager is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    info = vector_store_manager.get_collection_info()

    return KnowledgeBaseInfo(
        document_count=info.get("document_count", 0),
        last_updated=None,  # 可以添加最后更新时间
        vector_store_info=info
    )


@app.post("/api/knowledge-base/refresh")
async def refresh_knowledge_base(background_tasks: BackgroundTasks):
    """
    触发后台任务以重新加载和重建知识库索引，不影响主服务响应速度。

    Args:
        background_tasks (BackgroundTasks): FastAPI提供的后台任务调度器

    Returns:
        dict: 表示任务已被接受的状态消息

    Raises:
        HTTPException: 当系统未初始化时抛出相应HTTP异常
    """
    if document_processor is None or vector_store_manager is None:
        raise HTTPException(status_code=503, detail="系统未初始化")

    def refresh_task():
        try:
            # 重新处理知识库
            documents = document_processor.process_knowledge_base()

            if documents:
                # 重新创建向量存储
                vector_store_manager.create_vectorstore(documents)
                logger.info(f"知识库刷新完成，共 {len(documents)} 个文档块")
            else:
                logger.warning("知识库刷新失败，未找到文档")

        except Exception as e:
            logger.error(f"知识库刷新失败: {e}")

    # 在后台执行刷新任务
    background_tasks.add_task(refresh_task)

    return {"status": "accepted", "message": "知识库刷新任务已启动"}


@app.get("/api/models/available")
async def get_available_models():
    """
    返回当前可用的语言模型与嵌入模型配置信息，便于前端展示或选择。

    Returns:
        dict: 包括LLM模型名、Embedding模型名以及Ollama地址的信息字典
    """
    # 这里可以调用Ollama API获取模型列表
    # 简化实现，返回配置的模型
    return {
        "llm_model": config.model.llm,
        "embedding_model": config.model.embedding,
        "ollama_url": config.ollama.base_url,
        "enable_reranker": config.vector_store.enable_reranker,
        "rerank_top_n": config.vector_store.rerank_top_n,
        "enable_memory": config.conversation.enable_memory,
        "memory_window": config.conversation.memory_window,
        "intent_classifier_mode": config.conversation.intent_classifier_mode
    }


# 挂载静态资源目录
app.mount("/static", StaticFiles(directory="static"), name="static")


def run_server():
    """
    启动Uvicorn Web服务器，根据配置决定监听地址、端口及相关性能选项。
    """
    # 启动uvicorn服务器运行Web应用
    # 参数说明:
    #   "src.web_api:app" - 应用程序入口点，指向src/web_api.py文件中的app实例
    #   host - 服务器监听的主机地址，从配置文件中获取
    #   port - 服务器监听的端口号，从配置文件中获取
    #   reload - 是否启用自动重载功能，开发模式下使用，从配置文件中获取
    #   workers - 工作进程数量，当启用重载功能时强制设置为1以避免冲突
    #   log_level - 日志级别设置为info级别
    uvicorn.run(
        "src.web_api:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.reload,
        workers=config.server.workers if not config.server.reload else 1,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()
