"""
主程序入口
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config, config_manager
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.rag_pipeline import RAGPipeline
from src.chat_interface import create_interface
from src.web_api import run_server


def setup_logging():
    """
    配置日志系统

    根据配置文件中的日志设置，初始化日志等级、格式以及输出目标（控制台和文件）。
    """
    log_config = config.logging

    logging.basicConfig(
        level=getattr(logging, log_config.level),
        format=log_config.format,
        handlers=[
            logging.FileHandler(log_config.file),
            logging.StreamHandler()
        ]
    )


def initialize_knowledge_base():
    """
    初始化知识库

    包括以下步骤：
    1. 使用DocumentProcessor处理知识库文档
    2. 若存在文档，则使用VectorStoreManager创建或加载向量存储
    3. 返回初始化后的VectorStoreManager实例

    Returns:
        VectorStoreManager or None: 成功时返回向量存储管理器实例，失败则返回None
    """
    logger = logging.getLogger(__name__)

    try:
        # 初始化文档处理器
        processor = DocumentProcessor()

        # 处理知识库文档
        documents = processor.process_knowledge_base()

        if not documents:
            logger.warning("未找到任何文档，知识库为空")
            return None

        # 初始化向量存储管理器
        vector_manager = VectorStoreManager()

        # 创建或加载向量存储
        vectorstore = vector_manager.get_or_create_vectorstore(documents)

        logger.info(f"知识库初始化完成，共 {len(documents)} 个文档块")

        return vector_manager

    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")
        return None


def main():
    """
    主函数 - 系统启动入口

    支持三种运行模式：
    - api：启动RESTful API服务
    - web：启动自定义Web界面
    - gradio：启动Gradio交互界面

    同时支持命令行参数来指定运行模式、是否初始化知识库等选项。
    """
    parser = argparse.ArgumentParser(description="个人面试助手（RAG问答系统）")
    parser.add_argument("--mode", choices=["api", "web", "gradio"], default="api",
                        help="运行模式: api(API服务), web(Web界面), gradio(Gradio界面)")
    parser.add_argument("--init-kb", action="store_true", help="初始化知识库")
    parser.add_argument("--host", help="服务器主机")
    parser.add_argument("--port", type=int, help="服务器端口")

    args = parser.parse_args()

    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)

    # 覆盖配置
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port

    logger.info(f"启动个人面试助手 v{config.get('system.version', '1.0.0')}")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"模型提供方: {config.model.provider}")
    logger.info(f"LLM模型: {config.model.llm}")
    logger.info(f"嵌入模型: {config.model.embedding}")

    # 初始化知识库
    if args.init_kb:
        logger.info("开始初始化知识库...")
        vector_manager = initialize_knowledge_base()
        if vector_manager:
            logger.info("知识库初始化成功")
        else:
            logger.warning("知识库初始化失败")

    # 根据模式运行
    if args.mode == "api":
        # 运行API服务
        logger.info(f"启动API服务: http://{config.server.host}:{config.server.port}")
        run_server()

    elif args.mode == "gradio":
        # 运行Gradio界面
        try:
            # 初始化系统
            vector_manager = VectorStoreManager()
            vectorstore = vector_manager.load_vectorstore()

            if vectorstore is None:
                logger.warning("向量存储未找到，尝试初始化知识库...")
                vector_manager = initialize_knowledge_base()
                if vector_manager is None:
                    logger.error("无法初始化知识库，退出")
                    return

            # 初始化RAG管道
            rag_pipeline = RAGPipeline(vector_manager)

            # 创建Gradio界面
            demo = create_interface(rag_pipeline, use_gradio=True)

            # 运行Gradio
            demo.launch(
                server_name=config.server.host,
                server_port=config.server.port,
                share=False
            )

        except Exception as e:
            logger.error(f"启动Gradio界面失败: {e}")

    elif args.mode == "web":
        # 运行Web界面（自定义）
        logger.info(f"启动Web界面: http://{config.server.host}:{config.server.port}")
        run_server()


if __name__ == "__main__":
    main()
    # rm -rf chroma_db
    # python main.py --mode web --init-kb  
    # https://chat.deepseek.com/share/t9cp2ymzdhze77duv4
