"""
网页聊天界面
提供Gradio或自定义网页界面
"""
import gradio as gr
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStoreManager
from src.config import config

logger = logging.getLogger(__name__)


class ChatInterface:
    """聊天界面"""

    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.chat_history = []

    def create_gradio_interface(self):
        """创建Gradio界面"""
        llm_display_name = getattr(self.rag_pipeline.llm, "model", None) or config.model.llm

        def respond(message, history):
            """处理用户消息"""
            if history is None:
                history = []
            try:
                # 执行查询
                result = self.rag_pipeline.query(message)

                # 构建响应
                response = result["answer"]

                # 添加来源信息
                if result["relevant_sources"]:
                    response += "\n\n**参考来源：**\n"
                    for i, source in enumerate(result["relevant_sources"][:3], 1):
                        source_text = source["content"]
                        source_name = source.get("source", "未知")
                        response += f"{i}. {source_name}: {source_text[:100]}...\n"

                # Gradio 4.36 的 Chatbot 默认需要 list[tuple[user, assistant]]
                history.append((message, response))
                return history
            except Exception as e:
                logger.error(f"处理消息失败: {e}")
                error_response = f"抱歉，处理您的消息时出现错误：{str(e)}"
                history.append((message, error_response))
                return history

        # 创建聊天界面
        chatbot = gr.Chatbot(
            label="个人面试助手",
            height=600,
            avatar_images=(None, "🤖"),
            show_copy_button=True
        )

        # 创建Gradio界面
        with gr.Blocks(title="个人面试助手", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 💼 个人面试助手")
            gr.Markdown("基于RAG技术的面试问答助手，可根据你的个人知识文档模拟面试提问与回答。")

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot.render()

                    with gr.Row():
                        msg = gr.Textbox(
                            label="请输入您的问题",
                            placeholder="例如：请做一个自我介绍；你负责过最有挑战的项目是什么？",
                            scale=4
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("清空对话")
                        refresh_btn = gr.Button("刷新知识库")

                with gr.Column(scale=1):
                    gr.Markdown("### 示例问题")
                    examples = gr.Examples(
                        examples=[
                            "请按照简历内容介绍你的教育背景（学校、专业、时间、GPA）。",
                            "请根据简历说明你在华为云NA卓越运营岗位的核心职责与成果。",
                            "请根据简历介绍你在华为云存储产品经理阶段的代表项目与量化结果。",
                            "请基于简历讲述你在EVS快照商用项目中主导的关键工作。",
                            "请严格依据简历总结你的技术能力与综合能力，不要补充简历外信息。"
                        ],
                        inputs=msg,
                        label="点击试试"
                    )

                    gr.Markdown("### 系统信息")
                    info_text = gr.Markdown(f"""
                    - **模型**: {llm_display_name}
                    - **知识库**: 个人面试资料
                    - **状态**: 就绪
                    """)

            # 绑定事件
            msg.submit(respond, [msg, chatbot], [chatbot])
            submit_btn.click(respond, [msg, chatbot], [chatbot])

            def clear_chat():
                self.chat_history = []
                return []

            clear_btn.click(clear_chat, None, [chatbot])

            def refresh_kb():
                # 这里可以添加刷新知识库的逻辑
                return "知识库刷新功能需通过API调用"

            refresh_btn.click(refresh_kb, None, [msg])

        return demo

    def create_web_interface(self):
        """创建自定义Web界面"""
        # 这里返回HTML模板路径和静态文件配置
        return {
            "template": "templates/index.html",
            "static_dir": "static",
            "api_endpoint": "/api/query"
        }


def create_interface(rag_pipeline: RAGPipeline, use_gradio: bool = True):
    """
    创建界面

    Args:
        rag_pipeline: RAG管道实例
        use_gradio: 是否使用Gradio

    Returns:
        界面实例
    """
    interface = ChatInterface(rag_pipeline)

    if use_gradio:
        return interface.create_gradio_interface()
    else:
        return interface.create_web_interface()