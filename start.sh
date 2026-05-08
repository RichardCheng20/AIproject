#!/bin/bash

# 个人面试助手启动脚本

set -e

echo "========================================"
echo "  个人面试助手启动脚本"
echo "========================================"

# 检查Ollama服务
echo "检查Ollama服务..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama服务未启动，尝试启动..."

    # 检查是否已安装Ollama
    if ! command -v ollama &> /dev/null; then
        echo "Ollama未安装，开始安装..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi

    # 启动Ollama服务
    ollama serve &
    OLLAMA_PID=$!

    # 等待Ollama启动
    echo "等待Ollama服务启动..."
    sleep 10

    # 下载模型
    echo "下载模型..."
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text
fi

# 检查Python依赖
echo "检查Python依赖..."
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

echo "激活虚拟环境并安装依赖..."
source venv/bin/activate
pip install -r requirements.txt

# 检查知识库文件
echo "检查知识库文件..."
if [ ! -f "data/interview_knowledge.txt" ]; then
    echo "警告：未找到知识库文件，系统将以空知识库运行"
    mkdir -p data
    touch data/interview_knowledge.txt
fi

# 创建必要目录
echo "创建必要目录..."
mkdir -p chroma_db logs

# 选择运行模式
echo "请选择运行模式："
echo "1) API服务模式"
echo "2) Gradio网页界面"
echo "3) 自定义Web界面"
read -p "请输入选择 (1-3): " mode_choice

case $mode_choice in
    1)
        echo "启动API服务模式..."
        python main.py --mode api --init-kb
        ;;
    2)
        echo "启动Gradio界面..."
        python main.py --mode gradio
        ;;
    3)
        echo "启动自定义Web界面..."
        python main.py --mode web
        ;;
    *)
        echo "无效选择，使用默认API服务模式"
        python main.py --mode api --init-kb
        ;;
esac

# 清理
if [ ! -z "$OLLAMA_PID" ]; then
    wait $OLLAMA_PID
fi

echo "系统已停止"