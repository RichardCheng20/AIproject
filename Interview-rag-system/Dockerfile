# 基础镜像设置
# 使用Python 3.10的轻量级镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
# 将容器内的/app目录设置为工作目录
WORKDIR /app

# 安装系统依赖
# 更新apt包管理器并安装必要的系统工具和库
# 包括curl（网络工具）、build-essential（编译工具链）
# 清理apt缓存以减小镜像体积
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
# 将项目所需的配置文件、数据文件、源代码等复制到容器中
COPY requirements.txt .
COPY config.yaml .
COPY data/ ./data/
COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY main.py .

# 安装Python依赖
# 根据requirements.txt文件安装项目所需的Python包
# 使用--no-cache-dir选项避免缓存，减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 创建日志目录
# 创建用于存储应用日志的目录
RUN mkdir -p /app/logs

# 暴露端口
# 声明容器将监听8000端口
EXPOSE 8000

# 启动命令
# 容器启动时执行的默认命令，启动Python应用
# 参数说明：
# --mode api: 以API模式运行应用
# --host 0.0.0.0: 监听所有网络接口
# --port 8000: 监听8000端口
CMD ["python", "main.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
