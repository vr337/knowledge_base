# =============================================================================
# 知识库应用镜像（CPU 服务器专用）
#
# 包含：导入服务(8000) / 查询服务(8084)
# 启动：docker run -e SERVICE=query -p 8084:8084 knowledge-base:app
#
# 构建：
#   docker build -t knowledge-base:app -f docker/Dockerfile .
#
# 设计要点：
#   1. 本镜像面向「无 GPU 的 CPU 服务器」，torch 三件套从 PyTorch 官方 CPU 源解析，
#      不引入 CUDA 依赖（见 pyproject.toml 的 [tool.uv.sources]）。
#   2. 镜像内不打包任何模型权重。BGE-M3 / BGE-Reranker / LLM 均通过 .env 开关
#      走远端 HTTP 服务或云端 API，因此无需 GPU、无需挂载 models 目录。
#   3. 使用 uv + uv.lock 做可复现安装（pyproject.toml 已配清华镜像源）。
#   4. 根项目被标记为非包（package=false），故仅装依赖、不构建根项目。
#   5. docker/.dockerignore 已排除 .venv / .git / logs / output / .env 等，
#      保证 COPY . . 不会把本地环境和密钥带进镜像。
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    VENV_PATH=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 系统依赖：
#   build-essential : 部分 Python 包（如 flagembedding 依赖链）需要编译
#   libgl1          : opencv-python 运行时必需（torchvision / magic_pdf 传递依赖）
#   libglib2.0-0    : opencv 的 glib 依赖
#   poppler-utils   : pdf 处理工具（magic_pdf / pymupdf 生态常用）
#   curl            : 容器健康检查用
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv（项目用 uv 管理依赖）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先拷贝依赖清单，利用层缓存（源码变动不会触发重装依赖）
# 注意：.dockerignore 已排除 uv.lock 与 pyproject.toml，这里单独 COPY 后
#       COPY . . 不会覆盖它们，保证 lock 版本稳定。
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 拷贝源码（app/ prompts/ 等，.venv/.git/logs/.env 已被 .dockerignore 排除）
COPY . .

# 启动脚本
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ==================== 环境变量文件（.env）说明 ====================
# .env 不通过 COPY 进镜像（.dockerignore 已排除），原因有两个：
#   1. 避免密钥（OPENAI_API_KEY / MINERU_API_TOKEN / 各中间件密码）固化到镜像层；
#   2. 开发机与服务器的中间件地址不同，打进镜像会导致地址写死。
#
# 部署时由以下两种方式之一提供，二者都会被 load_dotenv() 读取：
#   A. docker compose 的 env_file: ../.env   （当前 docker-compose.yml 采用的方式，
#      以环境变量形式注入，os.getenv 直接可读）
#   B. 挂载到容器内 /app/.env                （docker run -v /host/.env:/app/.env，
#      load_dotenv() 默认查找当前工作目录下的 .env，容器 WORKDIR 即 /app）
#
# 关键：app/utils/path_util.py 通过查找含 .env 的目录来定位 PROJECT_ROOT，
# 因此方式 B 下 .env 必须位于 /app/.env；方式 A 下 PROJECT_ROOT 靠 __file__ 推导，
# 同样指向 /app，两种方式均可正确定位。
ENV PROJECT_ROOT=/app

# 导入服务 8000 / 查询服务 8084
EXPOSE 8000 8084

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
