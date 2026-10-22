#!/bin/sh
# =============================================================================
# 知识库应用容器启动脚本
#
# 通过环境变量 SERVICE 决定启动哪个服务：
#   SERVICE=import → 导入服务（端口 8000，含 /import.html 页面）
#   SERVICE=query  → 查询服务（端口 8084，含 /chat.html 页面与 SSE 流式接口）
#
# 模型说明（CPU 服务器部署方案）：
#   本镜像不加载任何本地模型权重。BGE-M3 与 BGE-Reranker 均通过
#   .env 中的 BGE_M3_USE_LOCAL=False / BGE_RERANKER_USE_LOCAL=False
#   调用远端 GPU 服务器上已部署的 HTTP 服务，因此本容器无需 GPU、无需挂载模型目录。
#   LLM 同理：LLM_USE_LOCAL=False 走云端 DashScope（或自行部署的 vLLM）。
#
# 用法：
#   docker run -e SERVICE=query -p 8084:8084 knowledge-base:app
# =============================================================================
set -e

# 当前工作目录（镜像内为 /app）
APP_DIR="${APP_DIR:-/app}"
cd "$APP_DIR"

# 校验必填的 SERVICE 变量，缺失时直接退出并给出可用取值，避免容器空跑
if [ -z "$SERVICE" ]; then
    echo "[entrypoint] 错误：未设置 SERVICE 环境变量，可选值为 import / query" >&2
    exit 1
fi

# ==================== .env 就位自检 ====================
# 项目所有配置（Milvus/MongoDB/MinIO/LLM/模型开关等）都来自 .env：
#   - app/utils/path_util.py 靠 .env 定位 PROJECT_ROOT
#   - app/conf/*.py 靠 load_dotenv() 读取配置
# 缺失时不会立刻报错，而是以「配置全为 None」的方式在后续连接数据库时才崩，
# 排查困难。这里在启动前主动检查，把问题暴露在第一步。
ENV_FILE="${APP_DIR:-/app}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "[entrypoint] 已检测到 .env 配置文件：$ENV_FILE"
elif [ -n "$MILVUS_URL" ] || [ -n "$MONGO_URL" ] || [ -n "$OPENAI_API_KEY" ]; then
    # 没有 .env 文件，但关键配置已通过环境变量注入（如 compose 的 environment/env_file）
    echo "[entrypoint] 未发现 .env 文件，但检测到环境变量注入的配置，继续启动"
else
    echo "[entrypoint] 错误：未找到 .env 文件，且未检测到关键配置的环境变量注入" >&2
    echo "[entrypoint]   预期位置：$ENV_FILE" >&2
    echo "[entrypoint]   解决方式（二选一）：" >&2
    echo "[entrypoint]     1) 挂载：docker run -v /host/path/.env:/app/.env ..." >&2
    echo "[entrypoint]     2) compose：env_file: ../.env 或 environment 中逐项配置" >&2
    exit 1
fi

# 打印启动信息，便于 docker logs 排查
echo "[entrypoint] 启动服务：SERVICE=$SERVICE"
echo "[entrypoint] 工作目录：$(pwd)"
echo "[entrypoint] Python：$(python --version 2>&1)"

case "$SERVICE" in
    import)
        echo "[entrypoint] 导入服务监听 0.0.0.0:8000"
        # --host 0.0.0.0 必须显式指定，否则容器内无法被外部访问
        exec python -m uvicorn \
            app.import_process.api.file_import_service:app \
            --host 0.0.0.0 \
            --port 8000
        ;;
    query)
        echo "[entrypoint] 查询服务监听 0.0.0.0:8084"
        exec python -m uvicorn \
            app.query_process.api.query_service:app \
            --host 0.0.0.0 \
            --port 8084
        ;;
    *)
        echo "[entrypoint] 错误：未知的 SERVICE 值「$SERVICE」，可选值为 import / query" >&2
        exit 1
        ;;
esac
