# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 补充：本仓库为用户「掌柜智库」知识库项目（RAG 系统），依赖全部走国内镜像源，`.env` 配置项清单见 `.env.example`。

## 项目定位

基于 **LangGraph 双工作流** 的 RAG 知识库系统：一条「导入」链路把 PDF/MD 文档切片、向量化、写入 Milvus；一条「查询」链路做多路召回 + 融合 + 重排 + LLM 生成，通过 SSE 流式吐给前端。

## 常用命令

```bash
# 环境安装（uv 管理，已配清华源，package=false 只装依赖不构建根项目）
uv sync

# 启动导入服务（端口 8000，页面 /import.html）
uv run python -m app.import_process.api.file_import_service

# 启动查询服务（端口 8084，页面 /chat.html，/health 健康检查）
uv run python -m app.query_process.api.query_service

# 跑单个测试（test/ 目录是手写脚本，非标准 pytest 用例，直接按文件运行）
uv run python test/test_node_pdf_to_md.py

# RAG 质量评估（需 Milvus/Mongo/LLM/MCP 全部在线，读 eval/qa.csv 写 eval/qa_result.csv）
uv run python eval/eval.py

# 本地中间件（Milvus/Neo4j/MongoDB/MinIO/Attu）+ 应用容器
docker compose -f docker/docker-compose.yml up -d
```

注意：`test/`、`output/`、`logs/`、`doc/`、`.venv` 均被 `.gitignore` 忽略，`test/` 下的脚本不在版本控制内。

## 架构

### 两条 LangGraph 流水线

| 链路 | 入口 | State | 编译产物 |
|------|------|-------|----------|
| 导入 | [app/import_process/agent/main_graph.py](app/import_process/agent/main_graph.py) | `ImportGraphState` | `kb_import_app` |
| 查询 | [app/query_process/agent/main_graph.py](app/query_process/agent/main_graph.py) | `QueryGraphState` | `kb_query_app` |

**导入链**（按 `node_entry` 判文件后缀路由，只支持 `.md` / `.pdf`）：

```
node_entry → (node_pdf_to_md →) node_md_img → node_document_split
  → node_item_name_recognition → node_bge_embedding → node_import_milvus
```

- `node_pdf_to_md`：调 **MinerU 云 API**（非本地 magic-pdf）上传 PDF → 轮询 → 下载 zip → 解压取 md。
- `node_md_img`：图片转 **MinIO** 公网 URL，用多模态模型（VL_MODEL）总结图片语义，回写 `![总结](url)` 到 md。
- `node_item_name_recognition`：LLM 识别文档「产品主体」（item_name），**同一 item_name 写入两个 Milvus collection**（切片库 + item_name 库）。
- `node_import_milvus`：按 `file_title + item_name` 删旧数据后插入，实现幂等。

**查询链**（`node_item_name_confirm` 之后三路并行，RRF 汇合）：

```
node_item_name_confirm ─有 answer(反问/拒答)─→ node_answer_output
        └─ 无 answer ─→ search_parallel_dispatcher ─┬→ node_search_embedding ┐
                                                    ├→ node_search_embedding_hyde ┼→ node_rrf → node_rerank → node_answer_output
                                                    └→ node_web_search_mcp ─────┘
```

- `node_item_name_confirm` 是**闸门**：LLM 从问题+历史提取 `item_names` → BGE-M3 向量化 → 在 Milvus `kb_item_names` 混合检索 → 按分数阈值（≥0.85 确认 / ≥0.6 候选）对齐。无法确认唯一型号时直接生成反问句塞进 `state["answer"]`，整条检索链被短路。
- 三路召回的查询都带 `item_name in [...]` 标量过滤，所以 item_name 对不上就查不到东西。
- `node_rrf` 只融合 embedding + hyde 两路（`source_weights` 里没放 web_search_docs），但 web 结果由 `node_rerank` 的 `step_1_merge_docs` 统一合并进候选。
- `node_answer_output` 里 `state["answer"]` 已存在（来自反问/拒答）时直接流式吐出，不再调 LLM。

### 节点编写约定（改节点必读）

每个节点文件 = 一个 `@node_log("node_xxx")` 装饰的主函数 + 若干 `@step_log("step_N_xxx")` 步骤函数，节点名 `node_` 前缀、步骤名 `step_N_` 前缀，且**必须与 [app/utils/task_utils.py](app/utils/task_utils.py) 的 `_NODE_NAME_TO_CN` 映射 key 一致**，否则前端进度条显示英文节点名。节点内需手动 `add_running_task(...)` / `add_done_task(...)`（第三个参数传 `is_stream`，查询链才推送 SSE）。

### 跨节点数据流

- State 是 `TypedDict`，节点可**返回整个 state 或部分 dict**（如 `return {"embedding_chunks": [...]}`），LangGraph 自动合并。
- 三个默认状态工厂在各自 `state.py`：`create_default_state` / `create_query_default_state` / `copy_query_state`，均基于 `copy.deepcopy` 防污染。
- **任务状态与结果全部存在进程内存**（[app/utils/task_utils.py](app/utils/task_utils.py) 的 `_tasks_*` 字典），重启即丢；导入进度靠前端轮询 `/status/{task_id}`，查询进度靠 SSE。

### 关键共享设施

- **提示词**：`prompts/*.prompt` 纯文本模板，用 `load_prompt(name, **kwargs)` 读取后 `str.format()` 渲染。占位符里的字面 `{}` 必须写成 `{{}}`。
- **LLM**：[app/lm/lm_utils.py](app/lm/lm_utils.py) 的 `get_llm_client(json_mode=)` 返回带缓存的 `ChatOpenAI`，指向阿里云百炼兼容接口；固定传 `extra_body={"enable_thinking": False}`。结构化输出务必开 `json_mode=True`。
- **Embedding**：[app/lm/embedding_utils.py](app/lm/embedding_utils.py) 单例 BGE-M3，`generate_embeddings(list)` 返回 `{"dense": [...], "sparse": [{维度: 权重}]}`，稀疏向量已转 Python 原生类型。
- **Reranker**：[app/lm/reranker_utils.py](app/lm/reranker_utils.py) 单例 FlagReranker（bge-reranker-large）。
- **客户端**：`app/clients/` 下 Milvus / MinIO 均为**懒加载全局单例**，Mongo 走 `HistoryMongoTool` 单例（模块加载时预热，失败仅告警）。
- **配置**：所有配置从根目录 `.env` 读，`app/conf/*.py` 各自 `load_dotenv()` 后包成 dataclass（`lm_config` / `milvus_config` / `minio_config` / `embedding_config` / `reranker_config` / `mineru_config` / `mcp_config`）。**新增配置项走「.env → dataclass 字段 → 业务引用」三步，不要在业务代码里直接 `os.getenv`。**
- **日志**：统一 `from app.core.logger import logger`（loguru 封装，`logs/app_YYYYMMDD.log`，按 `.env`  retention 自动清理）。`node_log` / `step_log` 装饰器自带耗时与堆栈。
- **项目根**：`app/utils/path_util.py` 向上查找含 `.env` 的目录作为 `PROJECT_ROOT`，所有相对路径（prompts、output）都基于它。

### 服务端 API

| 服务 | 端点 |
|------|------|
| 导入 (8000) | `POST /upload`（多文件，后台跑导入图）、`GET /status/{task_id}` |
| 查询 (8084) | `POST /query`（`is_stream=true` 时立即返回并建 SSE 队列）、`GET /stream/{session_id}`（SSE）、`GET/DELETE /history/{session_id}` |

前端页面是同名静态 html（`app/*/page/`），查询侧 SSE 事件类型见 `SSEEvent`（ready/progress/delta/final/error）。

### 部署

- [docker/docker-compose.yml](docker/docker-compose.yml) 一套拉起全部：etcd、minio(9000/9001)、milvus standalone(19530/9091)、attu(7000)、neo4j(7474/7687)、mongodb(27017)，外加 `kb-import`(8000) / `kb-query`(8084) / `kb-bge-m3`(6006) / `kb-bge-reranker`(8102) 四个应用容器。
- 同一镜像按 `SERVICE` 环境变量切换角色；模型服务（`deploy/bge_m3_service.py`、`deploy/bge_reranker_service.py`）是独立 FastAPI，把本地权重发布成 HTTP，通过 `MODE=model` 启动。
- ⚠️ `docker/Dockerfile` 引用了 `docker/entrypoint.sh` 与 `Dockerfile.gpu`，**这两个文件当前不存在**，直接 `docker build` 会失败——需要先补 `entrypoint.sh`（按 `SERVICE`/`MODE` 分发启动命令）和 GPU 版 Dockerfile。
- ⚠️ 应用容器里的中间件地址靠 compose 的 `environment` 覆盖（`standalone:19530` 等），与 `.env` 里的地址不同，别把 compose 的覆盖当成 `.env` 的默认值。

## 依赖管理

**以 `pyproject.toml` + `uv.lock` 为准**。根目录的 `requirements.txt`（未纳入 git）是另一套版本集合，内容与 pyproject 有出入（含 neo4j、mcp、sentence-transformers、beautifulsoup4 等 pyproject 未声明的包），仅作参考，不要拿它同步依赖。

## 已知问题（改到相关代码时注意）

- [app/query_process/agent/nodes/node_rerank.py](app/query_process/agent/nodes/node_rerank.py) 第 2 行 `from venv import logger` —— 误从标准库 venv 导入，`logger` 不是项目 logger。
- [app/import_process/agent/nodes/node_item_name_recognition.py](app/import_process/agent/nodes/node_item_name_recognition.py) 第 213 行 `has_collection(milvus_config.chunks_collection)` —— 创建 item_name 集合前误判了切片集合。
- [app/import_process/agent/nodes/node_document_split.py](app/import_process/agent/nodes/node_document_split.py) 第 90 行 `md_content.replace(...)` 未接收返回值，换行符统一实际未生效。
- `app/clients/mongo_history_utils_new.py` 与 `mongo_history_utils.py` 重复，业务侧只引用后者。
- `deploy/`、`docker/`、`eval/`、`requirements.txt` 尚未提交进 git。
