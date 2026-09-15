# 定义fastapi对象

import shutil
import uuid
from datetime import datetime
from typing import Any

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, status

app = FastAPI(title="import service", description="掌柜智库导入服务！")


# 跨域配置中间键，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

PAGE_DIR = PROJECT_ROOT / "app" / "import_process" / "page"


def serve_html(filename: str) -> FileResponse | RedirectResponse:
    """统一提供 HTML 页面"""
    html_path = PAGE_DIR / filename
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    # 如果是 404 页面本身不存在，直接报 500 或 404 异常
    if filename == "404.html":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="404 page not found"
        )
    # 其他页面跳转到 404
    return RedirectResponse(url="/404.html", status_code=status.HTTP_302_FOUND)


@app.get("/import.html", response_model=None)
async def get_import_page():
    return serve_html("import.html")


@app.get("/404.html", response_model=None)
async def get_404_page():
    return serve_html("404.html")


def run_graph_task(task_id: str, local_file_path: str, local_dir: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看
    Args:
        task_id (str): 全局唯一任务ID，关联单个文件的全流程处理
        local_file_path (str): 该任务的本地文件存储目录（含临时文件/解析结果）
        local_dir (str): 上传文件的本地绝对路径
    """
    try:
        # 1.更新任务全局状态为：处理中
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        # 2. 创建LangGraph状态，填充必要参数
        state = create_default_state(
            task_id=task_id, local_file_path=local_file_path, local_dir=local_dir
        )
        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        for event in kb_import_app.stream(state):
            for node_name, node_result in event.items():
                # 每个节点完成后标记为已完成，这里可以不写，因为每个节点完成后已经标记为已完成
                add_done_task(task_id, node_name)
        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.error(
            f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True
        )


@app.post(
    "/upload",
    summary="文件上传接口",
    description="支持多文件批量上传，自动触发知识库导入全流程",
)
async def upload_files(
    background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)
) -> dict[str, Any]:
    """
    文件上传核心接口（不上传 MinIO）
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    4. 实时更新任务状态，供前端轮询监控进度
    Args:
        background_tasks (BackgroundTasks): FastAPI后台任务对象，用于异步执行LangGraph流程
        files (list[UploadFile], optional):前端上传的文件列表（form-data格式）

    Returns:
        dict[str,Any]: 包含上传结果和所有任务ID的JSON响应
    """

    # 1. 构建本地存储根目录：项目根目录/output/YYYYMMDD（按日期分层，方便管理）
    today_str = datetime.now().strftime("%Y%m%d")
    data_based_root_dir = PROJECT_ROOT / "output" / today_str

    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []
    # 2. 遍历上传的文件，给每个文件生成一个task_id
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(
            f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}"
        )

        # 3.将当前任务的文件上传步骤标记为正在运行,前端轮询可查
        add_running_task(task_id, "upload_file")
        # 4.构建每个文件的存储目录,项目根目录/output/YYYYMMDD/task_id
        task_local_dir_path = data_based_root_dir / task_id
        # 目录不存在时创建
        if not task_local_dir_path.exists():
            task_local_dir_path.mkdir(parents=True, exist_ok=True)
        # 5.构建每个文件的存储路径
        task_local_path = task_local_dir_path / file.filename

        # 6.将上传的文件复制到指定目录
        with task_local_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        # 7.上传成功后，将当前任务的文件上传步骤标记已完成
        add_done_task(task_id, "upload_file")
        # 8.后台异步执行图
        background_tasks.add_task(
            run_graph_task, task_id, str(task_local_path), str(task_local_dir_path)
        )
    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID
    logger.info(
        f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}"
    )
    return {
        "code": 200,
        "message": f"Files uploaded successfully, total: {len(files)}",
        "task_ids": task_ids,
    }


@app.get(
    "/status/{task_id}",
    summary="任务状态查询",
    description="根据TaskID查询单个文件的处理进度和全局状态",
)
async def get_task_progress(task_id: str) -> dict[str, Any]:
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_uti1s.py），高性能无Io
    Args:
        task_id (str): 全局唯一任务ID（由/upload接口返回）

    Returns:
        dict[str,Any]: 包含任务全局状态，已完成节点，运行中节点的JSON响应
    """

    # 构建返回结构
    task_status_info: dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }

    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}"
    )
    return task_status_info