# 定义fastapi对象

import uuid
from typing import Annotated, Any

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.clients.mongo_history_utils import clear_history, get_recent_messages
from app.query_process.agent.main_graph import kb_query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    push_to_session,
    sse_generator,
)
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_task_result,
    update_task_status,
)
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status

app = FastAPI(title="import service", description="掌柜智库查询服务！")


# 跨域配置中间键，解决跨域问题
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

PAGE_DIR = PROJECT_ROOT / "app" / "query_process" / "page"


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


@app.get("/chat.html", response_model=None)
async def get_import_page():
    return serve_html("chat.html")


@app.get("/404.html", response_model=None)
async def get_404_page():
    return serve_html("404.html")


# 定义接口接收的数据结构
class QueryRequest(BaseModel):
    query: Annotated[str, Field(..., description="查询内容")]
    session_id: Annotated[str | None, Field(default=None, description="会话ID")]
    is_stream: Annotated[bool, Field(default=False, description="是否流式输出")]


@app.get("/health", description="健康检查")
async def health() -> dict[str, Any]:
    """
    服务健康检查
    Returns:
        dict[str,Any]: 返回结果
    """
    logger.info("健康检查接口调用成功")
    return {"ok": True}


# 创建后台任务，通过图对象处理以后的问题query
def run_query_graph(session_id: str, user_query: str, is_stream: bool = True) -> None:
    """
    后台任务：用图来处理当前会话用户的问题
    Args:
        session_id (str): 会话id
        user_query (str): 用户问题
        is_stream (bool, optional): 是否是流式调用 Defaults to True.
    """

    # 初始化状态
    init_state = create_query_default_state(
        session_id=session_id, original_query=user_query, is_stream=is_stream
    )
    try:
        # 执行图
        kb_query_app.invoke(init_state)
        # 执行图后，更新任务的全局状态为完成
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    except Exception as e:
        logger.error(f"{session_id}后台任务失败，{e}")
        # 图执行失败将任务全局状态更新为失败
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        if is_stream:
            # 如果是流式调用，将异常放到队列里,响应到前端显示
            push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query", description="查询请求，处理用户的问题")
async def query(req: QueryRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    查询请求，处理用户的问题
    1 解析参数
    2 更新任务状态
    3 调用处理流程图
    4 返回结果
    Args:
        req (QueryRequest): 查询请求参数
        background_tasks (BackgroundTasks): 后台任务对象，用来执行后台任务

    Returns:
        dict[str,Any]: _description_
    """
    # 获取session_id,user_query,is_stream
    session_id = req.session_id or str(uuid.uuid4())
    user_query = req.query
    is_stream = req.is_stream

    # 判断是否为流式调用
    if is_stream:
        # 创建当前会话对应的消息队列
        create_sse_queue(session_id)
        # 更新当前任务为处理中,并push到队列，里面包含了done_list、running_list和status,只有流式调用才显示步骤
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        logger.info(f"[{session_id}] 任务开始处理，查询内容：{user_query}")
    if is_stream:
        # 执行后台任务
        background_tasks.add_task(
            run_query_graph, session_id, user_query, is_stream
        )  # 底层会创建创建一个协程，等当前函数返回后，立即执行
        # 返回
        return {"message": "结果处理中.....", "session_id": session_id}
    else:
        # 不是流式调用，直接在当前协程执行图，然后通过session_id拿到结果
        run_query_graph(session_id, user_query, is_stream)
        answer = get_task_result(session_id, "answer", "")
        # 返回
        return {
            "message": "处理完成!",
            "session_id": session_id,
            "answer": answer,
            "done_list": [],
        }


@app.get("/stream/{session_id}", response_model=None, description="SSE 流式推送")
async def stream(session_id: str, request: Request) -> StreamingResponse:
    """
    处理sse请求
    Args:
        session_id (str): 会话id
        reqeust (Reqeust): 请求对象

    Returns:
        StreamingResponse: 流式响应
    """
    logger.info(f"[{session_id}] 客户端已建立 SSE 流式连接")
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}", description="获取最新的历史记录")
async def history(session_id: str, limit: int=10) -> dict[str, Any]:
    """
    获取历史记录
    Args:
        session_id (str): 会话id
        limit (int, optional): 获取最近的多少条. Defaults to Query(gt=1,default=10).

    Returns:
        dict[str,Any]: {"session_id": session_id, "items": history_list}
    """
    try:
        # 回去最近的历史记录
        history_list = get_recent_messages(session_id, limit)
        # 将id转换成字符串
        for history in history_list:
            history["_id"] = str(history["_id"])
        return {"session_id": session_id, "items": history_list}
    except Exception as e:
        logger.error(f"获取历史记录失败:{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"history error: {e}",
        )


@app.delete("/history/{session_id}", description="通过session_id清空历史记录")
async def clear_chat_history(session_id: str) -> dict[str, Any]:
    count = clear_history(session_id)
    return {"message": "History cleared", "deleted_count": count}
