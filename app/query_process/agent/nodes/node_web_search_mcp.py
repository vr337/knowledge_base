import asyncio
import json

from agents.mcp import MCPServerStreamableHttp
from loguru import logger

from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_web_search_mcp")
def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    """
    节点功能，调用外部搜索引擎补充信息
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """

    # 记录当前任务的状态为进行中
    add_running_task(
        state.get("session_id"), "node_web_search_mcp", state.get("is_stream")
    )
    try:
        # 获取rewritten_query
        rewritten_query = state.get("rewritten_query")
        # 若无重写的query，则使用用户的原始问题original_query
        rewritten_query = (
            state.get("original_query") if not rewritten_query else rewritten_query
        )
        # 校验rewritten_query是否为空
        if not rewritten_query:
            logger.warning("检索重写的问题为空，返回空结果")
            return {"web_search_docs": []}
        # 创建用来存储网络搜索结果的变量
        web_search_result = []
        # 异步调用mcp网络搜索工具
        result = asyncio.run(mcp_call_streamablel(rewritten_query))
        # 将json的字符串格式text转换成字典
        text_dict = json.loads(result.content[0].text)
        web_search_result = [
            {
                "title": page.get("title", "").strip(),
                "url": page.get("url", "").strip(),
                "snippet": page.get("snippet").strip(),
            }
            for page in text_dict.get("pages")
        ]
        return {"web_search_docs":web_search_result}
    except Exception as e:
        logger.error(f"网络搜索失败，{e}")
        return {"web_search_docs": []}
    finally:
        # 记录当前任务的状态为已完成
        add_done_task(
            state.get("session_id"), "node_web_search_mcp", state.get("is_stream")
        )


async def mcp_call_streamablel(query):
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": mcp_config.mcp_base_url,
            "headers": {"Authorization": mcp_config.api_key},
            "timeout": 300,
            "sse_read_timeout": 300,
            "terminate_on_close": True,
        },
        max_retry_attempts=2,
    )
    try:
        await search_mcp.connect()
        result = await search_mcp.call_tool(
            tool_name="bailian_web_search", arguments={"query": query, "count": 5}
        )
        return result
    finally:
        await search_mcp.cleanup()
