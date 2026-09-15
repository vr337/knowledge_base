import copy
from typing import TypedDict


class QueryGraphState(TypedDict):
    """
    QueryGraphstate定义了整个查询流程中流转的数据结构
    """

    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 不同向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF融合排序后的切牌你
    reranked_docs: list  # 重排序后的最终Top-K文档

    # 生成过程中的数据
    prompt: str  # RRF融合排序后的切片
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: list[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记


# 默认状态
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    Returns:
        QueryGraphState: 全局的检索状态
    """

    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """
    获取干净状态
    Returns:
        QueryGraphState: 全局的检索状态
    """
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """

    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state
