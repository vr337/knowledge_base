from typing import Any, Final
from venv import logger

from app.core.logger import node_log, step_log
from app.lm.reranker_utils import get_reranker_model
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 动态TopK硬上限：最多取前N条(<=10)
RERANK_MAX_TOPK: Final[int] = 10
# 最小TopK：至少保留前N条(>=1,且<=RERANK_MAX_TOPK)
RERANK_MIN_TOPK: Final[int] = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: Final[float] = 0.5
# 断崖阈值（绝对）
RERANK_GAP_ABS: Final[float] = 2.0


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """
    对检索到的文档进行重新排序，提高相关性
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """
    # 记录当前任务的状态为进行中
    add_running_task(state["session_id"], "node_rerank", state["is_stream"])
    try:
        # 步骤1：合并文档统一格式，[{text:xx,title:xx,doc_id:xx,chunk_id,url:xx,source:xx}]
        doc_items = step_1_merge_docs(state)
        # 步骤2：对文档进行重排序
        scored_docs = step_2_rerank_docs(state, doc_items)
        # 步骤3：动态topK
        topK_docs = step_3_topK(scored_docs)
        print("最终文档:", topK_docs)
        return {"reranked_docs":topK_docs}
    finally:
        # 记录当前任务的状态为已完成
        add_done_task(state["session_id"], "node_rerank", state["is_stream"])


@step_log("step_1_merge_docs")
def step_1_merge_docs(state: QueryGraphState) -> list[dict[str, Any]]:
    """
    文档合并与标准化
    目标：将多路召回（本地知识库 + 联网搜索）的异构数据，统一合并为 Reranker 模型可处理的标准格式。

    输入来源：
    1. rrf_chunks (List[Dict]): 本地知识库检索结果（经 RRF 融合排序）。
       - 结构：包含 Milvus entity 信息的复杂字典或对象。
       - 关键字段：chunk_id, content, title/item_name。
    2. web_search_docs (List[Dict]): 联网搜索结果（经 MCP 搜索返回）。
       - 结构：包含搜索摘要的扁平字典。
       - 关键字段：snippet, title, url。
    Args:
        state (QueryGraphState): 全局检索状态

    Returns:
        list[dict[Any]]: 标准化文档结果：[{text:xx,title:xx,doc_id:xx,chunk_id,url:xx,source:xx}]
    """

    # 分别获取rrf_chunks和web_search_docs
    rrf_chunks = state.get("rrf_chunks")
    web_search_docs = state.get("web_search_docs")

    # 创建最终合并的数据列表
    doc_items = []
    # 遍历rrf_chunks将其中数据转化成固定格式
    for i, chunk in enumerate(rrf_chunks):
        # 判断chunk是不是字典
        if not isinstance(chunk, dict):
            logger.warning(f"本地文档格式异常 (index={i}): {type(chunk)}")
            continue
        # 获取content
        content = chunk.get("content")
        if not content:
            # 仅在 debug 模式记录，避免生产环境日志刷屏
            logger.debug(f"跳过无内容文档 (index={i}, keys={list(chunk.keys())})")
            continue
        # 分别获取chunk_id和title(item_name)
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or ""
        title = chunk.get("title") or chunk.get("item_name") or ""

        # 将数据转换成固定机构并存储到doc_items里
        doc_items.append(
            {
                "text": content,
                "title": title,
                "chunk_id": chunk_id,
                "doc_id": chunk_id,
                "url": "",
                "source": "local",
            }
        )

    # 遍历web_search_docs，将其中的数据转换为固定的格式
    for i, doc in enumerate(web_search_docs):
        if not isinstance(doc, dict):
            logger.error(f"网络搜索文档格式异常 (index={i}): {type(chunk)}")
            continue
        # 分别获取网络搜索结果中的sinippet摘要，title标题，url网址
        snippet = (doc.get("sinippet") or doc.get("content") or "").strip()
        title = (doc.get("title") or "").strip()
        url = (doc.get("url") or "").strip()
        # 使用固定的格式存储数据
        doc_items.append(
            {
                "text": snippet,
                "title": title,
                "chunk_id": "",
                "doc_id": "",
                "url": url,
                "source": "web",
            }
        )
    return doc_items


@step_log("step_2_rerank_docs")
def step_2_rerank_docs(
    state: QueryGraphState, doc_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    对文档进行重排序
    Args:
        state (QueryGraphState): 全局检索状态
        doc_items (list[dict[str, Any]]): [{ text,doc_id}, ...]统一格式后的文档列表

    Returns:
        list[dict[str,Any]]: 重排序后的列表：[{text:xx,title:xx,score:xx,doc_id:xx,chunk_id,url:xx,source:xx}]
    """
    # 获取rewritten_query，没有original_query兜底
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""

    # 判断rewritten_query和doc_items是否为空
    if not rewritten_query or not doc_items:
        logger.warning("Step 2: 跳过重排序 (无文档或无问题)")
        return []
    # 获取需要融合排序的文本（问题对应的答案）
    texts = [item.get("text") for item in doc_items]
    try:
        # 获取rerank模型
        rerank_model = get_reranker_model()
        # 将融合排序的文本转换成[(rewritten_query,text),(问题，对应的答案)]
        sentence_pairs = [(rewritten_query, text) for text in texts]
        # 对数据进行重排序，获取对应的分数列表
        scores = rerank_model.compute_score(sentence_pairs=sentence_pairs)
        # 创建存储最终结果的列表
        scored_docs = []
        # 将scores、texts、doc_items进行压缩且遍历，组成标准格式
        for score, text, item in zip(scores, texts, doc_items):
            scored_docs.append(
                {
                    "text": text,
                    "score": float(score),
                    "title": item.get("title"),
                    "chunk_id": item.get("chunk_id"),
                    "doc_id": item.get("doc_id"),
                    "url": item.get("url"),
                    "source": item.get("source"),
                }
            )
            # 将最终结果进行降序排序
        scored_docs.sort(key=lambda item: item["score"], reverse=True)
        return scored_docs
    except Exception as e:
        logger.error(f"使用rerank模型排序失败，{e}")
        return [
            {
                **item,
                "score": 0.0,  # 分数降级
            }
            for item in doc_items
        ]


@step_log("step_3_topK")
def step_3_topK(scored_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    动态 TopK（默认最多 10）
    基于 scored_docs（已按 score 降序排序）进行智能截断，
    核心逻辑：结合固定上下限+断崖阈值判断，避免机械取前N条，保留语义相关的连续文档集合
    Args:
        scored_items (list[dict[str,Any]]): 列表，元素为带score的文档字典，已按score降序排列，格式如[{"doc": 文档对象, "score": 相关性分数}, ...]

    Returns:
        list[dict[str,Any]]: 列表，动态截断后的TopK文档列表，数量≤10
    """

    max_topk = min(
        RERANK_MAX_TOPK, len(scored_items)
    )  # 硬上限，文档长度超过最大上限，按最大上限算，小于有多少，上限是多少
    min_topk = RERANK_MIN_TOPK  # 硬下限，最低要保留几个
    gap_ratio = RERANK_GAP_RATIO  # 相对断崖阈值：分数下降的相对比例阈值
    gap_abs = RERANK_GAP_ABS  # 绝对断崖阈值：分数下降的绝对差值阈值

    topk = max_topk  # 动态topK，默认为硬上限

    if topk > min_topk:
        # 遍历范围，从min_top-1到
        for i in range(min_topk - 1, max_topk - 1):
            score1 = scored_items[i].get(
                "score"
            )  # 参与断崖的前一个分数，通过了断崖判断/不参与断崖判断的分数
            score2 = scored_items[i + 1].get("score")  # 参与断崖判断的分数

            gap = score1 - score2  # 计算相邻文档的分数绝对差距（因已降序，gap≥0）
            gap_rel = (
                gap / (abs(score1) + 1e-6)
            )  # 计算相对差距(绝对差距/相邻两文档的最大分数，+1e-6避免除数为0/极小值，防止程序报错)
            if gap >= gap_abs or gap_rel >= gap_ratio:
                # 断崖啦，后面截断掉，动态topK更新
                topk = i + 1
                break

    # 根据动态topK动态截断
    topk_docs = scored_items[:topk]

    return topk_docs
