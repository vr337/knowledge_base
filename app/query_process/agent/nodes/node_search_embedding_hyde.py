from typing import Any

from loguru import logger

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE (Hypothetical Document Embedding) 检索节点
    核心思想：通过LLM生成假设性答案（HyDE文档），将其向量化后用于检索，以解决短查询语义稀疏问题。

    执行步骤：
    1. 参数提取：从会话状态中获取改写后的查询（rewritten_query）和已确认的商品名（item_names）。
    2. 生成假设文档 (Step 1)：调用LLM，基于用户问题生成一段假设性的理想回答（即HyDE文档）。
    3. 混合检索 (Step 2)：
       - 将“用户问题 + 假设文档”合并，生成BGE-M3稠密+稀疏向量。
       - 在Milvus中执行混合检索（带商品名过滤），召回最相似的知识切片。
    4. 结果封装：返回检索到的切片列表和生成的假设文档，更新会话状态。
    Args:
        state (QueryGraphState): 会话状态字典，包含 session_id, rewritten_query, item_names 等

    Returns:
        QueryGraphState: 包含 hyde_embedding_chunks (检索结果) 和 hyde_doc (假设文档)的字典
    """
    # 记录任务的状态为运行中
    add_running_task(
        state.get("session_id"), "node_search_embedding_hyde", state.get("is_stream")
    )
    try:
        # 1.从会话状态中提取核心入参，为后续检索准备
        rewritten_query = state.get("rewritten_query")
        item_names = state.get("item_names")
        # 2.校验参数，补充值
        # 若无重写的query，则使用用户的原始问题original_query
        rewritten_query = (
            state.get("original_query") if not rewritten_query else rewritten_query
        )
        # 若rewritten_query还是没值，直接返回
        if not rewritten_query:
            logger.warning("检索重写的问题为空，返回空结果")
            return {"hyde_embedding_chunks": []}
        # 若无产品主体，则没必要查了，直接返回
        if not item_names:
            logger.warning("item_names为空，跳过检索，返回空结果")
            return {"hyde_embedding_chunks": []}

        hyde_doc = ""
        try:
            # 步骤1：通过rewritten_query获取假设性文档
            hyde_doc = step_1_create_hyde_doc(rewritten_query)
        except Exception as e:
            logger.error(f"获取假设性文档失败,{e}")
            return {"hyde_embedding_chunks": []}
        try:
            # 步骤2：通过hyde_doc和rewritten_query转换为向量检索数据
            result=step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names,
            )
            return {
                "hyde_embedding_chunks": result[0] if result else [],
                "hyde_doc":hyde_doc
            }
        except Exception as e:
            logger.error(f"获取假设性文档检索结果失败，{e}")
            return {"hyde_embedding_chunks": []}
    finally:
        # 记录任务的状态为已完成，无论任何情况都会执行，函数返回的话，会在返回之前执行
        add_done_task(
            state.get("session_id"), "node_search_embedding_hyde", state.get("is_stream")
        )


@step_log("step_1_create_hyde_doc")
def step_1_create_hyde_doc(rewritten_query: str) -> str:
    """
    利用大模型根据重写的用户查询生成假设性文档（Hypothetical Document）。
    HyDE的核心在于：利用LLM生成一个“虚构但相关”的文档，用该文档的向量去检索真实的文档，
    从而缓解短查询（Query）与长文档（Document）在语义空间不匹配的问题。
    Args:
        rewritten_query (str): 重写后的问题

    Returns:
        str: LLM生成的假设性文档内容
    """

    logger.info(f"Step 1: 开始生成假设性文档 (HyDE), Query: {rewritten_query}")

    # 获取llm客户端
    llm = get_llm_client()
    # 加载提示词模板
    hyde_prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    # 调用llms生成，生成假设性文档
    llm_response = llm.invoke(hyde_prompt)
    hyde_doc = llm_response.content

    logger.info(f"Step 1：假设性文档生成完成，长度：{len(hyde_doc)}字符")
    logger.debug(f"Step 1：文档预览：{hyde_doc[:50]}...")

    return hyde_doc


@step_log("step_2_search_embedding_hyde")
def step_2_search_embedding_hyde(
    rewritten_query: str,
    hyde_doc: str,
    item_names: list[str] | None = None,
    req_limit: int = 10,
    limit: int = 5,
    ranker_weights: tuple[float] = (0.8, 0.2),
    norm_score: bool = True,
    output_fields: list[str] = ["chunk_id", "content", "item_name"],
) -> Any | None:
    """
    利用“重写问题 + 假设性文档”生成 embedding，并到向量库检索切片。
    Args:
        rewritten_query (str): 重写后的问题
        hyde_doc (str): Step 1 生成的假设性文档
        item_names (list[str] | None, optional): 产品主体(商品名称)列表，用于元数据过滤 (item_name in [...])
        req_limit (int, optional): 搜索时的候选召回数量 Defaults to 10.
        limit (int, optional): 混合检索的数据量 Defaults to 5.
        ranker_weights (tuple[float], optional):调整默认权重以偏向稠密向量 Defaults to (0.8, 0.2).
        norm_score (bool, optional): 默认开启归一化 Defaults to True.
        output_fields (list[str], optional): 返回结果中包含的字段 Defaults to ["chunk_id", "content", "item_name"].

    Returns:
        Any | None: 检索结果列表
    """

    # 拼接rewritten_query和hyde_doc
    text = f"{rewritten_query}，{hyde_doc}"
    # 获取text对应的稠密向量和稀疏向量
    embeddings = generate_embeddings([text])
    dense_vector = embeddings.get("dense")[0]
    sparse_vector = embeddings.get("sparse")[0]

    # 设置item_name作为检索条件
    expr_data = ",".join([f"'{item_name}'" for item_name in item_names])
    expr = f"item_name in [{expr_data}]"
    # 设置稠密向量和稀疏向量的检索方式
    reqs = create_hybrid_search_requests(
        dense_vector=dense_vector,  # 稠密向量
        sparse_vector=sparse_vector,  # 稀疏向量
        expr=expr,  # 标量检索条件
        limit=req_limit,  # 搜索时的候选召回数量
    )
    # 获取客户端
    milvus_client = get_milvus_client()
    # 混合检索
    hybrid_search_result=hybrid_search(
        client=milvus_client,  # milvus客户端
        collection_name=milvus_config.chunks_collection,  # 检索的目标集合名（文本片段向量集合）
        reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
        ranker_weights=ranker_weights,  # 稠密/稀疏向量评分权重配比，切片内容比较多，语义占比更大些（可按照业务调优）
        norm_score=norm_score,  # 开启评分归一化，将距离值转为0-1区间的相似度评分
        limit=limit,  # 最终返回的TOP5相似度最高结果
        output_fields=output_fields,  # 需要返回的字段
    )

    return hybrid_search_result
