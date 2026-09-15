from loguru import logger

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.conf.milvus_config import milvus_config
from app.core.logger import node_log
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_search_embedding")
def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
    流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
    Args:
        state (QueryGraphState):  会话状态字典，包含上游传递的核心信息，关键字段：
                  {
                      "session_id": str,        # 会话唯一标识
                      "rewritten_query": str,   # step3改写后的完整用户问题（含商品名）
                      "item_names": list[str],  # step6已确认的标准化商品名列表
                      "is_stream": bool/None    # 是否为流式响应，可选
                  }

    Returns:
        QueryGraphState: 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
             {
                 "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                 # 每个元素为一条匹配的向量数据，含业务字段
             }
    """
    # 记录任务的状态为运行中
    add_running_task(
        state.get("session_id"), "node_search_embedding", state.get("is_stream")
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
            return {"embedding_chunks": []}
        # 若无产品主体，则没必要查了，直接返回
        if not item_names:
            logger.warning("item_names为空，跳过检索，返回空结果")
            return {"embedding_chunks": []}
        # 3.获取重写问题的稠密向量和稀疏向量
        embeddings = generate_embeddings([rewritten_query])
        dense_vector = embeddings["dense"][0]
        sparse_vector = embeddings["sparse"][0]
        # 4.向量检索
        # 获取客户端
        milvus_client = get_milvus_client()
        # 拼接item_name作为检索条件
        expr_data = ",".join(
            [f"'{item_name}'" for item_name in item_names]
        )  # 商品名过滤表达式，缩小检索范围（仅检索指定商品名的向量），如'苹果'，'梨子'
        expr = f"item_name in [{expr_data}]"
        # 设置稠密向量和稀疏向量的检索方式
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,  # 稠密向量
            sparse_vector=sparse_vector,  # 稀疏向量
            expr=expr,  # 检索条件，一般时标量检索
            limit=5,  # 稠密/稀疏检索取TOP5
        )
        # 5.检索
        hybird_search_result = hybrid_search(
            client=milvus_client,  # Milvus客户端
            collection_name=milvus_config.chunks_collection,  # 检索的目标集合名（文本片段向量集合）
            ranker_weights=(
                0.8,
                0.2,
            ),  # 稠密/稀疏向量评分权重配比，切片内容比较多，语义占比更大些（可按照业务调优）
            norm_score=True,  # 开启评分归一化，将距离值转为0-1区间的相似度评分
            reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
            limit=5,  # 最终返回的TOP5相似度最高结果
            output_fields=["chunk_id", "content", "item_name"],  # 需要返回的字段
        )
        # 6. 构造并返回结果：若检索结果非空，取res[0]（适配Milvus批量搜索格式），否则返回空列表
        return {
            "embedding_chunks": hybird_search_result[0] if hybird_search_result else []
        }
    except Exception as e:
        logger.error(f"切片检索失败，{e}")
        return {"embedding_chunks": []}
    finally:
        # 记录任务的状态为已完成
        add_done_task(
            state.get("session_id"), "node_search_embedding", state.get("is_stream")
        )
