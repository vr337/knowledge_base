from typing import Any

from app.core.logger import node_log
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF (Reciprocal Rank Fusion) 倒数排名融合节点
    功能：
    将来自不同检索源（如 Embedding 检索、HyDE 检索、知识图谱检索等）的结果进行融合排序。
    RRF 是一种无需训练的算法，仅根据文档在不同列表中的排名来计算最终得分。

    步骤：
    1. 提取各路检索结果：从 state 中获取 embedding_chunks 和 hyde_embedding_chunks。
    2. 结果标准化：将不同格式的检索结果统一转换为包含 chunk_id 的实体列表。
    3. 设置权重：为不同来源分配权重（当前配置：Embedding=1.0, HyDE=1.0）。
    4. 执行 RRF：计算融合分数并重新排序。
    5. 结果截断：保留 Top K 个结果。
    6. 更新状态：将融合后的结果存入 state["rrf_chunks"]
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """
    # 记录任务的状态为运行中
    add_running_task(
        state.get("session_id"), "node_rrf", state.get("is_stream")
    )
    try:
        # 步骤1：从state取出两路召回结果，并转换成只包含实体的列表
        embedding_chunks = _as_entity_list(state.get("embedding_chunks"))
        hyde_embedding_chunks = _as_entity_list(state.get("hyde_embedding_chunks"))
        # 设置多路检索的权重
        source_weights = [(embedding_chunks, 1.0), (hyde_embedding_chunks, 1.0)]
        # 步骤2：进行rrf融合及排序
        rrf_result=step_2_rrf(source_weights, k=60, max_results=10)
        # 步骤3：获取rrf融合排序之后的数据
        rrf_chunks=[chunk for chunk,score in rrf_result]
        return {"rrf_chunks":rrf_chunks}
    finally:
        # 记录任务的状态为已完成，无论任何情况都会执行，函数返回的话，会在返回之前执行
        add_done_task(
            state.get("session_id"), "node_rrf", state.get("is_stream")
        )


def _as_entity_list(chunks: list[Any]) -> list[dict[str, Any]]:
    """
    将向量检索结果进行处理，转换为只包含实体信息的列表
    兼容：
    - dict: {"entity": {..属性名和对应的字.}, "distance": ...} 或直接就是 {...}
    - pymilvus Hit: 不是 dict，但通常支持 hit.get("entity") 或 hit.entity
    其他：当作 chunk_id
    Args:
        chunks (list[Hit]): 切片列表,列表每个元素时hit对象：[
        {
            'chunk_id': 468841736962056177,
            'distance': 0.8409146070480347,
            'entity': {
                'item_name': 'HAK180烫金机',
                'content': '有关使用本设备的更多信息，请参阅使用说明书,
                "chunk_id": 468841736962056191
            }
        },...]
    Returns:
        list[dict[str,Any]]: [
        {
            "chunk_id":"468841736962056177",
            "score": 0.8409146070480347,
            "item_name": "HAK180烫金机",
            "entity": "有关使用本设备的更多信息，请参阅使用说明书"
        }
    ]
    """

    # 创建存储最终转换后结果的变量
    convert_results: list[dict[str, Any]] = []

    # 遍历每一个切片
    for chunk in chunks or []:
        # 判断当前切片是否为空，为空就下一个
        if not chunk:
            continue
        # 创建用来存储最终entity的变量
        final_entity = {}

        # ==============================================
        # 情况A：处理 Milvus 返回的 Hit 对象（含 entity、chunk_id）
        # ==============================================
        if hasattr(chunk, "entity") and hasattr(chunk, "chunk_id"):
            # 表示每个 chunk 是个 hit 对象，并且有 entity 和 chunk_id 属性
            # 取出 entity
            entity = chunk.entity
            # 创建用来接收 entity 转换后字典的变量
            raw_dict = {}

            # 1. 先把 entity 转成字典
            # 判断是否有 to_dict 方法
            if hasattr(entity, "to_dict"):
                # 表示是一个对象并且有 to_dict 属性，通过 to_dict 转化成字典
                raw_dict = entity.to_dict()
            # 判断是否一个字典
            elif isinstance(entity, dict):
                # 是字典，浅拷贝后赋值，后续添加字段就不会污染原对象
                raw_dict = entity.copy()
            else:
                # 都不是，尝试强转字典，sdk 兼容写法
                try:
                    raw_dict = dict(entity)
                except (TypeError, ValueError):
                    pass

            # 2. 核心修复：拆解套娃结构
            # 说明：pymilvus 的 to_dict() 会把外层元数据（chunk_id、distance）
            # 和内层业务数据（content）一起返回，形成嵌套的 entity 结构
            # 这里需要把内层的业务数据和外层的元数据拆开，重新拼成扁平字典
            if "entity" in raw_dict and isinstance(raw_dict["entity"], dict):
                # 真正的业务数据在内层 entity 里
                final_entity = raw_dict["entity"].copy()
                # 把外层的 chunk_id 补充到内层
                if "chunk_id" in raw_dict and "chunk_id" not in final_entity:
                    final_entity["chunk_id"] = raw_dict["chunk_id"]
                # 把外层的 distance 补充到内层，并改名为 score
                if "distance" in raw_dict and "score" not in final_entity:
                    final_entity["score"] = raw_dict["distance"]
            else:
                # 没有嵌套结构，直接用转换后的字典
                final_entity = raw_dict

            # 3. 兜底补充 chunk_id 和 score
            # 防止 to_dict() 返回的字典里缺少这两个字段
            if "chunk_id" not in final_entity:
                final_entity["chunk_id"] = chunk.chunk_id
            if hasattr(chunk, "distance") and "score" not in final_entity:
                final_entity["score"] = chunk.distance

        # ==============================================
        # 情况B：chunk 已经是字典（模拟数据 / 已格式化数据）
        # ==============================================
        elif isinstance(chunk, dict):
            # 子情况：字典嵌套 entity 结构 {entity:{...}, chunk_id:...}
            if "entity" in chunk and isinstance(chunk["entity"], dict):
                # 取出内层 entity
                ent = chunk["entity"]
                # 是字典，浅拷贝后赋值，后续添加字段就不会污染原对象
                final_entity = ent.copy()

                # 补充 chunk_id
                if "chunk_id" not in final_entity:
                    final_entity["chunk_id"] = chunk.get("chunk_id")
                # 补充 score
                # 注意：distance 在外层 chunk 字典里，不在内层 entity 里
                if "distance" in chunk and "score" not in final_entity:
                    final_entity["score"] = chunk.get("distance")
            else:
                # 是字典但没有 entity，说明是扁平字典（自身没有嵌套）
                # 浅拷贝后赋值，后续添加字段就不会污染原对象
                final_entity = chunk.copy()

        # ==============================================
        # 情况C：支持 .get() 方法的其他对象
        # ==============================================
        elif hasattr(chunk, "get"):
            # 尝试取 entity，取不到就用自己的
            entity = chunk.get("entity") or chunk
            if isinstance(entity, dict):
                # 浅拷贝后赋值，后续添加字段就不会污染原对象
                final_entity = entity.copy()

        # 判断 final_entity 是否为空，是否为字典
        if final_entity and isinstance(final_entity, dict):
            convert_results.append(final_entity)

    return convert_results


def step_2_rrf(
    source_weights: list[tuple[Any]], k: int, max_results: int
) -> list[tuple[Any]]:
    """
    rrf融合及排序，前提条件数据必须是同源，只有同源才会出现相同的数据，才会得分融合
    Args:
        source_weights (list[tuple[Any]]): 列表，每个元素是(来源chunk列表，权重)的元组，例如： [([doc1,doc2]，1.0)，([doc2,doc3]，0.8)]
        k (int): RRF常数。用于平滑排名影响，避免高排名文档站据过大优势
        max_results (int): 最多返回多少个

    Returns:
        list[tuple[Any]]: [(元素,RRF得分),....]按得分降序排序
    """

    # 创建用于存储chunk_id与对应rrf得分的映射map和chunk_id与对应的文档的映射map
    score_map = {}
    chunk_map = {}

    # 遍历source_weights,拿到每一个chunk列表和与之对应的权重
    for chunks, weight in source_weights:
        # 遍历chunk列表，拿到每个切片
        for rank, chunk in enumerate(chunks, start=1):
            # 获取chunk_id
            chunk_id = chunk.get("chunk_id")
            # rrf融合，不同路同数据(chunk_id判断是否同数据)得分累加
            # 得分公式：weight*(1.0/(k+rank))
            # 当score_map的chunk_id不存在时，表示当前chunk_id对应的数据是首次出现，那么从零开始加，0+1+2
            score_map[chunk_id]= score_map.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))
            # 融合相同的数据，存在则获取(不用),不存在则添加
            chunk_map.setdefault(chunk_id, chunk)

    # 获取数据以及对应得分的列表
    merged_chunks = [
        (chunk_map.get(chunk_id), score) for chunk_id, score in score_map.items()
    ]
    # 进行降序排序
    merged_chunks.sort(key=lambda item: item[1], reverse=True)
    # 对merged_chunks进行阶段，保留max_results个数据集
    merged_chunks = merged_chunks[:max_results]
    return merged_chunks
