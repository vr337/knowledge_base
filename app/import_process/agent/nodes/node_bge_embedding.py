"""
1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
"""

from typing import Any

from loguru import logger

from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        ImportGraphState: _description_
    """

    # 记录任务的状态为运行中
    add_running_task(state.get("task_id"), "node_bge_embedding")
    # 步骤1：输入数据校验，核心chunks无效则抛出异常
    texts_to_embed = step_1_validate_input(state)
    # 步骤2：批量生成双向量，为切片绑定向量字段
    final_chunks = step_2_generate_embeddings(texts_to_embed)
    # 步骤3: 输出数据处理
    state["chunks"] = final_chunks
    # 记录任务的状态为已完成
    add_done_task(state.get("task_id"), "node_bge_embedding")

    return state


@step_log("step_1_validate_input")
def step_1_validate_input(state: ImportGraphState) -> list[dict[str,Any]]:

    """
    向量化前置步骤1：输入数据有效性校验
    核心作用：
        1. 从全局状态提取待向量化的chunks切片列表
        2. 严格校验chunks类型和非空性，无有效数据则终止向量化
    Args:
        state (ImportGraphState): 全局状态
    Returns:
            list[dict[str,Any]]: 校验通过的文本切片列表，含item_name/content字段
    """

    # 获取chunks
    texts_to_embed = state.get("chunks")
    # 校验：必须是非空列表，否则无法进行向量化
    if not isinstance(texts_to_embed, list) or not texts_to_embed:
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")
    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(texts_to_embed)}")
    return texts_to_embed


@step_log("step_2_generate_embeddings")
def step_2_generate_embeddings(texts_to_embed: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """
    批量进行向量生成! 返回稠密和稀疏双向量
    核心逻辑(分批执行,每批独立异常处理)
       1. 文本拼接: item_name + 换行 + content , 强化核心特征
       2. 批量调用: 传入拼接后的文本,生成批量双向量
       3. 向量绑定: 为每个切片复制原数据.新增dense_vector和sparse_vector字段
       4. 异常兜底,每批次发生异常不影响全局处理
    Args:
        texts_to_embed (list[dict[str,Any]]): 校验通过的文本切片列表，含item_name/content字段

    Returns:
        list[dict[str,Any]]: 带向量字段的文本切片列表，异常批次保留原数据
    """

    # 存储最终结果的列表
    final_chunks = []
    # 定义每批切片的数量
    batch_size = 5
    # 切片列表的数量
    total = len(texts_to_embed)
    # 按步长遍历
    for i in range(0, total, batch_size):
        # 拿到每一批的切片
        batch_texts = texts_to_embed[i : i + batch_size]
        # 使用异常捕捉进行批量处理,防止异常导致数据丢失
        try:
            # 构造模型输入文本：拼接商品名+切片内容，增强核心特征
            input_texts = []
            # 构建模型输入文本: item_name + 换行 + content 拼接, 强化核心特征(chunk都明确item_name)
            for chunk in batch_texts:
                item_name = chunk.get("item_name")
                content = chunk.get("content")
                # 有商品名则拼接（换行分隔提升模型识别效率），无则直接使用内容
                # 几乎所有的 Embedding 模型（尤其是基于 BERT 架构的），对前 128 个 token 的注意力是最集中的。越往后的词，对最终向量方向的拉扯力越弱。
                # **“核心词前置”**的原则
                # 方案 1：用强标点代替换行（最简单、最推荐）
                # 优化前：苹果手机\n性能很好...
                # 优化后：苹果手机。性能很好...
                # 方案2：加一点“微量”的语义胶水（适合属性明确的场景）
                # 优化切片片
                text = f"商品：{item_name}，介绍：{content}" if item_name else content
                input_texts.append(text)

            # 对优化后的切片使用嵌入模型向量化，方便后续通过item_name检索
            docs_embeddings = generate_embeddings(input_texts)
            # 当前批生成的向量为空时，直接保留原切片数据,直接进入下一批，方便人工排查
            if not docs_embeddings:
                logger.error("向量化结果为空：请检查输入文本是否为空")
                final_chunks.extend(batch_texts)
                continue
            # 向量与批次的数量没对应不绑，也是直接当前批次添加进去
            dense_vecs = docs_embeddings.get("dense", [])
            sparse_vecs = docs_embeddings.get("sparse", [])
            current_batch_size = len(batch_texts)
            # 稠密和稀疏必须同时满足数量要求，当前批次的数量必须与向量的保持一致，强行绑定的话，会出现绑定错乱的情况，切片A绑定切片B的向量这种情况，这里直接保留原切片数据，后续人工排查
            if (
                len(dense_vecs) != current_batch_size
                or len(sparse_vecs) != current_batch_size
            ):
                logger.error(
                    f"向量数量不匹配，跳过整批。"
                    f"期望:{batch_size}, 稠密实际:{len(dense_vecs)}, 稀疏实际:{len(sparse_vecs)}。"
                    f"涉及数据: {[doc.get('item_name') for doc in batch_texts]}"
                )
                final_chunks.extend(batch_texts)
                continue

            # 有对应的向量列表时，并且数量对应上时，为当前批次的切片绑定对应的向量,切片的索引与向量的索引是一一对应的
            for j, doc in enumerate(batch_texts):
                # 进行浅拷贝，防止字典被硬生生加上了 dense_vector 和 sparse_vector 字段，从而污染texts_to_embed
                item = doc.copy()
                item["dense_vector"] = dense_vecs[j]  # 绑定稠密向量
                item["sparse_vector"] = sparse_vecs[j]  # 绑定稀疏向量

                # 将关联向量的切片添加到输出列表里
                final_chunks.append(item)
        except Exception as e:
            # 捕获异常，记录错误信息并跳过当前批次，对当前批次兜底
            logger.error(f"向量化批次处理异常：{e}")
            # 异常时批次保留原切片数据，保证数据完整性，后续可人工排查
            final_chunks.extend(batch_texts)
    return final_chunks
