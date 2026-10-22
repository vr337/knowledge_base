"""
1. 检查数据 chunks是否存在
2. 前置准备工作 准备 milvus的集合和字段等
3. 删除旧数据
4. 查询chunks的数据即可
"""

from typing import Any

from loguru import logger
from pymilvus import DataType, IndexType, MilvusClient
from pymilvus.client.types import MetricType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200  # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20


@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
        为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
        未来要实现:
        1. 连接 Milvus。
        2. 根据 item_name 删除旧数据 (幂等性)。
        3. 批量插入新的向量数据。
    Args:
        state (ImportGraphState): 全局的状态

    Returns:
        ImportGraphState: 全局的状态
    """
    # 记录任务的状态为运行中
    add_running_task(state.get("task_id"), "node_import_milvus")
    # 1. 检查数据 chunks是否存在
    chunks = step_1_validate_input(state)
    # 2. 前置准备工作 创建 Milvus 集合和字段
    milvus_client = step_2_prepare_collection()
    # 3. 删除旧数据
    step_3_delete_old_data(
        milvus_client, state.get("file_title"), state.get("item_name")
    )
    # 4. 插入chunks的数据返回携带chunk_id的chunks
    chunks_with_id = step_4_insert_collections(milvus_client, chunks)
    # 更新state
    state["chunks"] = chunks_with_id
    # 记录任务的状态为已完成
    add_done_task(state.get("task_id"), "node_import_milvus")
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
    chunks = state.get("chunks")
    # 校验：必须是非空列表，否则无法进行向量化
    if not isinstance(chunks, list) or not chunks:
        logger.error("向量化输入校验失败：chunks字段为空或非有效列表")
        raise ValueError("错误: 无有效文本切片数据，无法执行向量化处理")
    logger.info(f"向量化输入校验通过，待处理文本切片数量：{len(chunks)}")
    return chunks

@step_log("step_2_prepare_collection")
def step_2_prepare_collection() -> MilvusClient:
    """
    准备和创建chunks对应的集合
    Returns:
        MilvusClient: 准备好集合的milvus客户端
    """
    # 获取milvus客户端
    milvus_client = get_milvus_client()
    # 创建集合，集合不存在时创建
    if not milvus_client.has_collection(
        collection_name=milvus_config.chunks_collection
    ):
        # 准备集合结构
        schema = milvus_client.create_schema(
            auto_id=True,  # 开启主键自动增长
            enable_dynamic_field=True,  # 开启动态字段，可动态添加字段（本质上就是一个特殊字段，值是json，每添加一个字段，就添加到json里）
        )
        schema.add_field(
            field_name="chunk_id", datatype=DataType.INT64, is_primary=True
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="title", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        # 准备索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type=IndexType.HNSW,
            index_name="dense_vector_index",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type=MetricType.IP,
        )

        # 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params,
        )
    return milvus_client

@step_log("step_3_delete_old_data")
def step_3_delete_old_data(
    milvus_client: MilvusClient, file_title: str, item_name: str
) -> None:
    """
    删除旧数据 根据item_name删除
    Args:
        milvus_client (MilvusClient): milvus客户端
        file_title (str): 文件标题
        item_name (str): 产品主体
    """

    # 删除，误删同item_name，不同文档的切片
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"file_title=='{file_title}' and item_name=='{item_name}'",
    )
    # 调用 load_collection() 会触发 Milvus 重新加载集合数据、刷新索引、清理已标记删除的数据，确保删除操作真正生效，避免新旧数据混杂导致检索错误。
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)

@step_log("step_4_insert_collections")
def step_4_insert_collections(
    milvus_client: MilvusClient, chunks: list[dict[str,Any]]
) -> list[dict[str,Any]]:
    """
    插入集合的数据！
    Args:
        milvus_client (MilvusClient): milvus客户端
        chunks (list[dict[str,Any]]): 最终保持到向量数据库里的切片
    Returns:
            list[dict[str,Any]]: 保存后的chunks，里面包含保存生成chunk_id，id插入回显
    """

    # 保存
    insert_result = milvus_client.insert(
        collection_name=milvus_config.chunks_collection, data=chunks
    )
    if insert_result and insert_result.get("insert_count", 0) > 0:
        logger.info(f"成功插入 {insert_result['insert_count']} 条数据")
    # 获取保存的主键
    ids = insert_result.get("ids") or insert_result.get("primary_keys")
    # 主键回填,只有数量对应上才能绑定id，要不然会出现数据错绑问题和索引越界问题
    if len(chunks) == len(ids):
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]

    return chunks
