"""
主要目标：
   1. 录用文本大模型识别当前chunks对应的item_name！用于区分不同的文档
   2. 使用嵌入式模型，将item_name生成向量存储到向量数据库
   3. 修改state[chunks] -> chunk {title parent_title part file_title content item_name => 每个赋值 }
实现步骤：
   1. 校验和取值 （file_title,chunks）
   2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
   3. 调用模型，拼接提示词，识别chunks对应item_name
   4. 修改state chunks -》 item_name
   5. item_name生成向量（稠密/稀疏）
   6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
"""

from pathlib import Path
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType
from pymilvus.client.types import IndexType, MetricType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task

# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


@node_log("node_item_name_recognition")
def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        ImportGraphState: 全局状态
    """
    # 记录任务的状态为运行中
    add_running_task(state.get("task_id"), "node_item_name_recognition")
    # 步骤1：校验和取值 （file_title,chunks）
    chunks, file_title = step_1_get_chunks_and_file_title(state)
    # 步骤2：构建上下文环境  chunks -> top 5 -> 拼接成context文本
    context = step_2_build_context(chunks)
    # 步骤3：调用模型，拼接提示词，识别chunks对应item_name
    item_name=step_3_call_llm(context, file_title)
    # 步骤4：产品主体回填，修改state chunks -> item_name
    step_4_update_chunks_and_state(state,item_name,chunks)
    # 步骤5：item_name生成向量（稠密/稀疏）
    dense_vector,sparse_vector=step_5_generate_embeddings(item_name)
    # 步骤6：存储向量到向量数据库 kb_item_name (pk / file_title / item_name / 稠密 和 稀疏)
    step_6_save_to_vector_db(file_title,item_name,dense_vector,sparse_vector)
    # 记录任务的状态为已完成
    add_done_task(state.get("task_id"),"node_item_name_recognition")
    return state


@step_log("step_1_get_chunks_and_file_title")
def step_1_get_chunks_and_file_title(state: ImportGraphState) -> tuple[list[dict[str,Any]], str]:
    """
    对chunks和file_title进行校验，并获取
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        tuple[list[dict[str,Any]], str]: (切片列表，文件标题（去掉后缀的文件名）)
    """
    # 获取chunks和file_title
    chunks = state.get("chunks")
    file_title = state.get("file_title")

    # 判断chunks是否为空，都这里啦，为空不合理抛出异常
    if not chunks:
        raise RuntimeError("chunks为空，没有任何切片")
    # 判断file_title是否为空，为空的话补充file_title
    if not file_title:
        state["file_title"] = Path(state.get("md_path")).stem

    return chunks, file_title


@step_log("step_2_build_context")
def step_2_build_context(chunks: list[dict[str,Any]]) -> str:
    """
    构建上下文环境  chunks -> top 5 -> 拼接成context文本
    产品主体：就是文档描述，方便提高后续检索准确度
    Args:
        chunks (list[dict[str,Any]]): 所有切片

    Returns:
        str: 处理好的上下文
    """
    # 创建存储切片处理结果的变量
    parts = []
    # 记录当前切片总字符数
    total_chars = 0
    # 遍历切片
    for index, chunk in enumerate(chunks):
        # 将切片组装为：切片:idx，标题:title，内容:content
        data = f"切片：{index + 1}，标题：{chunk.get('title')}，内容：{chunk.get('content')}"
        parts.append(data)

        # 记录字符数
        total_chars += len(data)
        # 判断是否超过总的字符上限
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            # 超过了就不拼接了
            break
    # 转换成字符，用'\n\n'分割
    context = "\n\n".join(parts)
    # 兜底防拼接后超过2500，超出上下文窗口，因为这里是append完再判断是否超出
    context = context[:CONTEXT_TOTAL_MAX_CHARS]

    return context


@step_log("step_3_call_llm")
def step_3_call_llm(context: str, file_title: str) -> str:
    """
    调用llm获取产品主体
    Args:
        context (str): 提示词所需的上下文
        file_title (str): 文件标题

    Returns:
        str: 产品主体(文档描述)
    """

    # 根据参数获取用户提示词和系统提示词
    human_prompt = load_prompt(
        "item_name_recognition", file_title=file_title, context=context
    )
    system_prompt = load_prompt("product_recognition_system")

    # 获取llm
    llm = get_llm_client()

    # 构建chain
    chain = llm | StrOutputParser()
    item_name = chain.invoke(
        [HumanMessage(content=human_prompt), SystemMessage(content=system_prompt)]
    )
    # 判断item_name是否为空，为空的话，用标题名替代
    if not item_name:
        item_name = file_title
    return item_name


@step_log("step_4_update_chunks_and_state")
def step_4_update_chunks_and_state(state:ImportGraphState,item_name:str,chunks:list[dict[str,Any]])->None:
    """
    产品主体回填，修改state chunks -> item_name
    Args:
        state (ImportGraphState): 全局状态
        item_name (str): 产品主体
        chunks (list[dict[str,Any]]): 切片列表
    """

    # 更新state里的item_name
    state["item_name"]=item_name
    # 更新每个chunks里每个chunk的item_name
    for chunk in chunks:
        chunk["item_name"]=item_name
    # 同步一下state里的chunks
    state["chunks"]=chunks
    


@step_log("step_5_generate_embeddings")
def step_5_generate_embeddings(item_name:str)->tuple[list[float],dict[int,float]]:
    """
    将产品主体转换成稠密向量和稀疏向量
    Args:
        item_name (str): 产品主体

    Returns:
        tuple[list[float],dict[int,float]]: (稠密向量，稀疏向量)
    """
    embeddings=generate_embeddings([item_name])
    return embeddings.get("dense")[0],embeddings.get("sparse")[0]


@step_log("step_6_save_to_vector_db")
def step_6_save_to_vector_db(file_title:str,item_name:str,dense_vector:list[float],sparse_vector:dict[int,float])->None:
    """
    存储向量到向量数据库 kb_item_name (pk / file_title / item_name / 稠密 和 稀疏)
    Args:
        file_title (str): 文件标题（去掉扩展名的md文件名）
        item_name (str): 产品主体
        dense_vector (list[float]): 稠密向量
        sparse_vector dict[int,float]: 稀疏向量
    """
    # 获取milvus客户端
    milvus_client=get_milvus_client()

    # 若milvus中没有kb_item_names集合，则创建
    if not milvus_client.has_collection(milvus_config.chunks_collection):
        # 准备集合结构
        schema=milvus_client.create_schema(
            auto_id=True, # 开启主键自动增长
            enable_dynamic_field=True # 开启动态字段，可动态添加字段（本质上就是一个特殊字段，值是json，每添加一个字段，就添加到json里）
        )
        schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True)
        schema.add_field(field_name="file_title",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name="item_name",datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name="dense_vector",datatype=DataType.FLOAT_VECTOR,dim=1024)
        schema.add_field(field_name="sparse_vector",datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 创建索引
        index_params=milvus_client.prepare_index_params()
        # 添加稠密向量索引
        index_params.add_index(
            field_name="dense_vector",
            index_type=IndexType.HNSW,
            index_name="dense_vector_index",
            metric_type="COSINE"
        )
        # 添加稀疏向量索引
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_name",
            metric_type=MetricType.IP
        )
        # 创建集合
        milvus_client.create_collection(
            milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
    # 保存前，删除同文档旧的与item_name相关的数据,防止误删其他文档同item_name的切片
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f"file_title=='{file_title}' and item_name=='{item_name}'"
    )
    # 准备要保存的数据
    data={
        "file_title":file_title,
        "item_name":item_name,
        "dense_vector":dense_vector,
        "sparse_vector":sparse_vector
    }
    # 保存
    milvus_client.insert(
        collection_name=milvus_config.item_name_collection,
        data=[data]
    )
