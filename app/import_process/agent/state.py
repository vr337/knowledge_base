


import copy
from typing import TypedDict


class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有的节点产生和消费的数据字段。
    TYpedDict让我再代码中能有自动补全和类型检查
    使用字典访问（如state["sesion_id"]、state.get("embedding_chunks")）
    """

    task_id:str # 任务唯一ID，用于追踪日志

    # --- 流程控制标记 ---
    is_md_read_enabled: bool # 是否启用Markdown读取路径
    is_pdf_read_enabled:bool # 是否启用PDF读取路径

    # --- 路径相关 --
    local_dir:str # 当前工作目录或输出目录
    local_file_path:str # 原始输入文件路径
    file_title:str # 文件标题（文件去后缀）
    pdf_path:str # PDF文件路径（如果输入时PDF）
    md_path:str # Markdown文件路径（转换后或直接输入的）

    # --- 内容数据 ---
    md_content:str # Markdown的全文内容
    chunks:list # 切片后的文本列表，包含metadata
    item_name:str # 识别出的主体名称（如："万用表"），用于检索增强

    embedding_content:list # 包含向量数据的列表，准备些入Milvus



# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}


def create_default_state(**overrides:ImportGraphState)->ImportGraphState:
    """
    创建默认状态，支持覆盖
    Args:
        **overrides: 要覆盖的字段（关键字参数解包）
    Returns:
        新的状态实例
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态，进行深拷贝进行数据隔离，防止其他创建的state影响
    state=copy.deepcopy(graph_default_state)
    # 覆盖默认值
    state.update(overrides)
    # 返回新的状态字典实例
    return state


def get_default_state()->ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)