"""
图的入口文件
"""
from typing import Literal

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from loguru import logger

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_item_name_recognition import (
    node_item_name_recognition,
)
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState

# 加载配置
load_dotenv(override=True)


# 创建工作流对象
work_flow = StateGraph(state_schema=ImportGraphState)
# 添加节点
work_flow.add_node("node_entry", node_entry)
work_flow.add_node("node_md_img", node_md_img)
work_flow.add_node("node_pdf_to_md", node_pdf_to_md)
work_flow.add_node("node_document_split", node_document_split)
work_flow.add_node("node_item_name_recognition", node_item_name_recognition)
work_flow.add_node("node_bge_embedding", node_bge_embedding)
work_flow.add_node("node_import_milvus", node_import_milvus)


# 添加条件函数,更具is_pdf_read_enabled和is_md_read_enabled来路由节点
def route_after_entry(
    state: ImportGraphState,
) -> Literal["node_md_img", "node_pdf_to_md", "__end__"]:
    # 分支1：开启MD直接导入 → 跳过PDF转MD，直接执行MD图片处理
    if state.get("is_md_read_enabled"):
        return "node_md_img"
     # 分支2：开启PDF导入 → 执行PDF转MD，再走后续流程
    elif state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    # 分支3：未开启任何导入配置 → 直接终止工作流（END是LangGraph内置结束常量）
    else:
        return END


# 注册静态顺序边
work_flow.set_entry_point("node_entry")
# 添加条件边,根据条件路由函数来判断是走node_md_img还是node_pdf_to_md还是结束
work_flow.add_conditional_edges(
    "node_entry",
    route_after_entry,
    {"node_md_img": "node_md_img", "node_pdf_to_md": "node_pdf_to_md", END: END},
)
work_flow.add_edge("node_pdf_to_md","node_md_img") # PDF转MD完成 → MD图片处理
work_flow.add_edge("node_md_img", "node_document_split") # MD处理完成 → 文档分块
work_flow.add_edge("node_document_split","node_item_name_recognition") # 分块完成 → 项目名识别
work_flow.add_edge("node_item_name_recognition","node_bge_embedding") # 项目名识别完成 → BGE向量化
work_flow.add_edge("node_bge_embedding","node_import_milvus")  # 向量化完成 → 导入Milvus向量库
work_flow.set_finish_point("node_import_milvus") # Milvus入库完成 → 工作流执行结束（END是内置结束节点）


# 创建图对象
kb_import_app= work_flow.compile()
