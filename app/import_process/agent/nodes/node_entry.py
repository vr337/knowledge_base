from pathlib import Path

from loguru import logger

from app.core.logger import node_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_entry")
def node_entry(state: ImportGraphState)->ImportGraphState:
    """
    节点: 入口节点 (node_entry)
         为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
         未来要实现:
             1. 进行任务状态记录,开始和结束列表记录
             2. 根据state中 local_file_path属性判断数据类型进而修改
                相关参数, is_md_read_enabled 或者 is_pdf_read_enabled
                         md_path 或者 pdf_path
             3. 不可解析结果类型不可用,直接输出对应警告日志! 逻辑路由节点会自动处理
             4. 获取file_tile标识,用于后期识别pdf对应的主体(item_name)进行兜底
    Args:
        state (ImportGraphState): 全局的状态

    Returns:
        ImportGraphState: 全局的状态
    """
    # 1.记录任务的节点的开始，用task_id隔离不同的用户,用于任务监控面板，展示节点执行进度(fastapi使用)
    add_running_task(state.get("task_id"), "node_entry")
    # 2.判断文件路径是否为空
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        # 没有输入路径，无法处理直接结束
        logger.warning("没有输入文件地址，无法处理，直接跳转到结束节点！")
        # 直接记录节点的完成
        add_done_task(state.get("task_id"), "node_entry")
        return state
    # 3.根据文件的类型合理的修改state中的参数，方便后续节点路由
    if local_file_path.endswith(".md"):
        # 表示上传的是md文件
        state["is_md_read_enabled"] = True
        # 记录下路径
        state["md_path"] = local_file_path
    elif local_file_path.endswith(".pdf"):
        # 表示上传的是md文件
        state["is_pdf_read_enabled"] = True
        # 记录下路径
        state["pdf_path"] = local_file_path
    else:
        logger.warning(
            f"暂不支持此类型，目前只支持md或pdf文件类型，请检查一下{local_file_path}文件类型!"
        )
        # 直接记录节点的完成
        add_done_task(state.get("task_id"), "node_entry")
        return state

    # 4获取文件标识（文件名称）
    # 基于os.path处理，拆分文件名和后缀如abc.cde.pdf，拆分后("abc.cde",".pdf")
    # file_title_os = os.path.splitext(os.path.basename(local_file_path))[0]
    # 基于Path对象获取
    file_title = Path(local_file_path).stem  # 文件名 .name  文件夹名 .parent   文件后缀
    state["file_title"] = file_title
    # 记录任务的节点的完成
    add_done_task(state.get("task_id"), "node_entry")
    return state
