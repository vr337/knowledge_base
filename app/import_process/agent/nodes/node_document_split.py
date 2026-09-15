"""
1. **获取与清洗内容 (Step 1)**
   从 `state` 中提取 Markdown 内容与文件标题，统一换行符格式（`\r\n` / `\r` → `\n`），保证跨平台兼容。
2. **按标题语义初切 (Step 2)**
   基于 Markdown 标题语法（`#` ~ `######`）进行**语义级切分**，自动跳过代码块内的标题匹配，避免误切注释，保证每个块语义完整。
3. **无标题文档兜底处理 (Step 2 内置)**
   若文档无任何标题，自动生成默认标题 `无主题`，确保内容不丢失、流程不中断。
4. **超长块递归精细化切割 (Step 3)**
   使用 `RecursiveCharacterTextSplitter` 对**超过指定长度**的语义块进行二次切割，按「段落 → 换行 → 句子 → 空格」优先级切割，**不产生碎片、不硬断句子、无需手动合并**。
5. **构建标准 Chunk 结构**
   为每个切片补充完整元数据：`title`、`content`、`file_title`、`parent_title`、`part` 序号，保证可检索、可溯源。
6. **本地备份与状态更新 (Step 4)**
   将切分结果备份到本地 `chunks.json` 文件，同时将最终 chunks 存入 `state`，供后续向量入库使用。
"""

import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_done_task, add_running_task

# 单个文本块最大长度（控制不超过模型上下文）
CHUNK_SIZE = 200  # 小值方便测试切割
# 块之间重叠长度（保证语义不丢失）
CHUNK_OVERLAP = 20


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        ImportGraphState: 全局状态
    """

    # 记录节点的状态为运行中
    add_running_task(state.get("task_id"), "node_document_split")
    # 步骤1：获取与清洗内容，进行state中数据清晰(md_content / file_title (做标题兜底))
    md_content, file_title = step_1_get_content(state)
    # 步骤2：通过标题进行初切，保证语义的完整,返回[{content:标题的内容,title：标题,file_title：文件名},{},{}]
    chunks = step_2_split_by_title(md_content, file_title)
    # 步骤3：进一步切割，进行语义内递归切割，保长度，返回[{content:标题的内容,title：标题,file_title：文件名,parent_title,part},{},{}]
    final_chunks = step_3_refine_chunks(chunks)
    # 步骤4：更新状态并备份切片数据到json文件中
    state["chunks"] = final_chunks
    step_4_backup_chunks(state)
    # 记录节点的状态为已完成
    add_done_task(state.get("task_id"), "node_document_split")
    return state


def step_1_get_content(state: ImportGraphState) -> tuple[str, str]:
    """
    数据清晰,处理md_content中不同系统的换成分割! 统一处理
    并且获取文件file_tile用于整个内容title兜底
    Args:
        state (ImportGraphState): 全局状态·

    Returns:
        tuple[str,str]: (清洗后的md_content,清洗后的file_title)
    """

    # 获取md_content
    md_content = state.get("md_content")
    # 判断是否为空
    if not md_content:
        logger.error("md文档内容获取失败，无法完成切分!")
        raise RuntimeError("md文档内容获取失败，无法完成切分!")
    # 2.清晰数据统一换行符号
    """
        window \r\n
        linux/mac \n
        老mac   \r
    """
    # 换行符统一为linux的换行符
    md_content.replace("\r\n", "\n").replace("\r", "\n")

    # 有file_title使用file_title的值，没有就"default_title
    file_title = state.get("file_title", "default_title")
    return md_content, file_title


@step_log("step_2_split_by_title")
def step_2_split_by_title(md_content: str, file_title: str) -> list[dict[str,Any]]:
    """
    语义切割,根据标题,进行内容切割!
    Args:
        md_content (str): 清洗后的md文档的内容
        file_title (str): 文件标题(去掉后缀的文件名)

    Returns:
        list[dict[str,Any]]: 按标题切后的结果[{content:标题的内容,title：标题,file_title：文件名},{},{}]
    """

    # 1. 定义切割正则 / md_content按行切割
    # \s* 空格 tab * 0 - n
    # #{1,6} 匹配1-6个 #
    # \s+  + 1->n   #### 标题名
    # .+ .任意字符串 + 1->n   [空格]###[空格]标题描述

    # 创建匹配标题的正则表达式
    title_pattern = re.compile(r"^#{1,6}\s+.+")
    # 是否在代码块里
    is_code_block = False
    # 当前标题
    current_title = None
    # 准备存储当前标题下的行内容
    current_lines = []
    # 准备存储所有切片
    chunks = []
    # 将md_content通过\n分割（列表中一行一行的数据）
    lines = md_content.split("\n")

    # 遍历lins，拿到每一行
    for line in lines:
        line = line.strip()
        # 判断是否遇到代码块的开头或结尾,代码块必须以"```"或"~~~"结尾，而且必须成对，开头和结尾符号必须保持一致
        if line.startswith(("```", "~~~")):
            # 表示进入代码代码块或退出代码块
            # 进入is_code_block=True
            # 退出is_code_block=False
            # 在代码块外遇到"```"表示进入代码块，变为True，在代码块里遇到"```"表示退出代码块，变为False
            is_code_block = not is_code_block
            # 将当前行添加到当前标题存储的内容里
            current_lines.append(line)
            # 在代码块里是不需要校验是否是标题行
            continue

        # 当不在代码块里，并且行是标题时
        if not is_code_block and title_pattern.match(line):
            # 判断current_title是否为空,为空时表示是遇到了第一个标题，继续并内容
            if current_title:
                # 不为空时，表示有上一个标题并且上一个标题及内容已经添加到current_lines里了，合并成块并添加到chunks里
                # 每个块的结构是{content:标题的内容,title：标题,file_title：文件名}
                chunks.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_lines),
                        "file_title": file_title,
                    }
                )
            # 更新当前标题
            current_title = line
            # 标题作为标题下行内容的第一行
            current_lines = [line]
        else:
            # 在代码块里或非标题行也添加
            current_lines.append(line)

    # 最后一块存储 (最后一次跳出循环没有保存)，没有下一个标题，导致没有合并成块，手动合并一下
    if current_title:
        # 不为空时，表示有上一个标题并且上一个标题及内容已经添加到current_lines里了，合并成块并添加到chunks里
        # 每个块的结构是{content:标题的内容,title：标题,file_title：文件名}
        chunks.append(
            {
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            }
        )

    # 没有标题行的情况，整个md_content作为块
    if not chunks:
        chunks.append(
            {
                "title": "无标题",
                "content": md_content,
                "file_title": file_title,
            }
        )

    return chunks


def step_3_refine_chunks(chunks: list[dict[str,Any]]) -> list[dict[str,Any]]:
    """
    二次切分，递归切分，保大小和补充细节
    同一标题下,同一语义,进行二次超长切割!!
    Args:
        chunks (list[dict[str,Any]]): 按标题切割数据

    Returns:
        list[dict[str,Any]]: 二次切割的结果:[{content:标题的内容,title：标题,file_title：文件名,parent_title,part},{},{}]
    """

    # 创建循环分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 切割优先级：段落 → 换行 → 句子 → 空格
        separators=["\n\n", "\n", "。", "！", "；", " "],
    )

    final_chunks = []

    for chunk in chunks:
        # 拿到每个标题块进行循环切割
        sub_chunks = splitter.split_text(chunk.get("content"))
        # 判断切割后是否为多个
        has_multiple_chunks = len(sub_chunks) > 1

        # 生成代编号的子块
        for index, sub_chunk in enumerate(sub_chunks):
            # 二次递归分割后市单个块时：title=当前title(abc)，多个块时：每个块是title=当前title_index+1(如abc_1)
            current_title = (
                f"{chunk.get('title')}_{index + 1}"
                if has_multiple_chunks
                else chunk.get("title")
            )

            # 构建子块,并添加到final_chunks里
            final_chunks.append(
                {
                    "title": current_title,  # 标题
                    "parent_title": chunk.get("title"),  # 父标题
                    "content": sub_chunk.strip(),  # 块内容
                    "file_title": chunk.get("file_title"),  # 文件标题
                    "part": index + 1,  # 第几部分
                }
            )
    return final_chunks


def step_4_backup_chunks(state: ImportGraphState) -> None:
    """
    备份切片数据到json文件中
    Args:
        state (ImportGraphState): 全局状态
    """

    # 获取存储切片的json文件按的路径

    chunks_backup_path = Path(state.get("md_path")).parent / "backup.json"

    # jsonx序列化并保存
    with open(chunks_backup_path, "wt", encoding="utf-8") as f:
        json.dump(
            state.get("chunks"),
            f,
            ensure_ascii=False,  # 为假时，中文直接原文存储
            indent=4,
        )
