import re
from typing import Any

from loguru import logger

from app.clients.mongo_history_utils import save_chat_message
from app.core.load_prompt import load_prompt
from app.core.logger import node_log, step_log
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    add_done_task,
    add_running_task,
    set_task_result,
    update_task_status,
)

MAX_CONTEXT_CHARS = 12000


@node_log("node_answer_output")
def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    1 判断state 中的answer是否已经存在，如果存在直接输出answer中的答案，注意判断是否需要流式输出需要则流式输出
    2 根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt 并调用llm 生成答案
    3 阶段三：调用大模型输出答案 注意判断是否需要流式输出需要则流式输出
    4 把答案写入到mongodb的history中 利用utils/mongo_history_utils.py中的save_chat_message方法
    5 做最后一次push操作（主要是为了触发前端图片渲染)
     {
        "answer": "HAK 180 烫金机的操作面板位于...（大模型生成的纯文本）...",
        "status": "completed",
        "image_urls": [
            "http://local-server/images/panel_view.jpg",
            "http://local-server/images/button_detail.jpg"
        ]
      }
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """

    # 记录任务的状态为运行中
    add_running_task(
        state.get("session_id"), "node_answer_output", state.get("is_stream")
    )
    try:
        # 步骤1：检查answer是否存在，如果存在直接输出answer中的答案
        answer_exists = step_1_check_answer(state)

        if not answer_exists:
            # 步骤2：anwser不存在时，构造prompt
            prompt = step_2_construct_prompt(state)
            # 步骤3：anwser不存在时，调用大模型生成答案
            step_3_generate_response(state, prompt)
        # 提取图片url（用于历史记录和前端展示）
        images = _extract_images_from_docs(state.get("reranked_docs") or [])
        if state.get("answer"):
            # 有"answer",将这作为ai的结果写入到mongodb
            step_4_write_history(state, images)
        # ★ 1) 先标记节点完成 → 前端看到"生成答案"打勾
        add_done_task(state["session_id"], "node_answer_output", state["is_stream"])

        #  这必须在 final 之前，否则前端断连后收不到
        if state.get("is_stream"):
            update_task_status(
                state["session_id"], TASK_STATUS_COMPLETED, push_queue=True
            )
        # 是流式，就推送给前端
        if state.get("is_stream"):
            push_to_session(
                state.get("session_id"),
                SSEEvent.FINAL,
                {
                    "answer": state["answer"],
                    "image_urls": images,  # 发送图片URL给前端
                },
            )
    finally:
        add_done_task(
            state.get("session_id"), "node_answer_output", state.get("is_stream")
        )


@step_log("step_1_check_answer")
def step_1_check_answer(state: QueryGraphState) -> bool:
    """
    检查 state 中是否已有 answer。
     - 若已存在：按需推送流式 delta（用于 SSE），并返回 True
    - 若不存在：返回 False
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        bool: "answer是否存在"
    """

    # 获取answer、is_stream
    answer = state.get("answer")
    is_stream = state.get("is_stream")
    # 判断answer是否存在
    if not answer:
        # 不存在时
        return False
    # answer存在时
    # 判断是否是流式输出
    if is_stream:
        push_to_session(
            session_id=state.get("session_id"),
            event=SSEEvent.DELTA,
            data={"delta": answer},
        )
    else:
        # 同步输出
        set_task_result(state.get("session_id"), "answer", answer)
    return True


@step_log("step_2_construct_prompt")
def step_2_construct_prompt(state: QueryGraphState) -> str:
    """
    构建 Prompt
    根据state中的问题、重新问题、历史对话、提问商品（item_names）、 重排内容 组织prompt
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        str: 构建好的提示词
    """
    # 1. 获取所需信息
    original_query = state.get("original_query", "")
    rewritten_query = state.get("rewritten_query", "")

    # 优先使用重写的问题
    question = rewritten_query or original_query

    item_names = state.get("item_names")
    reranked_docs: list[dict[str, Any]] = state.get("reranked_docs")
    history: list[dict[str, Any]] = state.get("history")

    # 创建用来上下文使用量的变量
    used = 0

    # 2 从重排内容中，提取为资料字符串，不可超过限额
    # 优先使用结构化 reranked_docs（包含 source/chunk_id/url/score），便于约束与引用
    # ---------------------------------------------------------
    # 逻辑解释：
    # 1. 遍历重排序后的文档列表 (reranked_docs)，这些文档已经按相关性从高到低排序。
    # 2. 对每个文档提取关键信息 (text, source, chunk_id, url, title, score)。
    # 3. 构造 "元数据头 + 正文" 格式的字符串，例如：
    #    "[1] [local] [chunk_id=123] [score=0.95] [title=操作手册]
    #     这里是文档的正文内容..."
    # 4. 累加字符长度，如果超过 MAX_CONTEXT_CHARS (如 12000 字符)，则停止添加，
    #    确保 Prompt 长度在 LLM 的处理范围内，避免 Token 溢出。
    # ---------------------------------------------------------

    docs = []

    for i, doc in enumerate(reranked_docs, start=1):
        # 获取每个重排序文档的text
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        # 获取其他内容
        title = (doc.get("title") or "").strip()
        score = doc.get("score")
        source = (doc.get("source") or "").strip()
        chunk_id = doc.get("chunk_id")
        url = (doc.get("url") or "").strip()

        # 先拼接成["[1]","[local]","[chunk_id=123]","[url=https://]","[score=0.95]","[title=操作手册]"]这样的列表
        meta_parts = [f"[{i}]"]
        if source:
            meta_parts.append(f"[{source}]")
        if chunk_id:
            meta_parts.append(f"[chunk_id={chunk_id}]")
        if url:
            meta_parts.append(f"[url={url}]")
        # score分数为0.0也显示，而0.0为False，所以这里判断是否为None
        if score is not None:
            # 保留4位小数
            meta_parts.append(f"[score={float(score):.4f}]")
        if title:
            meta_parts.append(f"[title={title}]")

        # 将meta_parts转换成字符串，用空格拼接
        doc = " ".join(meta_parts)
        # 预判断一下拼接后是否超出最大上下文
        if used + len(doc) > MAX_CONTEXT_CHARS:
            # 超出了，不拼接
            break
        # 没超，拼接
        docs.append(doc)
        # 更新一下使用长度,+2是包含了2个(\n\n)的长度
        used += len(doc) + 2
    # 构造最终的上下文
    context_str = "\n\n".join(docs)

    # 3. 格式化 History (历史对话)
    # ---------------------------------------------------------
    # 逻辑解释：
    # 1. 遍历历史对话记录 (history)。
    # 2. 将每轮对话格式化为 "用户: ... \n 助手: ..." 的文本块。
    # 3. 同样进行长度累加判断 (used)，确保历史记录+参考文档的总长度不超过 MAX_CONTEXT_CHARS。
    #    注意：这里的 used 变量是接着上面处理文档后的长度继续累加的，
    #    意味着如果文档占用了太多 Token，历史记录可能会被截断或完全丢弃。
    # ---------------------------------------------------------

    # 创建用来存储history字符串化的变量
    history_str = ""
    if history:
        for msg in history:
            # 获取历史记录里的role和text
            role = msg.get("role")
            text = msg.get("text")
            # 创建用来存储单端历史记录的变量
            history_text = ""
            if text:
                # 根据角色的不同进行不同的拼接
                if role == "user":
                    history_text += f"用户：{text}\n"
                elif role == "assistant":
                    history_text += f"助手：{text}\n"
            # 预判断一下拼接后是否超出最大上下文的长度
            if used + len(history_text) > MAX_CONTEXT_CHARS:
                break
            # 没超，拼接
            history_str += history_text
            # 更新一下使用长度
            used += len(history_text)
    else:
        history_str = "无历史对话"
    # 4. 格式化 Item Names (提问商品)
    item_names_str = "，".join(item_names)
    # 5. 组装 Prompt
    prompt = load_prompt(
        "answer_out",
        context=context_str,
        history=history_str,
        item_names=item_names_str,
        question=question,
    )
    logger.info(f"组装后的提示词为：{prompt}")
    state["prompt"] = prompt
    return prompt


@step_log("step_3_generate_response")
def step_3_generate_response(state: QueryGraphState, prompt: str) -> str:
    """
    调用llm生成答案，支持流式输出
    Args:
        state (QueryGraphState): 全局检索状态
        prompt (str): 提示词

    Returns:
        str: 全局检索状态
    """

    # 获取session_id和is_stream
    session_id = state.get("session_id")
    is_stream = state.get("is_stream")

    # 判断是否是流式输出
    if is_stream:
        try:
            answer = ""
            # 流式调用llm
            llm = get_llm_client()
            for chunk in llm.stream(prompt):
                # sse的方式响应
                content = chunk.content
                push_to_session(session_id, SSEEvent.DELTA, {"delta": content})
                # 拼接最终的answer
                answer += content
        except Exception as e:
            # 异常时，也要响应
            push_to_session(session_id, SSEEvent.ERROR, {"error": e})
        # 更新状态
        state["answer"] = answer
    else:
        try:
            # 非流式时
            # 直接调用模型
            llm = get_llm_client()
            llm_response = llm.invoke(prompt)
            # 获取最终answer
            answer = llm_response.content
            # 设置当前任务的返回结果
            set_task_result(session_id, "answer", answer)
            # 更新状态
            state["answer"] = answer
        except Exception as e:
            state["answer"] = "抱歉，生成回答时出现错误。"
    return state


def _extract_images_from_docs(reranked_docs: list[dict[str, Any]]) -> list[str]:
    """
    从最终的文档中提取图片的url
    核心逻辑：
    1. 遍历所有相关文档（包括本地知识库切片和联网搜索结果）。
    2. 策略一：直接检查文档的 'url' 字段（常见于联网搜索结果）。
       - 验证后缀名是否为图片格式 (.jpg, .png 等)。
    3. 策略二：使用正则表达式扫描文档 'text' 正文内容（常见于本地 Markdown 文档）。
       - 匹配 Markdown 图片语法: ![alt text](image_url)。
    4. 对提取到的 URL 进行去重处理，返回唯一图片列表。
    Args:
        reranked_docs (list[dict[str,Any]]): 重排序后的最终文档

    Returns:
        list[str]: 提取后图片的url列表
    """

    # 创建用来存储图片url的列表
    images = []

    # ---------------------------------------------------------
    # 正则表达式解释：r'!\[.*?\]\((.*?)\)'
    # 1. !\[   -> 匹配 Markdown 图片语法的开头 "![" (注意 [ 需要转义)
    # 2. .*?   -> 非贪婪匹配图片描述文本 (Alt Text)，即 [] 中间的内容
    # 3. \]    -> 匹配描述文本的结束符 "]"
    # 4. \(    -> 匹配 URL 部分的开始符 "("
    # 5. (.*?) -> 捕获组 (Group 1)：非贪婪匹配括号内的实际 URL 内容
    # 6. \)    -> 匹配 URL 部分的结束符 ")"
    # ( ... ) （不带反斜杠）：这就是 捕获组 。
    # 它的作用是告诉程序：“虽然我匹配了整个 ![...](...) 结构，但我 只要 这括号里的内容”。
    # ---------------------------------------------------------

    # 创建匹配图片的正则
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")

    for doc in reranked_docs:
        # 先判断url里是否是图片
        url: str = doc.get("url")
        if url and url.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
        ):
            # 是图片
            images.append(url)
        # 从text中提取
        text = doc.get("text")
        if text:
            matchs = pattern.findall(text)
            if matchs:
                for match in matchs:
                    if match not in images:
                        # 不在里面才添加，防止图片重复
                        images.append(match)
    return images


def step_4_write_history(state: QueryGraphState, images: list[str]) -> None:
    """
    把本轮答案写入 MongoDB history。
    Args:
        state (QueryGraphState): 全局检索状态
        images (list[str]): 提取的与之相关并且提到过的图片
    """

    answer = state.get("answer")
    if answer:
        # 作为ai回复保存到mongodb
        save_chat_message(
            session_id=state.get("session_id"),
            role="assistant",
            text=state.get("answer"),
            rewritten_query=state.get("rewritten_query"),
            item_names=state.get("item_names"),
            image_urls=images,
            message_id=None,
        )
