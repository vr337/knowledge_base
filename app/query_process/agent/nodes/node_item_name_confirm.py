import json
from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from loguru import logger

from app.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.clients.mongo_history_utils import (
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import node_log, step_log
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_item_name_confirm")
def node_item_name_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    Args:
        state (QueryGraphState): 全局的检索状态

    Returns:
        QueryGraphState: 全局的检索状态
    """
    # 记录任务的状态为运行中
    add_running_task(
        state.get("session_id"), "node_item_name_confirm", state.get("is_stream")
    )
    # 分别获取session_id,original_query,is_stream
    session_id = state["session_id"]
    original_query = state["original_query"]
    is_stream = state["is_stream"]
    # 步骤1：保存用户信息，防止中间异常而没保存成功，异常也属于消息
    message_id = save_chat_message(
        session_id,
        "user",
        original_query,
        "",
        state.get("item_names", []),
    )
    logger.debug(f"Node: 用户消息已初始保存，ID：{message_id}")
    # 步骤2：获取历史记录
    history = get_recent_messages(session_id, limit=10)
    # 步骤3：从用户的问题中提取item_names并重写用户问题
    extract_result = step_3_extract_info(original_query, history)
    # 分别获取提取的item_names和重写之后的问题rewritten_query
    item_names = extract_result.get("item_names")
    rewritten_query = extract_result.get("rewritten_query")
    # 创建用来存储对齐之后结果的字典
    align_result = {}
    # 如果有提取的产品主体，进行搜索和对齐
    if len(item_names) > 0:
        # 步骤4：通过提取的item_names在向量数据中进行检索
        query_results = step_4_vectorize_and_query(item_names)
        # 步骤5：根据检索结果进行对齐，获取最终对齐的结果
        align_result = step_5_align_item_names(query_results)
    # 步骤6：检查确认状态
    state = step_6_check_confirmation(state, align_result, history, rewritten_query)
    # 步骤7：写入最终历史
    final_state = step_7_write_history(
        state, session_id, history, rewritten_query, message_id
    )
    # 保存历史记录到状态中
    final_state["history"] = history
    # 记录任务的状态为已完成
    add_done_task(session_id, "node_item_name_confirm", is_stream)
    return state


@step_log("step_3_extract_info")
def step_3_extract_info(query: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    """
    利用LLM从当前问题以及历史会话中提取出主要询问的商品名称item_names（可多个，JSON列表形式）
    若商品名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
    Args:
        query (str): 用户当前原始查询问题（如："这个多少钱？"）
        history (list[dict[str,Any]]): 近期会话历史，每条消息含role/text等字段，格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]

    Returns:
        dict[str,Any]: 提取结果，固定包含2个字段，格式：
            {
                "item_names": ["商品名1", "商品名2", ...],  # 提取的商品名列表，无则空列表
                "rewritten_query": "改写后的完整问题"       # 包含商品名的独立问题，无则返回原始query
            }
    """
    # 代码核心步骤总结：
    # 1. 初始化准备：获取LLM客户端，拼接历史会话为文本格式，加载并拼接提示词，构造LLM调用的消息列表
    # 2. LLM调用与响应处理：调用LLM客户端获取响应，清理响应内容中的JSON代码块格式，解析为JSON字典
    # 3. 结果校验与异常处理：确保返回字典包含item_names/rewritten_query字段（缺失则补默认值），捕获所有异常并返回兜底结果

    # 1.构建历史对话文本，拼接为"{user:"用户消息1"}\n{"assistant":"ai消息1"}\n...."
    history_text = "\n".join(
        [f"{{{msg.get('role')}:{msg.get('text')}}}" for msg in history]
    )
    logger.info(f"Step 3：历史上下文准备完成(长度：{len(history_text)})")

    # 2.处理和动态拼接提示词
    """
    在f-string，{}有特殊含义，如果要表示一个'{',用'{{',反之'}'用'}}'表示，前面的花括号是转义符，是转义后面的花括号
    """
    prompt = load_prompt(
        "rewritten_query_and_itemnames", history_text=history_text, query=query
    )
    # 组织用户提示词和系统提示词
    messages = [
        SystemMessage(content="你是一个专业的客服助手，擅长理解用户意图和提取关键信息"),
        HumanMessage(content=prompt),
    ]
    # 获取模型，调用模型，涉及到网络，api_key可能会出现异常，这里捕获一下
    try:
        # 3.调用模型
        # 获取模型客户端
        llm = get_llm_client(json_mode=True)
        # 调用
        response = llm.invoke(messages)
        # 4.处理结果
        # 结果可能是json或代码块
        # 是代码块,将```json和```替换成空字符串
        result = response.content
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "")
        # 处理完后，结果确定是json，将其转换成python对象(字典)
        extract_result = json.loads(result)
        # llm可能没有生成item_names和rewritten_query，为其添加并设置默认值，列如历史记录里没有任何提产品主体就会出现这种情况
        if "item_names" not in extract_result:
            extract_result["item_names"] = []
        if "rewritten_query" not in extract_result:
            extract_result["rewritten_query"] = query

        # 5.返回
        return extract_result
    except Exception as e:
        logger.error(f"提取产品主体并且重写用户问题出现了异常：{e}")
        # 异常时返回一个默认的
        return {"item_names": [], "rewritten_query": query}


@step_log("step_4_vectorize_and_query")
def step_4_vectorize_and_query(extracted_item_names: list[str]) -> list[dict[str, Any]]:
    """
    把分析出的item_names逐个向量化（BGEM3模型），并在Milvus向量数据库(kb_item_names)中执行混合搜索，获取匹配评分
    Args:
        extract_item_names (list[str]): step3提取的商品名列表（如["苹果15", "华为P60"]）

    Returns:
        list[dict[str,Any]]: 查询结果：final_results=[
            {
                "extracted_name":从用户的问题中提取的产品主体(商品名称),
                "matches":[
                 分   {
                        "item_name":从向量数据库中检索到的产品主体,
                        "score":分数
                    },
                    ...
                ]
            },
            ...
        ]
    """

    # 创建存储最终结果的列表
    final_results = []
    try:
        # 1.获取Milvus的客户端
        milvus_client = get_milvus_client()
        # 2.判断milvus客户端是否为空，连接失败时会返回None
        if not milvus_client:
            logger.error("Milvus客户端连接失败")
            return final_results
        # 3.获取要检索的集合
        collection_name = milvus_config.item_name_collection
        # 4.判断collection_name是否为空
        if not collection_name:
            logger.error("获取产品主体的集合名称失败")
            return final_results
        # 5.获取提取的模糊的产品主体对应的稠密向量和稀疏向量
        embeddings = generate_embeddings(extracted_item_names)
        # 6.遍历提取的产品主体列表拿到索引，然后用其索引匹配对应的稠密向量和稀疏向量进行检索
        for i in range(len(extracted_item_names)):
            # 拿到对应的稠密向量和稀疏向量
            dense_vector = embeddings["dense"][i]
            sparse_vector = embeddings["sparse"][i]
            # 设置稠密向量和稀疏向量的检索方式
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vector, sparse_vector=sparse_vector, limit=5
            )
            # 进行混合检索
            """
                混合检索的结果的结构：
                [
                    [
                        {
                            'pk': 468868229533272140, 
                            'distance': 0.9151462912559509, 
                            'entity': {'item_name': 'HAK 180 烫金机'}
                        },
                        ....
                    ]
                ]
            """
            hybird_search_results = hybrid_search(
                client=milvus_client,  # Milvus的客户端
                collection_name=collection_name,  # 集合名称
                reqs=reqs,  # 检索方式
                ranker_weights=(0.8, 0.2),  # 排序的权重(稠密向量的权重，稀疏向量的权重)
                norm_score=True,  # 是否对分数进行归一化
                limit=5,  # 限制的返回条数
                output_fields=["item_name"],  # 输出的字段
            )
            # 7.处理返回结果
            # 创建存储检索的结果的列表
            matches = []
            # 判断检索结果是否为空
            if hybird_search_results and len(hybird_search_results) > 0:
                # 不为空，构建matches的每个元素，并添加到matches中
                for result in hybird_search_results[0]:
                    matches.append(
                        {
                            "item_name": result["entity"][
                                "item_name"
                            ],  # 向量数据库搜索后的精确的产品主体
                            "score": result["distance"],  # 分数
                        }
                    )
            # 构建最终返回结果
            final_results.append(
                {"extracted_name": extracted_item_names[i], "matches": matches}
            )
        return final_results
    except Exception as e:
        logger.error(f"混合检索item_name失败，{e}")


def step_5_align_item_names(
    query_results: list[dict[str, Any]],
) -> dict[str, list[str | None]]:
    """
    根据Milvus搜索评分，逐个对齐step3提取的item_names，生成「确认商品名」和「候选商品名」
    对齐规则（优先级a>b>c>d）：
            a  如果只有一个匹配结果评分高于0.85 → 直接确认该商品名
            b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
            c  如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
            d  如果无0.6分及以上结果 → 不返回任何商品名（确认+候选均为空）
    Args:
        query_results (list[dict[str,Any]]): [
            {
                "extracted_name":从用户的问题中提取的产品主体(商品名称),
                "matches":[
                 分   {
                        "item_name":从向量数据库中检索到的产品主体,
                        "score":分数
                    },
                    ...
                ]
            },
            ...
        ]
    Returns:
        dict[str,list[str|None]]: 对齐后的结果：{
        "confirmed_item_names": ["HAK 180 烫金机","Brother HAK 180 烫金机"], # 确定的产品主体
        "options": ["Brother HAK 180 烫金机"] # 可选的产品主体
    }，二者只能有一有值
    """
    # 创建用来存储已确认的item_name列表
    confirmed_item_names: list[str] = []
    # 创建用来存储待确认的item_name列表
    options: list[str] = []

    # 遍历混合查询的列表
    for result in query_results:
        # 获取从用户问题和历史记录中提取的产品主体
        extracted_name = result.get("extracted_name")
        # 获取extract_name检索的数据
        matches: list[dict[str, Any]] = result.get("matches")
        # 将检索到的matches数据根据分数score倒序排序(从大到小)
        matches.sort(key=lambda match: match.get("score"), reverse=True)
        # 判断matches是否为空
        if not matches:
            logger.warning(f"{extracted_name}没有检索到任何数据")
            continue
        # 分别获取高分(>=0.85)和中间分数(>=0.6 and < 0.85)的数据
        high = [match for match in matches if match.get("score") >= 0.85]
        middle = [match for match in matches if match.get("score") >= 0.6]

        # 从高到低判断,优先取高分的item_name，没有取中间分数的item_name
        if len(high) == 1:
            # 只有一条时，直接作为已确认的item_name
            confirmed_item_names.append(high[0].get("item_name"))
            continue
        if len(high) > 1:
            # 有多条时，优先检索与extracted_name(提取的产品主体)数据一致的item_name(检索的产品主体)
            # 创建存储已确认的item_name的遍历
            picked = None
            # 遍历high
            for item in high:
                if item.get("item_name") == extracted_name:
                    # 找到了
                    picked = item
                    break
            # 判断picked是否为None，是的话，表示没找到数据一致的，那就用最大的
            if not picked:
                # 前面已经根据分数倒叙排序了，最大的就是第一个
                picked = high[0]
            # 保存已确认的item_name
            confirmed_item_names.append(picked.get("item_name"))
            continue
        # 走到这里，以上条件都不满足，表示分数没有>=0.85的item_name,即索引的数据的score在0.6-0.85之间或<0.6
        # 判断是否有中间分数的item_name
        if len(middle) > 0:
            # 有的话，取前三个作为待确认的item_name
            for item in middle[:3]:
                options.append(item.get("item_name"))
    return {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options": list(set(options)),
    }


def step_6_check_confirmation(
    state: QueryGraphState,
    align_result: dict[str, list[str | None]],
    history: list[dict[str, Any]],
    rewritten_query: str,
) -> QueryGraphState:
    """
    检查step5对齐后的商品名状态，分3种分支更新会话状态（state），并同步更新历史消息的商品名关联
    Args:
        state (QueryGraphState): 原始会话状态，包含session_id/original_query等核心字段
        align_result (dict[str, list[str  |  None]]): step5的对齐结果（格式同step5返回值）
        history (list[dict[str,Any]]): 近期会话历史（格式同step3的history入参）
        rewritten_query (str): step3改写后的完整问题

    Returns:
        QueryGraphState: 全局的搜索状态
    """

    # 分别获取已确认和待确认的item_name的列表
    confirmed_item_names = align_result.get("confirmed_item_names", [])
    options = align_result.get("options", [])

    # 分支1：有已确认的item_name
    if confirmed_item_names:
        # 更新历史记录中item_names
        # 先获取要更新的数据的_id(mongodb插入数据时生成的id),只更新用户没有item_names的历史记录
        # 只更新空的，因为有item_name的表示已经确认的产品主体，没必要更新
        ids = [item.get("_id") for item in history if not item.get("item_names")]
        # 判断ids是否为空
        if ids:
            # 不为空时更新
            update_message_item_names(ids, confirmed_item_names)
        # 更新状态
        state["item_names"] = confirmed_item_names
        state["rewritten_query"] = rewritten_query
        # 有确定的item_names,最后节点会生成answer,删除之前可能存在的answer，防止影响后续节点
        if state.get("answer"):
            del state["answer"]
        return state
    # 分支2：有待确定的item_name
    if options:
        # 获取并拼接待确认的item_name
        options_str = "、".join(options)
        # 拼接待确认信息
        answer = f"您是想问以下哪个产品:{options_str}?请明确一下型号。"
        # 更新状态
        state["item_names"] = []
        state["answer"] = answer
        return state
    # 分支3：以上都不是，既没有已确认也么有待确认
    # 拼接回答信息
    answer = "抱歉，未找到相关产品，请提供准确型号以便我为你查询"
    # 更新状态
    state["item_names"] = []
    state["answer"] = answer
    return state


def step_7_write_history(
    state: QueryGraphState,
    session_id: str,
    history: list[dict[str, Any]],
    rewritten_query: str,
    message_id: str,
) -> QueryGraphState:
    """
    把本次处理的核心信息（用户问题、助手答案、商品名、改写查询）写入MongoDB的会话历史
    包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
    Args:
        state (QueryGraphState): step6更新后的会话状态，包含answer/item_names等字段
        session_id (str): 会话唯一标识
        history (list[dict[str,Any]]): 近期会话历史（无实际业务逻辑，预留扩展）
        rewritten_query (str): step3改写后的完整问题
        message_id (str): 本次用户问题的消息唯一ID（step2生成）

    Returns:
        QueryGraphState: 最终的会话状态（无额外修改，直接返回入参state）
    """

    # 若有answer，则作为ai的回答新增到历史记录里
    if state.get("answer"):
        save_chat_message(session_id, "assistant", state.get("answer"), "", [])
    # 不管有没有answer都更新历史记录
    save_chat_message(
        session_id=session_id,  # 会话ID，关联所属会话
        role="user",  # 消息角色：用户
        text=state.get("original_query"),  # 消息内容：用户原始查询
        rewritten_query=rewritten_query,  # 补充step3改写后的完整问题
        item_names=state.get("item_names"),  # 补充关联的商品名列表
        message_id=message_id,  # 消息ID，指定更新已存在的用户消息（而非新增）
    )
    return state
