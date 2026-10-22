# -*- coding: utf-8 -*-
"""
基于 ragas 的 RAG 评估程序

功能：
    读取 eval/qa.csv 中的黄金问题集，逐条调用项目 RAG 主流程获取答案与检索上下文，
    再用 ragas 计算 5 个指标，最终把结果写入 eval/qa_result.csv（UTF-8 BOM，可直接用 Excel 打开）。

前置条件（本程序不会自动拉起这些服务）：
    1. Milvus 向量库、MongoDB 历史库在线，且已导入待评估的知识库数据；
    2. .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL 指向可用的大模型服务；
    3. 联网搜索节点依赖的 MCP（MCP_DASHSCOPE_BASE_URL）可用；
    4. BGE-M3 与 Reranker 本地模型权重就绪（BGE_M3_PATH / BGE_RERANKER_LARGE）。

运行方式：
    在项目根目录执行： python eval/eval.py

设计说明：
    ragas 0.4.x 的指标（Faithfulness 等）已不再是 Metric 子类，无法传给 evaluate()，
    因此这里直接 await 各指标的 ascore() 逐条打分，再自行组装结果表。
    LLM 通过 ragas 官方推荐的 llm_factory 构建（需传异步客户端 AsyncOpenAI），
    复用 app/conf/lm_config.py 中的同一套模型配置；
    Embedding 通过本文件内的 BgeM3RagasEmbedding 适配器包装项目的 BGE-M3。
"""

import asyncio
import csv
import sys
import uuid
from pathlib import Path

# ---------- 把项目根目录加入模块搜索路径，保证可以直接 import app.* ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- 统一标准输出/错误的编码为 UTF-8 ----------
# Windows 控制台默认 GBK，而 RAG 流程内部（如 node_rerank 的调试 print）会输出含 “•” 等
# 非 GBK 字符的内容，未处理时会抛 UnicodeEncodeError 并中断整个流程图。
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from app.conf.lm_config import lm_config
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.query_process.agent.main_graph import kb_query_app
from app.query_process.agent.state import create_query_default_state
from app.utils.task_utils import get_task_result

# ---------- 全局常量 ----------
EVAL_DIR = Path(__file__).resolve().parent  # 本程序所在目录（eval/）
QA_PATH = EVAL_DIR / "qa.csv"  # 输入：黄金问题集
RESULT_PATH = EVAL_DIR / "qa_result.csv"  # 输出：评估结果

# 输出 CSV 的列顺序（共 9 列，与需求一致）
RESULT_COLUMNS = [
    "question",
    "context",
    "answer",
    "ground_truth",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

# 多条检索上下文拼接时的分隔符
CONTEXT_SEPARATOR = "\n\n"

# ragas 结构化输出的最大 token 数
# ragas 默认仅 1024，中文长答案在 AnswerCorrectness 等指标上会触发
# "The output is incomplete due to a max_tokens length limit"，这里放宽到 4096
RAGAS_MAX_TOKENS = 4096


class BgeM3RagasEmbedding(BaseRagasEmbedding):
    """
    BGE-M3 的 ragas 适配器

    作用：把项目 app/lm/embedding_utils.py 中的 generate_embeddings 包装成 ragas 要求的
    BaseRagasEmbedding 接口（需实现 embed_text / aembed_text 两个抽象方法）。
    """

    def __init__(self):
        """初始化：调用父类构造函数（本项目不需要 ragas 的缓存后端）。"""
        super().__init__()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        """
        对单条文本生成稠密向量

        Args:
            text (str): 待编码文本
            **kwargs: ragas 透传的附加参数，本项目忽略

        Returns:
            list[float]: 文本的稠密向量（模型已做 L2 归一化）
        """
        # generate_embeddings 支持批量，这里取第一条结果即为单文本向量
        return generate_embeddings([text])["dense"][0]

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        """
        批量生成稠密向量（覆盖父类逐条调用的默认实现，减少模型调用次数）

        Args:
            texts (list[str]): 待编码文本列表
            **kwargs: ragas 透传的附加参数，本项目忽略

        Returns:
            list[list[float]]: 与输入顺序一一对应的向量列表
        """
        if not texts:
            return []
        # 过滤空文本时需保持顺序一致，这里直接整体编码
        return generate_embeddings(list(texts))["dense"]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        """
        异步版单条文本向量生成（BGE-M3 是同步 CPU/GPU 计算，放到线程池执行避免阻塞事件循环）

        Args:
            text (str): 待编码文本
            **kwargs: ragas 透传的附加参数，本项目忽略

        Returns:
            list[float]: 文本的稠密向量
        """
        return await asyncio.to_thread(self.embed_text, text)


def step1_load_qa(qa_path: Path = QA_PATH) -> list[dict[str, str]]:
    """
    步骤1：读取黄金问题集

    Args:
        qa_path (Path): qa.csv 路径，默认 eval/qa.csv

    Returns:
        list[dict[str, str]]: [{"question": 问题, "ground_truth": 参考答案}, ...]
    """
    if not qa_path.exists():
        raise FileNotFoundError(f"未找到黄金问题集：{qa_path}")

    # qa.csv 为 UTF-8 BOM 编码，用 utf-8-sig 读取可自动去除 BOM
    with qa_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    qa_list = []
    for row in rows:
        question = (row.get("question") or "").strip()
        ground_truth = (row.get("ground_truth") or "").strip()
        if not question:
            logger.warning(f"跳过缺少 question 的行：{row}")
            continue
        qa_list.append({"question": question, "ground_truth": ground_truth})

    logger.info(f"[步骤1] 黄金问题集读取完成，共 {len(qa_list)} 条：{qa_path}")
    return qa_list


def step2_build_ragas_llm():
    """
    步骤2：构建 ragas 使用的 LLM 客户端

    说明：ragas 0.4.x 的指标要求 llm 为 InstructorBaseRagasLLM，
    因此用官方 llm_factory 创建，并复用项目的模型配置（lm_config）。

    Returns:
        InstructorBaseRagasLLM: ragas 可用的 LLM 客户端
    """
    if not lm_config.api_key or not lm_config.base_url:
        raise ValueError("缺少大模型配置，请在 .env 中配置 OPENAI_API_KEY 与 OPENAI_BASE_URL")

    # ragas 的指标内部通过异步接口调用 LLM，必须传入异步客户端 AsyncOpenAI，
    # 传同步客户端会报 "Cannot use agenerate() with a synchronous client"。
    client = AsyncOpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url)
    ragas_llm = llm_factory(
        lm_config.llm_model,
        provider="openai",
        client=client,
        max_tokens=RAGAS_MAX_TOKENS,
    )

    logger.info(f"[步骤2] ragas LLM 初始化完成，模型={lm_config.llm_model}")
    return ragas_llm


def step3_build_ragas_embeddings() -> BgeM3RagasEmbedding:
    """
    步骤3：构建 ragas 使用的 Embedding 客户端

    Returns:
        BgeM3RagasEmbedding: 包装了项目 BGE-M3 的 ragas 嵌入适配器
    """
    embedding = BgeM3RagasEmbedding()
    logger.info("[步骤3] ragas Embedding 适配器初始化完成（BGE-M3）")
    return embedding


def step4_build_metrics(ragas_llm, ragas_embeddings) -> dict:
    """
    步骤4：构建 5 个 ragas 评估指标

    Args:
        ragas_llm (InstructorBaseRagasLLM): ragas LLM 客户端
        ragas_embeddings (BaseRagasEmbedding): ragas Embedding 客户端

    Returns:
        dict: {指标英文名: 指标实例}
    """
    # Context Precision 采用带参考答案的版本（与经典 ragas context_precision 语义一致）
    metrics = {
        "faithfulness": Faithfulness(llm=ragas_llm),
        "answer_relevancy": AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        "context_precision": ContextPrecisionWithReference(llm=ragas_llm),
        "context_recall": ContextRecall(llm=ragas_llm),
        "answer_correctness": AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    }
    logger.info(f"[步骤4] 评估指标构建完成：{list(metrics.keys())}")
    return metrics


def step5_run_rag_query(question: str, index: int) -> tuple[str, str]:
    """
    步骤5：执行 RAG 主流程，获取答案与检索上下文

    入口即 app/query_process/agent/main_graph.py 中的 kb_query_app.invoke，
    执行完成后从 state 的 reranked_docs 中取上下文、从 answer 中取答案。

    Args:
        question (str): 用户问题
        index (int): 问题序号，用于生成可区分的 session_id

    Returns:
        tuple[str, str]: (模型答案, 检索上下文拼接串)
    """
    # 每条问题使用独立 session_id，避免历史记录相互串扰
    session_id = f"eval-{index}-{uuid.uuid4()}"
    # 非流式调用，避免触发 SSE 队列相关逻辑
    init_state = create_query_default_state(
        session_id=session_id, original_query=question, is_stream=False
    )

    result_state = kb_query_app.invoke(init_state)

    # 取最终答案：
    # node_answer_output 节点没有 return state（只原地修改了传入的字典），
    # 该修改不会合并进 invoke 的返回值，因此需按 query_service 的用法从任务结果中取。
    answer = get_task_result(session_id, "answer", "") or result_state.get("answer") or ""

    # 取重排后的文档正文作为上下文
    reranked_docs = result_state.get("reranked_docs") or []
    contexts = [doc.get("text") for doc in reranked_docs if doc.get("text")]

    logger.info(f"[步骤5] 第 {index} 条执行完成，上下文切片数={len(contexts)}")
    return answer, CONTEXT_SEPARATOR.join(contexts)


async def step6_evaluate_one(
    metrics: dict, question: str, answer: str, context: str, ground_truth: str
) -> dict[str, float | None]:
    """
    步骤6：用 5 个指标对单条样本打分

    说明：指标走异步接口，这里统一 await ascore()，全程共用一个事件循环，
    避免反复 asyncio.run 导致异步 LLM 客户端绑定到已关闭的循环上。
    各指标独立捕获异常：单个指标失败只把该列置空，不影响其余指标与后续样本。

    Args:
        metrics (dict): 步骤4 构建的指标字典
        question (str): 用户问题
        answer (str): 模型答案
        context (str): 检索上下文拼接串
        ground_truth (str): 参考答案

    Returns:
        dict[str, float | None]: {指标英文名: 分数}，失败或无法计算时为 None
    """
    # ragas 需要的是上下文列表
    retrieved_contexts = [c for c in context.split(CONTEXT_SEPARATOR) if c]

    # 各指标的调用入参（键为输出列名，值为调用 lambda，便于统一异常处理）
    score_calls = {
        "faithfulness": lambda m: m.ascore(
            user_input=question, response=answer, retrieved_contexts=retrieved_contexts
        ),
        "answer_relevancy": lambda m: m.ascore(user_input=question, response=answer),
        "context_precision": lambda m: m.ascore(
            user_input=question, reference=ground_truth, retrieved_contexts=retrieved_contexts
        ),
        "context_recall": lambda m: m.ascore(
            user_input=question, retrieved_contexts=retrieved_contexts, reference=ground_truth
        ),
        "answer_correctness": lambda m: m.ascore(
            user_input=question, response=answer, reference=ground_truth
        ),
    }

    scores: dict[str, float | None] = {}
    for name, call in score_calls.items():
        try:
            # ascore() 返回 MetricResult
            result = await call(metrics[name])
            scores[name] = float(result.value)
        except Exception as e:
            logger.error(f"[步骤6] 指标 {name} 计算失败：{e}")
            scores[name] = None
    return scores


def step7_save_result(records: list[dict], result_path: Path = RESULT_PATH) -> None:
    """
    步骤7：把评估结果写入 CSV

    编码为 UTF-8 BOM，行尾 CRLF，保证 Excel 双击打开不乱码。

    Args:
        records (list[dict]): 每条问题的完整评估记录
        result_path (Path): 输出文件路径，默认 eval/qa_result.csv
    """
    with result_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\r\n")
        writer.writeheader()
        for record in records:
            # 只写约定的 9 列，缺失的分数统一写空字符串
            row = {col: record.get(col, "") for col in RESULT_COLUMNS}
            row = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(row)

    logger.success(f"[步骤7] 评估结果已保存：{result_path}（共 {len(records)} 条）")


async def main() -> None:
    """
    主流程：串联步骤1~步骤7，完成一次完整的 RAG 评估。

    整体为异步函数，5 个 ragas 指标共用同一个事件循环；RAG 主流程本身是同步调用，
    直接顺序执行（单条问题耗时远大于事件循环阻塞开销）。
    """
    logger.info("========== RAG 评估开始 ==========")

    # 步骤1：读取黄金问题集
    qa_list = step1_load_qa()

    # 步骤2、3、4：构建 ragas 的 LLM、Embedding 与评估指标
    ragas_llm = step2_build_ragas_llm()
    ragas_embeddings = step3_build_ragas_embeddings()
    metrics = step4_build_metrics(ragas_llm, ragas_embeddings)

    # 逐条执行 RAG 流程并评估
    records = []
    for index, qa in enumerate(qa_list, start=1):
        question = qa["question"]
        ground_truth = qa["ground_truth"]
        logger.info(f"---------- 第 {index}/{len(qa_list)} 条：{question} ----------")

        # 步骤5：执行 RAG 主流程
        try:
            answer, context = step5_run_rag_query(question, index)
        except Exception as e:
            logger.error(f"第 {index} 条 RAG 流程执行失败：{e}")
            answer, context = "", ""

        # 步骤6：计算 5 个指标
        scores = await step6_evaluate_one(metrics, question, answer, context, ground_truth)

        records.append(
            {
                "question": question,
                "context": context,
                "answer": answer,
                "ground_truth": ground_truth,
                **scores,
            }
        )

    # 步骤7：保存结果
    step7_save_result(records)

    logger.info("========== RAG 评估结束 ==========")


if __name__ == "__main__":
    asyncio.run(main())
