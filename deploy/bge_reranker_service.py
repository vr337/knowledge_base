"""Reranker HTTP 服务。

本文件提供一个最小可部署的 FastAPI 服务，用于把本地已经下载好的
BGE Reranker 模型发布为 HTTP 接口。

接口路径：POST /v1/rerank
默认端口：6008
"""

import logging
import os
from typing import List, Optional, Tuple

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from FlagEmbedding import FlagReranker
from pydantic import BaseModel, Field
from transformers import XLMRobertaTokenizer

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局模型实例：服务进程内只加载一次，避免每次请求重复加载 GPU 模型。
_reranker_model: Optional[FlagReranker] = None


class RerankRequest(BaseModel):
    """Rerank 请求体。

    sentence_pairs 与 FlagEmbedding 的 reranker.compute_score 入参保持一致：
    每个元素都是一个二元组，格式为 (query, document_content)。

    请求示例：
    {
        "sentence_pairs": [
            ["用户问题", "候选文档内容1"],
            ["用户问题", "候选文档内容2"]
        ]
    }
    """

    sentence_pairs: List[Tuple[str, str]] = Field(
        ...,
        description="待打分的 query-document 成对列表，格式为 [[query, document], ...]",
    )


class RerankResponse(BaseModel):
    """Rerank 响应体。

    scores 与 reranker.compute_score 的返回结果保持同序：
    scores[i] 对应 sentence_pairs[i] 的相关性分数。
    """

    scores: List[float] = Field(..., description="每组 query-document pair 对应的 rerank 分数")


def _patch_xlm_roberta_tokenizer() -> None:
    """为 transformers 5.x 兼容 FlagEmbedding 的旧 tokenizer 调用。

    当前业务中的 knowledge.utils.bge_rerank_util.py 已包含相同兼容逻辑。
    FlagEmbedding 的部分版本仍会调用 XLMRobertaTokenizer.prepare_for_model，
    但该方法在 transformers 5.x 中可能不存在，因此这里按现有业务代码补齐。
    """

    if hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        return

    def prepare_for_model(self, ids, pair_ids=None, **kwargs):
        truncation = kwargs.get("truncation")
        max_length = kwargs.get("max_length")
        pair_ids = pair_ids or []

        if max_length and truncation in (True, "only_second"):
            special_tokens_count = 4 if pair_ids else 2
            max_pair_length = max_length - len(ids) - special_tokens_count
            if max_pair_length < len(pair_ids):
                pair_ids = pair_ids[: max(0, max_pair_length)]

        cls_token_id = getattr(self, "cls_token_id", None) or 0
        sep_token_id = getattr(self, "sep_token_id", None) or 2

        if pair_ids:
            input_ids = [cls_token_id] + ids + [sep_token_id, sep_token_id] + pair_ids + [sep_token_id]
        else:
            input_ids = [cls_token_id] + ids + [sep_token_id]
        return {"input_ids": input_ids}

    XLMRobertaTokenizer.prepare_for_model = prepare_for_model


def get_reranker_model() -> Optional[FlagReranker]:
    """获取 Reranker 模型实例。

    模型使用单例方式缓存到全局变量中。第一次请求时加载模型，后续请求直接复用。

    环境变量：
    - BGE_RERANKER_LARGE：模型路径，默认 /usr-data/models/BAAI/bge-reranker-large
    - BGE_RERANKER_DEVICE：运行设备，默认 cuda
    - BGE_RERANKER_FP16：是否使用 fp16，默认 True

    :return: FlagReranker 实例；加载失败时返回 None。
    """

    global _reranker_model

    if _reranker_model is not None:
        return _reranker_model

    _patch_xlm_roberta_tokenizer()

    model_path = os.getenv("BGE_RERANKER_LARGE", "/usr-data/models/BAAI/bge-reranker-large")
    device = os.getenv("BGE_RERANKER_DEVICE", "cuda")
    use_fp16 = os.getenv("BGE_RERANKER_FP16", "True").lower() in ("true", "1", "yes")

    try:
        logger.info(
            "开始加载 Reranker 模型：model_path=%s, device=%s, use_fp16=%s",
            model_path,
            device,
            use_fp16,
        )
        _reranker_model = FlagReranker(
            model_name_or_path=model_path,
            device=device,
            use_fp16=use_fp16,
        )
        logger.info("Reranker 模型加载完成")
        return _reranker_model
    except Exception as exc:
        logger.exception("Reranker 模型加载失败：%s", exc)
        return None


def compute_rerank_scores(reranker: FlagReranker, sentence_pairs: List[Tuple[str, str]]) -> List[float]:
    """计算 query-document pair 的 rerank 分数。

    本函数直接对齐当前业务代码中的调用方式：
    reranker.compute_score(sentence_pairs=query_doc_content_pairs)

    :param reranker: 已加载的 FlagReranker 模型实例。
    :param sentence_pairs: 待打分的 (query, document_content) 列表。
    :return: 与 sentence_pairs 顺序一致的分数列表。
    """

    scores = reranker.compute_score(sentence_pairs=sentence_pairs)

    # compute_score 在单条输入或不同版本依赖下，可能返回单个数值、list 或 numpy array。
    if isinstance(scores, (int, float)):
        return [float(scores)]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    return [float(score) for score in scores]


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。

    这里只注册 /v1/rerank 一个接口，保持部署实现最小化。
    """

    app = FastAPI(title="BGE Reranker Service", description="BGE Reranker 重排序打分服务")

    @app.post("/v1/rerank", response_model=RerankResponse)
    def rerank_api(request: RerankRequest) -> RerankResponse:
        """Rerank 打分接口。

        入参结构参考 reranker.compute_score：
        {
            "sentence_pairs": [["query", "document"], ...]
        }

        出参结构：
        {
            "scores": [0.1, 0.2, ...]
        }
        """

        if not request.sentence_pairs:
            raise HTTPException(status_code=400, detail="sentence_pairs 不能为空")

        reranker = get_reranker_model()
        if reranker is None:
            raise HTTPException(status_code=500, detail="Reranker 模型加载失败")

        try:
            scores = compute_rerank_scores(reranker=reranker, sentence_pairs=request.sentence_pairs)
            return RerankResponse(scores=scores)
        except Exception as exc:
            logger.exception("Rerank 打分失败：%s", exc)
            raise HTTPException(status_code=500, detail="Rerank 打分失败") from exc

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=6008)