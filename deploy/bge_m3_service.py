"""BGE-M3 向量化 HTTP 服务。

本文件提供一个最小可部署的 FastAPI 服务，用于把已经下载好的本地
BGE-M3 模型发布为 HTTP 接口。
"""

import logging
import os
from typing import Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 全局模型对象：服务启动后只加载一次，避免每次请求重复加载模型。
bge_m3_ef: Optional[BGEM3EmbeddingFunction] = None


class HybridEmbeddingRequest(BaseModel):
    """混合向量生成请求体。

    字段保持和 knowledge.utils.bge_m3_embedding_util.generate_hybrid_embeddings
    的 embedding_documents 参数一致，传入需要生成向量的文本列表。
    """

    embedding_documents: List[str] = Field(..., description="需要生成向量的文本列表")


class HybridEmbeddingResponse(BaseModel):
    """混合向量生成响应体。

    dense 为稠密向量列表，sparse 为稀疏向量列表。
    sparse 中每个字典的 key 是 token_id，value 是对应权重。
    """

    dense: List[List[float]]
    sparse: List[Dict[int, float]]


def get_bge_m3_embedding_model() -> Optional[BGEM3EmbeddingFunction]:
    """获取 BGE-M3 嵌入模型实例。

    模型会被缓存在全局变量中，后续请求复用同一个模型对象。
    默认模型路径为 /usr-data/models/BAAI/bge-m3，可通过 BGE_M3_PATH 覆盖。
    默认使用 cuda 和 fp16，适合云 GPU 服务器；如需调整可设置环境变量：
    BGE_DEVICE=cpu 或 BGE_FP16=False。

    :return: BGEM3EmbeddingFunction 实例；加载失败时返回 None。
    """

    global bge_m3_ef

    if bge_m3_ef is not None:
        return bge_m3_ef

    model_name = os.getenv("BGE_M3_PATH", "/usr-data/models/BAAI/bge-m3")
    device = os.getenv("BGE_DEVICE", "cuda")
    use_fp16_str = os.getenv("BGE_FP16", "True")
    use_fp16 = use_fp16_str.lower() in ("true", "1", "yes")

    try:
        logger.info("开始加载 BGE-M3 模型，model_name=%s, device=%s, use_fp16=%s", model_name, device, use_fp16)
        bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
        )
        logger.info("BGE-M3 模型加载完成")
        return bge_m3_ef
    except Exception as e:
        logger.exception("BGE-M3 模型加载失败: %s", e)
        return None


def generate_hybrid_embeddings(
    embedding_model: BGEM3EmbeddingFunction,
    embedding_documents: List[str],
) -> Optional[Dict[str, List]]:
    """为文本列表生成 BGE-M3 混合向量。

    本函数参考 knowledge.utils.bge_m3_embedding_util.generate_hybrid_embeddings
    的处理逻辑，返回结构保持为：
    {
        "dense": [[...], ...],
        "sparse": [{token_id: weight, ...}, ...]
    }

    :param embedding_model: 已加载的 BGE-M3 嵌入模型。
    :param embedding_documents: 需要生成嵌入向量的文本列表。
    :return: 包含 dense 和 sparse 的字典；生成失败时返回 None。
    """

    try:
        embedding_result = embedding_model.encode_documents(embedding_documents)
        csr_array = embedding_result["sparse"]

        processed_sparse_result = []
        for index in range(len(embedding_documents)):
            # csr_array 使用 CSR 稀疏矩阵格式存储，这里按行取出每条文本的稀疏向量。
            start_ind_ptr = csr_array.indptr[index]
            end_ind_ptr = csr_array.indptr[index + 1]

            token_ids = csr_array.indices[start_ind_ptr:end_ind_ptr].tolist()
            weights = csr_array.data[start_ind_ptr:end_ind_ptr].tolist()
            sparse_vector = dict(zip(token_ids, weights))

            processed_sparse_result.append(sparse_vector)

        return {
            "dense": [dense_vector.tolist() for dense_vector in embedding_result["dense"]],
            "sparse": processed_sparse_result,
        }
    except Exception as e:
        logger.exception("BGE-M3 向量生成失败: %s", e)
        return None


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""

    app = FastAPI(title="BGE-M3 Embedding Service", description="BGE-M3 混合向量生成服务")

    @app.post("/v1/embeddings", response_model=HybridEmbeddingResponse)
    def generate_hybrid_embeddings_api(request: HybridEmbeddingRequest):
        """生成 BGE-M3 混合向量。

        请求示例：
        {
            "embedding_documents": ["我是中国人", "我喜欢自由"]
        }

        响应结构和 generate_hybrid_embeddings 函数保持一致，包含 dense 与 sparse。
        """

        if not request.embedding_documents:
            raise HTTPException(status_code=400, detail="embedding_documents 不能为空")

        embedding_model = get_bge_m3_embedding_model()
        if embedding_model is None:
            raise HTTPException(status_code=500, detail="BGE-M3 模型加载失败")

        embedding_result = generate_hybrid_embeddings(
            embedding_model=embedding_model,
            embedding_documents=request.embedding_documents,
        )
        if embedding_result is None:
            raise HTTPException(status_code=500, detail="BGE-M3 向量生成失败")

        return embedding_result

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=6006)