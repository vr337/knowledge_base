"""
BGE-M3 混合向量生成工具（本地加载 / 远端HTTP服务 双模式）

对外只暴露一个入口：generate_embeddings(texts) -> {"dense": [[...]], "sparse": [{维度: 权重}]}
上层业务（node_bge_embedding / node_search_embedding / node_item_name_confirm 等）无感知，
通过 .env 开关 BGE_M3_USE_LOCAL 切换，判断逻辑全部收敛在 get_bge_m3_ef() 工厂函数内。

远端HTTP服务接口规范（deploy/bge_m3_service.py）：
    POST {BGE_M3_HTTP_URL}/v1/embeddings
    请求体：{"embedding_documents": ["文本1", "文本2"]}
    响应体：{"dense": [[...], ...], "sparse": [{token_id: weight}, ...]}
"""
import os
from typing import Any

import numpy as np
import requests
from loguru import logger
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from app.conf.embedding_config import embedding_config

# 模型单例对象，避免重复初始化
_bge_m3_ef = None


class _FakeCSR:
    """
    模拟 scipy CSR 稀疏矩阵的最小实现。

    为什么需要它：generate_embeddings 内部按 CSR 的行区间切分稀疏向量
    （见本文件 generate_embeddings 中对 indices/indptr/data 的取值），
    而远端HTTP服务返回的 JSON 里 sparse 已经是 {token_id: weight} 的字典列表，
    无法直接提供 CSR 的三个数组。这里把字典列表逆向重建成 CSR 结构，
    使 HTTP 模式与本地模式在 generate_embeddings 内部的表现完全一致。

    注意：三个属性都用 numpy 数组承载，因为 generate_embeddings 内部调用的是
    indices[...].tolist() / data[...].tolist()，本地模型返回的也是 numpy 数组，
    类型必须对齐，否则 HTTP 模式下会 AttributeError。
    """

    def __init__(self, sparse_dicts: list[dict[int, float]]):
        """
        把 [{维度: 权重}, ...] 转换为 CSR 三段式结构
        Args:
            sparse_dicts (list[dict[int,float]]): 每条文本一个稀疏向量字典
        """
        # 非零元素的列索引（维度），所有行的非零维度按行顺序拼接
        raw_indices: list[int] = []
        # 非零元素的权重值，与 raw_indices 一一对应
        raw_data: list[float] = []
        # 行指针，长度 = 行数 + 1，indptr[i]:indptr[i+1] 即第 i 行的数据区间
        self.indptr: list[int] = [0]

        # 逐行填充：保证第 i 行的数据落在 indptr[i]:indptr[i+1] 区间内
        for sparse_dict in sparse_dicts:
            # 按维度排序，保证同一文本多次生成的向量顺序稳定（便于排查与比对）
            for token_id in sorted(sparse_dict.keys()):
                # JSON 反序列化后 key 是字符串，这里统一转回 int，与本地模型的 int64 一致
                raw_indices.append(int(token_id))
                raw_data.append(float(sparse_dict[token_id]))
            # 记录当前累计长度，作为下一行的起点
            self.indptr.append(len(raw_indices))

        # 转成 numpy 数组，与本地模型返回的 numpy 类型保持一致（关键是支持 .tolist()）
        # dtype 对齐：索引用 int64、权重用 float32，与 scipy CSR 及 BGE-M3 原生输出一致
        self.indices = np.array(raw_indices, dtype=np.int64)
        self.data = np.array(raw_data, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, int]:
        """稀疏矩阵形状（行数, 列数），列数取最大维度+1，仅为兼容性预留"""
        return len(self.indptr) - 1, (int(self.indices.max()) + 1 if len(self.indices) else 0)


class _HttpBGEM3Client:
    """
    BGE-M3 远端HTTP客户端。

    通过实现与本地 BGEM3EmbeddingFunction 相同的 encode_documents 方法，
    让上层调用方（get_bge_m3_ef 的消费者）感知不到本地/远端的差异。
    """

    def __init__(self, http_url: str, timeout: int = 120):
        """
        Args:
            http_url (str): 远端服务基础地址，如 http://127.0.0.1:6006
            timeout (int): 请求超时时间（秒），默认120
        """
        # 去掉末尾斜杠，避免拼接出 //
        self.http_url = http_url.rstrip("/")
        self.timeout = timeout
        # 与远端 deploy/bge_m3_service.py 的路由保持一致
        self.endpoint = f"{self.http_url}/v1/embeddings"
        logger.info(f"BGE-M3 HTTP客户端初始化完成，服务地址：{self.endpoint}")

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        """
        调用远端服务生成混合向量，返回值结构与本地模型的 encode_documents 保持一致。

        Args:
            texts (list[str]): 待向量化的文本列表

        Returns:
            dict[str,Any]: {"dense": [[...]], "sparse": _FakeCSR}，与本地模型同构
        """
        try:
            # 按远端接口规范构造请求体
            response = requests.post(
                self.endpoint,
                json={"embedding_documents": texts},
                timeout=self.timeout,
            )
            # 判断HTTP层是否成功
            if response.status_code != 200:
                logger.error(
                    f"BGE-M3远端服务请求失败，状态码：{response.status_code}，"
                    f"响应内容：{response.text}"
                )
                raise RuntimeError(
                    f"BGE-M3远端服务请求失败，状态码：{response.status_code}"
                )
            # 解析响应体
            result = response.json()
            dense = result.get("dense")
            sparse_dicts = result.get("sparse")
            # 校验关键字段，缺失说明服务端返回异常
            if dense is None or sparse_dicts is None:
                logger.error(f"BGE-M3远端服务响应缺少dense或sparse字段：{result}")
                raise RuntimeError("BGE-M3远端服务响应格式异常，缺少dense或sparse字段")
            # 数量一致性校验，防止向量与文本错位绑定
            if len(dense) != len(texts) or len(sparse_dicts) != len(texts):
                logger.error(
                    f"BGE-M3远端服务返回数量不匹配，输入{len(texts)}条，"
                    f"dense返回{len(dense)}条，sparse返回{len(sparse_dicts)}条"
                )
                raise RuntimeError("BGE-M3远端服务返回数量与输入不一致")

            logger.debug(f"BGE-M3远端服务返回{len(texts)}条向量，开始转换为CSR结构")
            # 关键：把字典列表转成兼容CSR的结构，使 generate_embeddings 内部逻辑零改动
            # dense 也要转 numpy：generate_embeddings 内部对 dense 同样调用 .tolist()，
            # 而 JSON 反序列化回来是 Python list，没有该方法
            return {"dense": np.array(dense, dtype=np.float32), "sparse": _FakeCSR(sparse_dicts)}
        except requests.RequestException as e:
            # 网络层异常（连接失败/超时等），直接抛出，不静默降级
            logger.error(f"BGE-M3远端服务网络请求异常：{e}")
            raise


def get_bge_m3_ef():
    """
    获取BGE-M3模型客户端（工厂函数，根据开关返回本地实例或HTTP客户端）

    切换逻辑：
    - BGE_M3_USE_LOCAL=True（默认）：本地加载 BGEM3EmbeddingFunction
    - BGE_M3_USE_LOCAL=False：返回 _HttpBGEM3Client，调用远端HTTP服务

    :return: 本地BGEM3EmbeddingFunction实例 或 _HttpBGEM3Client实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型/重复建连接
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3客户端单例已存在，直接返回实例")
        return _bge_m3_ef

    # 分支1：远端HTTP模式，不加载本地模型权重，省显存/内存
    if not embedding_config.use_local:
        if not embedding_config.http_url:
            logger.error("BGE_M3_USE_LOCAL=False 时缺少 BGE_M3_HTTP_URL 配置")
            raise ValueError("BGE_M3_USE_LOCAL=False 时缺少 BGE_M3_HTTP_URL 配置")
        _bge_m3_ef = _HttpBGEM3Client(http_url=embedding_config.http_url)
        return _bge_m3_ef

    # 分支2：本地加载模式（原有逻辑，保持不变）
    # 从环境变量加载配置，无配置则使用默认值
    # 本地有可以使用本地地址！ 没有使用 "BAAI/bge-m3" 会自动下载！ 如果云端部署也可以使用url地址！
    model_name = embedding_config.bge_m3_path or "BAAI/bge-m3"
    device = embedding_config.bge_device or "cpu"
    use_fp16 = embedding_config.bge_fp16 or False

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True
        }
    )

    try:
        # 初始化BGE-M3模型，开启原生L2归一化（适配Milvus IP内积检索）
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True  # 模型原生对稠密+稀疏向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts):
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3客户端单例（本地模型 或 HTTP客户端，由开关决定）
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        # HTTP模式下返回的 sparse 是 _FakeCSR，同样提供 indices/indptr/data 三个数组
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ].tolist()
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（解决numpy数组不可序列化问题）
        result = {
            "dense": [emb.tolist() for emb in embeddings["dense"]],  # 嵌套列表，与输入文本一一对应
            "sparse": processed_sparse  # 字典列表，模型已做L2归一化
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


"""
核心设计亮点&适配说明：
1. 模型原生归一化：开启normalize_embeddings = True，自动对稠密+稀疏向量做L2归一化，完美适配Milvus IP内积检索（单位化后IP等价于余弦，计算更快）；
2. 彻底解决NumPy类型做key问题：sparse_indices加.tolist()，将np.int64转为Python原生int，满足字典key的可哈希要求，无报错风险；
3. 稀疏值适配序列化：sparse_data加.tolist()，将np.float32转为Python原生float，支持JSON写入/接口返回/Milvus入库等所有场景；
4. 单例模式优化：模型仅初始化一次，避免重复加载耗时耗资源，提升批量处理效率；
5. 格式匹配业务调用：返回dense嵌套列表、sparse字典列表，与vector_result["dense"][0]/sparse_vector["sparse"][0]取值逻辑完美契合；
6. 分级日志覆盖：从模型初始化、向量生成到异常报错，全流程日志记录，便于生产环境问题排查；
7. 入参合法性校验：防止空列表/非列表入参导致的内部报错，提升工具类健壮性；
8. 双模式无缝切换：HTTP客户端通过_FakeCSR把JSON的字典列表重建成CSR三段式结构，
   使 generate_embeddings 内部的 indices/indptr/data 取值逻辑在两种模式下完全一致，上层业务零改动。
"""
