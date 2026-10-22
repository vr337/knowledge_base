"""
BGE Reranker 重排序工具（本地加载 / 远端HTTP服务 双模式）

对外入口：get_reranker_model()，返回具备 compute_score(sentence_pairs) 方法的对象。
上层业务（node_rerank）只调用 compute_score，不感知本地/远端差异，
通过 .env 开关 BGE_RERANKER_USE_LOCAL 切换，判断逻辑全部收敛在本工厂函数内。

远端HTTP服务接口规范（deploy/bge_reranker_service.py）：
    POST {BGE_RERANKER_HTTP_URL}/v1/rerank
    请求体：{"sentence_pairs": [["query", "doc"], ...]}
    响应体：{"scores": [0.9, 0.1, ...]}
"""
import requests
from loguru import logger

from app.conf.reranker_config import reranker_config

_reranker_model = None


class _HttpRerankerClient:
    """
    BGE Reranker 远端HTTP客户端。

    通过实现与本地 FlagReranker 相同的 compute_score 方法，
    让上层调用方感知不到本地/远端的差异。
    """

    def __init__(self, http_url: str, timeout: int = 120):
        """
        Args:
            http_url (str): 远端服务基础地址，如 http://127.0.0.1:6008
            timeout (int): 请求超时时间（秒），默认120
        """
        # 去掉末尾斜杠，避免拼接出 //
        self.http_url = http_url.rstrip("/")
        self.timeout = timeout
        # 与远端 deploy/bge_reranker_service.py 的路由保持一致
        self.endpoint = f"{self.http_url}/v1/rerank"
        logger.info(f"BGE-Reranker HTTP客户端初始化完成，服务地址：{self.endpoint}")

    def compute_score(self, sentence_pairs) -> list[float]:
        """
        调用远端服务对 query-document 成对列表打分。

        Args:
            sentence_pairs: [(query, document), ...] 二元组列表，
                            与本地 FlagReranker.compute_score 入参一致

        Returns:
            list[float]: 与输入顺序一致的分数列表
        """
        # 统一转成可JSON序列化的二元组列表（入参可能是tuple，JSON序列化后变成数组）
        pairs = [list(pair) for pair in sentence_pairs]
        try:
            response = requests.post(
                self.endpoint,
                json={"sentence_pairs": pairs},
                timeout=self.timeout,
            )
            # 判断HTTP层是否成功
            if response.status_code != 200:
                logger.error(
                    f"BGE-Reranker远端服务请求失败，状态码：{response.status_code}，"
                    f"响应内容：{response.text}"
                )
                raise RuntimeError(
                    f"BGE-Reranker远端服务请求失败，状态码：{response.status_code}"
                )
            # 解析响应体
            scores = response.json().get("scores")
            if scores is None:
                logger.error(f"BGE-Reranker远端服务响应缺少scores字段：{response.text}")
                raise RuntimeError("BGE-Reranker远端服务响应格式异常，缺少scores字段")
            # 数量一致性校验，防止分数与输入错位绑定
            if len(scores) != len(pairs):
                logger.error(
                    f"BGE-Reranker远端服务返回数量不匹配，输入{len(pairs)}对，返回{len(scores)}个分数"
                )
                raise RuntimeError("BGE-Reranker远端服务返回数量与输入不一致")
            # 统一转成Python原生float，与本地模型返回类型保持一致
            return [float(score) for score in scores]
        except requests.RequestException as e:
            # 网络层异常（连接失败/超时等），直接抛出，不静默降级
            logger.error(f"BGE-Reranker远端服务网络请求异常：{e}")
            raise


def get_reranker_model():
    """
    获取Reranker模型客户端（工厂函数，根据开关返回本地实例或HTTP客户端）

    切换逻辑：
    - BGE_RERANKER_USE_LOCAL=True（默认）：本地加载 FlagReranker
    - BGE_RERANKER_USE_LOCAL=False：返回 _HttpRerankerClient，调用远端HTTP服务

    Returns:
        本地FlagReranker实例 或 _HttpRerankerClient实例，
        两者都提供 compute_score(sentence_pairs) -> List[float]
    """
    global _reranker_model
    # 单例模式：已初始化则直接返回，避免重复加载模型/重复建连接
    if _reranker_model is not None:
        return _reranker_model

    # 分支1：远端HTTP模式，不加载本地模型权重，省显存/内存
    if not reranker_config.use_local:
        if not reranker_config.http_url:
            logger.error("BGE_RERANKER_USE_LOCAL=False 时缺少 BGE_RERANKER_HTTP_URL 配置")
            raise ValueError("BGE_RERANKER_USE_LOCAL=False 时缺少 BGE_RERANKER_HTTP_URL 配置")
        _reranker_model = _HttpRerankerClient(http_url=reranker_config.http_url)
        return _reranker_model

    # 分支2：本地加载模式（原有逻辑，保持不变）
    # FlagEmbedding 延迟导入：仅本地模式才需要，避免HTTP模式下引入重依赖
    from FlagEmbedding import FlagReranker

    _reranker_model = FlagReranker(
        model_name_or_path=reranker_config.bge_reranker_large,
        device=reranker_config.bge_reranker_device,
        use_fp16=reranker_config.bge_reranker_fp16
    )
    return _reranker_model
