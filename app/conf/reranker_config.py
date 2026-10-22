# 导入核心依赖：数据类、环境变量读取、路径处理
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

@dataclass
class RerankerConfig:
    bge_reranker_large: str  # 本地模型路径
    bge_reranker_device: str       # 模型仓库标识
    bge_reranker_fp16: bool    # 是否开启半精度（1=True/0=False）
    # ===== 本地/远端切换 =====
    # use_local：True=本地加载模型（默认） / False=调用远端HTTP服务
    use_local: bool
    http_url: str     # 远端Reranker服务地址（仅use_local=False时生效）

# 实例化配置对象，和原代码lm_config风格保持一致
reranker_config = RerankerConfig(
    bge_reranker_large=os.getenv("BGE_RERANKER_LARGE"),
    bge_reranker_device=os.getenv("BGE_RERANKER_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_reranker_fp16=os.getenv("BGE_RERANKER_FP16") in ("1", "True", "true", 1),
    # 默认True：保持原有本地加载行为，存量环境零改动
    use_local=os.getenv("BGE_RERANKER_USE_LOCAL", "True") in ("1", "True", "true", 1),
    http_url=os.getenv("BGE_RERANKER_HTTP_URL"),
)