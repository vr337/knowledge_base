# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（保持和原代码一致，只需执行一次）
load_dotenv()

# 定义Embedding配置（适配BGE-M3的所有配置，类名embedding_config）
@dataclass
class EmbeddingConfig:
    bge_m3_path: str  # 本地模型路径
    bge_m3: str       # 模型仓库标识
    bge_device: str   # 运行设备(cuda:0/cpu)
    bge_fp16: bool    # 是否开启半精度（1=True/0=False）
    # ===== 本地/远端切换 =====
    # use_local：True=本地加载模型（默认） / False=调用远端HTTP服务
    use_local: bool
    http_url: str     # 远端BGE-M3服务地址（仅use_local=False时生效）

# 实例化配置对象，和原代码lm_config风格保持一致
embedding_config = EmbeddingConfig(
    bge_m3_path=os.getenv("BGE_M3_PATH"),
    bge_m3=os.getenv("BGE_M3"),
    bge_device=os.getenv("BGE_DEVICE"),
    # 特殊处理：将.env中的1/0转为布尔值，兼容常见的数字/字符串格式
    bge_fp16=os.getenv("BGE_FP16") in ("1", "True", "true", 1),
    # 默认True：保持原有本地加载行为，存量环境零改动
    use_local=os.getenv("BGE_M3_USE_LOCAL", "True") in ("1", "True", "true", 1),
    http_url=os.getenv("BGE_M3_HTTP_URL"),
)