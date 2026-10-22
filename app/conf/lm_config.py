# 导入核心依赖：数据类、环境变量读取、路径处理
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义minerU服务配置
@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    lv_model: str
    llm_model: str
    llm_temperature: float
    # ===== 自建模型（vLLM）切换开关与专属配置 =====
    # use_local：True=走自建 vLLM（Qwen3-8B），False=走云端 DashScope（默认）
    use_local: bool
    # 自建模型专属配置，仅在 use_local=True 时生效
    local_api_key: str
    local_base_url: str
    local_llm_model: str

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    lv_model=os.getenv("VL_MODEL"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE")),
    # 特殊处理：将.env中的True/False转为布尔值，默认False（云端优先）
    use_local=os.getenv("LLM_USE_LOCAL", "False") in ("1", "True", "true", 1),
    # 自建 vLLM 的三件套：key / base_url / 模型名
    local_api_key=os.getenv("LLM_LOCAL_API_KEY"),
    local_base_url=os.getenv("LLM_LOCAL_BASE_URL"),
    local_llm_model=os.getenv("LLM_LOCAL_MODEL_NAME"),
)
