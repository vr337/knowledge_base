# 导入核心依赖：数据类、环境变量读取、路径处理
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_key : str
    use_local: bool # 是否使用本地mineru命令行工具（True=本地CLI，False=在线API）

mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL"),
    api_key=os.getenv("MINERU_API_TOKEN"),
    # 特殊处理：将.env中的True/False转为布尔值，默认True（本地模式优先）
    use_local=os.getenv("MINERU_USE_LOCAL", "True") in ("1", "True", "true", 1)
)