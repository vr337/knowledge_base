# 环境配置与依赖导入
import os
from typing import Optional

from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI

# 项目内部依赖
from app.conf.lm_config import lm_config
from app.core.logger import logger

# 全局缓存：键为(模型名, JSON输出模式)元组，值为ChatOpenAI实例
# 作用：避免重复初始化客户端，提升性能，统一实例管理
_llm_client_cache = {}


def _resolve_llm_credentials(model: Optional[str] = None) -> tuple[str, str, str, bool]:
    """
    根据 LLM_USE_LOCAL 开关，决定本次客户端使用哪一组「密钥/地址/模型名」。

    切换效果说明：
    - LLM_USE_LOCAL=False（默认，云端模式）：走 DashScope，使用 OPENAI_API_KEY
      + OPENAI_BASE_URL + LLM_DEFAULT_MODEL，与改造前行为完全一致。
    - LLM_USE_LOCAL=True（自建模式）：走 vLLM 部署的 Qwen3-8B，改用
      LLM_LOCAL_API_KEY + LLM_LOCAL_BASE_URL + LLM_LOCAL_MODEL_NAME。

    多模态（VL）例外：自建 vLLM 当前只部署文本模型，未部署 VL 模型，
    因此 VL 请求（调用方显式传入 VL_MODEL）无论开关如何都回落云端，
    避免切到自建模式后 node_md_img 的图片总结因模型不存在而失败。

    Args:
        model (Optional[str]): 调用方显式指定的模型名，优先级最高

    Returns:
        tuple[str,str,str,bool]: (api_key, base_url, target_model, is_local)

    Raises:
        ValueError: 自建模式下缺少 LLM_LOCAL_* 三件套配置
    """
    # 1. 多模态（VL）请求优先判定：识别依据是调用方显式传入的模型名与配置的 VL_MODEL 一致
    # 放在最前面，保证 VL 回落时不需要自建模型三件套配置也能正常工作
    if model and lm_config.lv_model and model == lm_config.lv_model:
        logger.warning(
            f"[LLM客户端] 检测到多模态模型请求（{model}），自建vLLM未部署VL模型，自动回落至云端DashScope"
        )
        return lm_config.api_key, lm_config.base_url, lm_config.lv_model, False

    # 2. 云端模式（默认）：保持原有配置不变，保证存量行为零回归
    if not lm_config.use_local:
        return lm_config.api_key, lm_config.base_url, model or lm_config.llm_model or "qwen3-32b", False

    # 3. 自建模式：校验自建模型三件套，缺失时给出明确指引
    if not lm_config.local_api_key:
        raise ValueError("[LLM客户端] 配置缺失：LLM_USE_LOCAL=True 时需在.env中配置LLM_LOCAL_API_KEY（自建模型API密钥）")
    if not lm_config.local_base_url:
        raise ValueError("[LLM客户端] 配置缺失：LLM_USE_LOCAL=True 时需在.env中配置LLM_LOCAL_BASE_URL（自建模型服务地址）")

    # 4. 模型名优先级：调用方显式传入 > 自建模型配置 > 内置默认值
    target_model = model or lm_config.local_llm_model or "Qwen3-8B"
    return lm_config.local_api_key, lm_config.local_base_url, target_model, True


def get_llm_client(model: Optional[str] = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带全局缓存的LangChain ChatOpenAI客户端实例
    适配OpenAI/千问/即梦AI等**OpenAI兼容API**，支持自定义模型和JSON标准化输出
    核心特性：缓存机制+配置统一加载+异常精准捕获+国产模型参数适配+云端/自建双模式切换

    :param model: 模型名称，优先级：传入参数 > 配置文件模型名（云端取LLM_DEFAULT_MODEL，自建取LLM_LOCAL_MODEL_NAME）
                   注意：若传入的是多模态模型（VL_MODEL），无论开关如何都走云端
    :param json_mode: 是否开启JSON输出模式，开启后返回标准json_object格式（适配结构化数据解析）
    :return: 初始化完成的ChatOpenAI实例（优先从全局缓存获取，未命中则新建并缓存）
    :raise ValueError: 缺失API密钥/基础地址等核心配置
    :raise Exception: 模型初始化失败（LangChain封装层异常）
    """
    # 1. 确定目标模型与凭据（根据 LLM_USE_LOCAL 分流云端/自建两套配置）
    api_key, base_url, target_model, is_local = _resolve_llm_credentials(model)
    # 缓存键：模型名+JSON模式，唯一标识不同配置的客户端
    # 自建与云端即使模型名相同也要区分，故把 is_local 一并纳入键
    cache_key = (target_model, json_mode, is_local)

    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}，来源={'自建' if is_local else '云端'}")
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置对应的API密钥（云端为OPENAI_API_KEY，自建为LLM_LOCAL_API_KEY）")
    if not base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置对应的API基础地址（云端为OPENAI_BASE_URL，自建为LLM_LOCAL_BASE_URL）")
    logger.info(
        f"[LLM客户端] 开始初始化新实例：来源={'自建vLLM' if is_local else '云端DashScope'}，"
        f"模型={target_model}，JSON模式={json_mode}"
    )

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    # 自建 vLLM 不认 enable_thinking 这个百炼私有参数，仅在云端模式下透传
    extra_body = {} if is_local else {"enable_thinking": False}
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 5. 客户端初始化：捕获LangChain封装层异常，抛出更友好的提示
    try:
        llm_client = ChatOpenAI(
            model=target_model,  # 目标模型名
            temperature=lm_config.llm_temperature or 0.1,  # 低温度保证输出确定性（0~1）
            api_key=api_key,  # API密钥（云端/自建二选一）
            base_url=base_url,  # API基础地址（云端/自建二选一）
            extra_body=extra_body,  # 国产模型私有参数透传（自建模式不下发）
            model_kwargs=model_kwargs,  # OpenAI通用参数
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}，来源={'自建' if is_local else '云端'}")

    return llm_client


# 测试示例：验证客户端创建、缓存机制及日志输出
if __name__ == "__main__":
    logger.info("===== 开始执行LLM客户端工具测试 =====")
    try:
        # 测试1：默认配置（默认模型+普通模式）
        client1 = get_llm_client()
        logger.info("✅ 测试1通过：默认配置客户端创建成功")

        # 测试2：指定多模态模型（qwen-vl-plus）+ 普通模式
        client2 = get_llm_client(model="qwen-vl-plus")
        logger.info("✅ 测试2通过：指定多模态模型客户端创建成功")

        # 测试3：同一模型+模式，验证缓存命中
        client3 = get_llm_client(model="qwen-vl-plus")
        logger.info(f"✅ 测试3通过：缓存机制验证成功，client2与client3为同一实例：{client2 is client3}")

        # 测试4：开启JSON输出模式
        client4 = get_llm_client(model="qwen3-32b", json_mode=True)
        logger.info("✅ 测试4通过：JSON输出模式客户端创建成功")

    except Exception as e:
        logger.error(f"❌ LLM客户端工具测试失败：{str(e)}", exc_info=True)
    finally:
        logger.info("===== LLM客户端工具测试结束 =====")