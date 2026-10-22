"""
主要目标: 就是将md中的图片进行单独处理,图片转成对应的语义文本,方便后续进行切片搜索!
主要动作: 图片 -> 图片服务器 -> (上文)图片内容(下文) -> 传递到视觉模型 -> 生成图片总结
         -> 替换md原有的图片显示 ![图片总结](图片的minio网络地址) -> state 修改md_content / md_path 新内容 -> 结束
技术总结: minio reg正则 多模态模型 提示词
实现步骤:
      1. 进行任务和日志处理
      2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
      3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
      4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
      5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
      6. 备份新的md内容,改为原名称 _new.md
      7. 进行md_path和md_content内容更新(state)
      8. 返回目标结果即可
"""

import base64
import re
from collections import deque
from pathlib import Path

from langchain.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task, add_running_task

# MiniIO支持的图片格式集合（小写后缀，同一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    Args:
        filename (str): 文件名称（包含扩展名）

    Returns:
        bool: 图片是否支持
    """
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
        为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
        未来要实现:
        1. 扫描 Markdown 中的图片链接。
        2. 将图片上传到 MinIO 对象存储。
        3. (可选) 调用多模态模型生成图片描述。
        4. 替换 Markdown 中的图片链接为 MinIO URL。
    Args:
        state (ImportGraphState): 全局导入的状态

    Returns:
        ImportGraphState: 全局导入的状态
    """
    # 1. 注解日志和进行状态处理
    add_running_task(state.get("task_id"), "node_md_img")
    # 2. 进行核心参数校验 [校验md_path/md_content/返回images的文件夹地址]
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # 3. 查找md中使用的图片和上下文 [传入md_content和images文件夹,返回进行模型访问准备 [(图片名,图片地址,(上文,下文))]]
    image_targets = step_2_scan_images(md_content, images_dir_obj)
    # 4. 进行图片内容总结和处理[调用多模态模型,总结图片内容,最终返回 图片名/总结]
    image_summaries = step_3_image_summary(image_targets, md_path_obj.stem)
    # 5. 上传图片到minio服务器,替换图片的本地地址和描述!返回替换后的md_content内容
    new_md_content = step_4_upload_images_replace(
        image_summaries, image_targets, md_content, md_path_obj.stem
    )
    # 6. 备份新的md内容,改为原名称 _new.md
    new_md_file_path_str = step_5_backup_md_file(md_path_obj, new_md_content)
    # 7. 进行md_path和md_content内容更新(state)
    state["md_path"]=new_md_file_path_str
    state["md_content"]=new_md_content
    # 8. 注解日志和完成状态处理
    add_done_task(state.get("task_id"), "node_md_img")
    # 9. 返回目标结果即可
    return state


@step_log("step_1_get_content")
def step_1_get_content(state: ImportGraphState) -> tuple[str, Path, Path]:
    """
    提取和校验内容,并且返回图片的地址
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        tuple[str,Path,Path]: (md文件的内容,md路径的path对象,md文件关联的图片目录的Path对象)
    """
    # 校验md_path
    md_file_path = state.get("md_path")
    if not md_file_path:
        logger.error("md_path不能为空，请检查参数")
        raise ValueError("md_path不能为空，请检查参数")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        logger.error(f"md_path参数错误,请检查输入参数! {md_file_path}")
    # 判断md_content是否为空
    if not state.get("md_content"):
        # 为空的话，表示上传的是.md文件，工作流顺序为node_entry-->node_md_img，补充一下md_content
        state["md_content"] = md_path_obj.read_text(encoding="utf-8")
    # 不为空的话，表示上传的是.pdf文件，工作流顺序为node_entry-->node_pdf_to_md-->node_md_img
    # 获取md文件关联的图片存储位置，转换成Path对象
    images_dir_obj = md_path_obj.parent / "images"
    # 返回
    return state.get("md_content"), md_path_obj, images_dir_obj


@step_log("step_2_scan_images")
def step_2_scan_images(
    md_content: str, images_dir_obj: Path
) -> list[tuple[str, str, tuple[str, str]]]:
    """
    扫描 MD 文档中的图片，匹配本地图片文件，并截取图片上下文（前后100字符）
    Args:
        md_content (str): Markdown 原文内容
        images_dir_obj (Path): 图片所在目录（Path 对象）

    Returns:
        list[tuple[str,str,tuple[str,str]]]: 图片信息[(图片文件名, 图片完整路径, (上文, 下文))]
    """
    # 存储最终处理好的图片信息
    image_targets = []

    for image_file in images_dir_obj.iterdir():
        # 获取图片名称
        image_name = image_file.name
        # 跳过非图片文件
        if not is_supported_image(image_name):
            logger.warning(f"跳过非图片文件：{image_name}")
            continue
        # 获取图片完整路径
        image_complete_path = str(image_file)
        # 正则匹配 MD 中的图片语法：![...](...图片名...)
        # re.escape 处理文件名中带特殊字符（如括号、点）导致正则爆炸的问题
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_name) + r".*?\)")
        items = list(pattern.finditer(md_content))
        # 没有任何匹配的，跳过
        if not items:
            logger.warning(f"图片{image_name}未在MD中引用，跳过")
            continue
        # 因为匹配的是同一张图片，所有只需要其中一张图片的上下文作为生成描述的内容即可
        # 获取上下文，上文：当前匹配项的上100字符，下文：当前匹配项的下100个字符
        # 拿到匹配的第一个![...](...指定图片名...)的开始索引和结束索引
        start, end = items[0].span()
        # 可能-100出现负数，所以小于0则用0代替
        pre_context = md_content[max(0, start - 100) : start]
        # +100可能超过len(md_content)越界，超过了按最大长度算
        post_context = md_content[end: min(end + 100, len(md_content))]
        

        # 构建返回结果
        image_targets.append(
            (image_name, image_complete_path, (pre_context, post_context))
        )
    # 返回
    return image_targets


@step_log("step_3_image_summary")
def step_3_image_summary(
    image_targets: list[tuple[str, str, tuple[str, str]]], stem: str
) -> dict[str,str]:
    """
    总结图片,生成图片名.png - 对应的图片描述内容
    Args:
        image_targets (list[tuple[str,str,tuple[str,str]]]): 图片信息 [(文件名,文件地址,(上文,下文))]
        stem (str): 文件标题(去掉扩展名的文件名)->提示词需要

        Returns:
            dict[str,str]: 图片总结{图片文件名:图片摘要(视觉模型生的描述)}
    """

    # 1. 定义总结字典
    summaries = {}
    # 2. 定义任务队列 -> 模型访问队列 -> 限制访问次数
    # - 在 node_md_img.py 里 request_times = deque() 定义在函数内部。
    # - 所以每次调用 step_3_generate_img_summaries() 都会创建一个 新的队列 。
    # - 它只能限制“这一次函数调用里的图片循环速率”， 不是全局限流 。
    # 也就是说：
    # - 单次任务内：有效（同一批图片会被限速）
    # - 多次请求/多任务并发：彼此不共享队列，不会互相限速
    # 如果你要“全局限制”，要改成共享状态，比如：
    # - 模块级全局 deque （仅单进程有效）
    # - Redis 限流（多进程/多实例推荐，企业常用）
    # - 网关层限流（如 Nginx/API Gateway）

    # 获取一个双端队列
    # 双端队列指的是两边都可以进和出的队列
    requests_limiter = deque()

    # 遍历图片信息，拿到每个图片对应的文件名,文件地址,(上文,下文)
    for image_name, image_path, context in image_targets:
        # 访问限速问题（我们模型的限速标准 1分钟 可以访问100  限制并发访问次数..）
        # 具体要根据模型的配置 https://help.aliyun.com/zh/model-studio/rate-limit?spm=a2c4g.11186623.help-menu-2400256.d_0_0_3.29c5d355nLkkXf&scm=20140722.H_2840182._.OR_help-T_cn~zh-V_1

        apply_api_rate_limit(requests_limiter, max_requests=100, window_seconds=60)
        # 获取多模态模型对象,这里使用视觉视觉模型
        vm_model = get_llm_client(model=lm_config.lv_model)
        # 准备提示词
        prompt = load_prompt(name="image_summary", root_folder=stem, image_content=context)
        # 将图片转成base64字符串
        # path.read_text()	读取文本（txt/md）	str 字符串
        # path.write_text()	写入文本	str 字符串
        # path.read_bytes()	读取二进制（图片 / 视频）	bytes 字节
        # path.write_bytes()	写入二进制（保存文件）	bytes 字节
        # 准备模型需要的参数，应为是通过api调用模型，无法传模型，只能传字符串，所以需要转换成base64(这里取决于那个视觉模型，大部分都是base64)
        # 图片->字节->b64字节->b64字符串
        if isinstance(image_path, str):
            image_path = Path(image_path)
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        )

        # 调用模型并获取结果
        chain = vm_model | StrOutputParser()
        summary = chain.invoke([message])

        # 构建返回结果，key：图片名称，value：对应的摘要
        summaries[image_name] = summary
    return summaries


@step_log("step_4_upload_images")
def step_4_upload_images_replace(
    image_summaries: dict[str, str],
    image_targets: list[tuple[str, str, tuple[str, str]]],
    md_content: str,
    stem: str,
) -> str:
    """
    将图片上传到minio服务器!
    同时替代原md内容中的图片地址和描述内容!
    确保任何位置可以进行访问图片和现实
    Args:
        image_summaries (dict[str,str]): 每个图片对应的摘要 {图片名称：摘要,....}
        image_targets (list[tuple[str, str, tuple[str, str]]]): 图片信息 [(文件名,文件地址,(上文,下文))]
        md_content (str): md文件内容
        stem (str): md文件标题名(去掉扩展名的文件名)

    Returns:
        str: 替换后的md文件的内容
    """

    # 1. 获取minio客户端对象
    minio_client = get_minio_client()
    # 2. 先清空原有md在minio中所在的图片，防止上传同名文件，但图片不一样，从而出现minio服务器下同名目录下出现旧图片占空间
    # 先拿到要清空的minio中文件的对象列表
    object_list = minio_client.list_objects(
        bucket_name=minio_config.bucket_name,
        # 从桶名后的目录名开始，如upload-images/上传的文件名(stem)，这里是目录
        prefix=f"{minio_config.minio_img_dir}/{stem}",
        # 允许递归，可以拿到目录里的目录里的文件，会去递归找
        recursive=True,
    )
    # 拿到要删除的对象列表
    delete_object_list = [DeleteObject(object.object_name) for object in object_list]
    error_list = minio_client.remove_objects(
        bucket_name=minio_config.bucket_name,
        delete_object_list=delete_object_list,
    )
    for error in error_list:
        logger.warning(f"删除图片失败：{error}")
    # 3.上传图片到minio服务器
    # 定义一个字典，存放上传minio的图片地址，{"图片名称":"完整的访问minio里存储的图片地址"}，方便后面替换md_content使用
    image_urls = {}
    # 开始上传
    for image_name, image_path, _ in image_targets:
        try:
            # 这里是文件上传，可能会失败，捕获一下
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                # 这里是上传的对象名，从目录开始，不包含桶，如upload_images/abc/abc.jpg
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_name}",
                file_path=image_path,
                content_type="image/jpeg",
            )
            # 上传成功后，构建image_url
            image_urls[image_name] = (
                f"http://{minio_config.endpoint}/{minio_config.bucket_name}/{minio_config.minio_img_dir}/{stem}/{image_name}"
            )
            logger.debug(f"完成图片:{image_name}上传,URL:{image_urls[image_name]}")
        except Exception as e:
            logger.exception(f"上传图片失败：{image_name}，失败原因：{e}")
            logger.debug("继续尝试上传下一张图片!")
    # 4.拼接替换的完整资料 {image_file:(summary,url)}
    images_info = {
        image_name: (summary, image_urls.get(image_name))
        for image_name, summary in image_summaries.items()
    }
    # 5.进行md_content内容替换
    # 有替换的内容才替换,没有的话，就表示纯文本
    if images_info:
        for image_name, (summary, url) in images_info.items():
            # 构建正则，匹配![...](...image_name...) 替换成对应的![summary](url)
            pattern = re.compile(r"!\[.*?\]\(.*?"+ re.escape(image_name) +r".*?\)")
            # 将匹配的内容动态替换
            md_content = pattern.sub(lambda _: f"![{summary}]({url})", md_content)
    logger.debug(f"完成新旧md内容替换,最新内容:{md_content[:200]}")
    return md_content


@step_log("step_5_backup_md_file")
def step_5_backup_md_file(md_path_obj: Path, new_md_content: str) -> str:
    """
    完成新的md的磁盘备份,并且返回新的地址!
    新的命名规则: 原名称_new.md
    Args:
        md_path_obj (Path): md文件路径的Path对象
        new_md_content (str): 替换后新的md_content

    Returns:
        str: 替换后的md文件的路径
    """

    new_md_path_obj = md_path_obj.parent / f"{md_path_obj.stem}_new{md_path_obj.suffix}"
    new_md_path_obj.write_text(new_md_content, encoding="utf-8")

    return str(new_md_path_obj)
