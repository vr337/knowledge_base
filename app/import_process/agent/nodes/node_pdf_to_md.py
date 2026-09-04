"""
node_pdf_to_md
  参数： state [is_pdf_read_enabled = True | pdf_path = xxx.pdf | local_dir = output ]
  返回： state [md_path = 地址 | md_content = 内容 ]
  1. 日志和任务状态
  2. step_1_validate_paths路径校验
  3. step_2_upload_and_poll minerU的交互
  4. step_3_download_and_extract 下载和解压
  5. 日志和任务状态 return state
step_1_validate_paths
  参数：state pdf_path = xxx.pdf | local_dir = output
  返回： pdf_path_obj Path  local_dir_obj Path
  1. 非空校验
  2. 文件校验 pdf_path_obj 没有抛异常 local_dir_obj 没有给与默认
  3. 返回完成可用的Path对象即可
step_2_upload_and_poll
  参数：pdf对应Path  pdf_path_obj
  返回：str zip url地址
  1. 进行申请，获取要上传文件的地址
  2. 进行文件上传 session | requests.put
  3. 轮询获取返回结果 zip_url  （确定一个最大等待时间 1页pdf 1s 间隔时间3 错误码 200 -》 500能容忍）
  4. 返回地址即可
step_3_download_and_extract
  参数：zip_url , out_dir_obj , 原文件名 path.stem
  返回：解压后的.md的str地址
  1. zip下载 get    output / stem_result.zip
  2. 检查解压的文件夹地址  output / stem
  3. 检查解压的文件夹进行防重复处理
  4. 进行解压 zipFile  extractall(解压的目标文件夹)
  5. 考虑文件名字 原文件件名 还是 full 还是其他
  6. 重命名处理
  7. 路径转成字符串 获取绝对路径最终返回即可！
"""

import shutil
import time
import zipfile
from pathlib import Path

import requests
from loguru import logger

from app.conf.mineru_config import mineru_config
from app.core.logger import node_log, step_log
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_done_task, add_running_task


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    Args:
        state (ImportGraphState): 全局状态

    Returns:
        ImportGraphState: 全局状态
    """
    # 1. 注解日志和进行状态处理
    add_running_task(state.get("task_id"), "node_pdf_to_md")
    # 2. step_1_validate_paths 校验路径完整以及是否真实存在
    pdf_path_obj,output_dir_obj=step_1_validate_paths(state)
    # 3. step_2_upload_and_poll 上传文件并轮询判断
    zip_url=step_2_upload_and_poll(pdf_path_obj)
    # 4. step_3_download_and_extract 下载并解压
    md_path=step_3_download_and_extract(zip_url,output_dir_obj,pdf_path_obj.stem)
    # 5. 处理响应结果 state进行赋值处理
    logger.info(f"成功获取MD文件路径: {md_path}")
    # 同步一下最新md_path
    state["md_path"]=md_path
    # 同步一下md_content
    state["md_content"] = Path(md_path).read_text(encoding="utf-8")
    # 6. 注解日志完成任务状态处理
    add_done_task(state.get("task_id"), "node_pdf_to_md")
    return state

@step_log("step_1_validate_paths")
def step_1_validate_paths(state: ImportGraphState) -> tuple[Path, Path]:
    """
    步骤1：路径校验与初始化
    校验PDF输入文件与输出目录的有效性，遵循「输入严格校验、输出自动修复」的鲁棒性设计原则：
    1. 校验PDF路径非空且文件真实存在，不存在则直接抛出异常（快速失败）
    2. 校验输出目录，为空则赋予默认值，不存在则自动创建（自动容错）
    3. 统一转换为Path对象处理，保证路径操作的规范性与跨平台兼容性
    Args:
        state (ImportGraphState): 全局的状态

    Returns:
        tuple[Path,Path]: (校验后的PDF路径的Path对象，输出目录的Path对象)
    """

    # 1. 获取路径参数
    pdf_path = state.get("pdf_path", "").strip()
    output_dir = state.get("local_dir", "").strip()
    # 2. 参数非空校验
    if not pdf_path:
        # 输入严格校验
        logger.error("pdf_path不能为空，请提供有效的PDF文件路径！")
        raise ValueError("pdf_path不能为空，请提供有效的PDF文件路径！")
    if not output_dir:
        # 输出自动补充，没有的就默认值
        output_dir = PROJECT_ROOT / "output"
        # 同步到state里
        state["local_dir"] = str(output_dir)
        logger.warning(f"未指定输出目录，已自动设置为默认目录: {output_dir}")
    # 3. 统一转换为Path对象，标准化路径处理
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(output_dir)
    # 4. 路径有效性校验（差异化处理：输入严格校验，输出自动修复）
    if not pdf_path_obj.exists():
        logger.error(f"PDF文件路径不存在：{pdf_path_obj}，请检查文件路径是否正确")
        raise FileNotFoundError(f"PDF文件路径不存在：{pdf_path}")
    if not output_dir_obj.exists():
        logger.warning(f"输出目录不存在，已自动创建：{output_dir_obj}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)
    return pdf_path_obj, output_dir_obj

@step_log("step_2_upload_and_poll")
def step_2_upload_and_poll(pdf_path_obj: Path) -> str:
    """
    步骤2：上传PDF至MinerU并轮询解析任务状态
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    参数：pdf_path_obj-已校验的PDF Path对象；output_dir_obj-输出目录Path对象
    返回：解析结果ZIP包下载链接full_zip_url
    异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    Args:
        pdf_path_obj (Path): 上传pdf文件路径的Path对象
    Returns:
        str: 解析结果ZIP包下载链接
    """
    # 1. 前期配置校验，拦截无效配置
    if not mineru_config.base_url or not mineru_config.api_key:
        logger.error("MinerU的(base_url)或(api_key)配置缺失，请检查配置文件")
        raise ValueError("MinerU的(base_url)或(api_key)配置缺失，请检查配置文件")
    # 2. 构造请求头，调用批量接口获取预签名上传地址与批次ID
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {mineru_config.api_key}",
    }
    url_get_upload = f"{mineru_config.base_url}/file-urls/batch"
    request_data = {
        "files": [{"name": "demo.pdf", "data_id": "abcd"}],
        "model_version": "vlm",
    }
    # 发送获取上传连接请求
    response = requests.post(
        url_get_upload, headers=request_headers, json=request_data
    )
    # 判断请求是否成功
    if response.status_code != 200:
        logger.error(
            f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}"
        )
        raise RuntimeError(
            f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}"
        )
    # 请求成功，判断接口是否调用成功，0为成功
    # 将响应的json转换成字典
    res_data = response.json()
    if res_data.get("code") != 0:
        logger.error(
            f"调用接口返回失败，状态码：{res_data.get('code')}，响应内容：{res_data.get('msg')}"
        )
        raise RuntimeError(
            f"请求MinerU服务失败，状态码：{response.status_code}，响应内容：{response.text}"
        )
    # 获取文件上传url和bathId(获取文件的凭证)
    upload_file_url = res_data.get("data").get("file_urls")[0]
    batch_id = res_data.get("data").get("batch_id")
    # 3. 读取PDF文件二进制内容
    file_data = pdf_path_obj.read_bytes()
    # 4. 使用稳定Session上传文件（关闭系统环境变量，避免OSS预签名URL校验失败）
    with requests.Session() as session:
        # 不信任系统环境变量，这样就不会发送请求时把系统环境变量带上，特别是开代理的时候
        session.trust_env = False
        res_upload = session.put(upload_file_url, data=file_data)
        # 判断上传的网络请求是否成功
        if res_upload.status_code != 200:
            logger.error(
                f"上传文件失败，状态码：{res_upload.status_code}，响应内容：{res_upload.text}"
            )
            raise RuntimeError(
                f"上传文件失败，状态码：{res_upload.status_code}，响应内容：{res_upload.text}"
            )
    # 5. 轮询解析任务状态（带超时控制 + 服务端异常自动重试）
    poll_url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600  # 最大超时时间
    poll_interval = 3  # 轮询间隔
    # 开始计时
    start_time = time.time()
    logger.debug("开始轮询MinerU解析结果...")
    while True:
        # 超时判断
        if time.time() - start_time > timeout_seconds:
            logger.error(
                f"MinerU解析任务超时（超时时间{timeout_seconds}秒），请检查服务或文件大小"
            )
            raise TimeoutError(
                f"MinerU解析任务超时（超时时间{timeout_seconds}秒），请检查服务或文件大小"
            )
        try:
            poll_response = requests.get(poll_url, headers=request_headers, timeout=10)
        except Exception as e:
            # 出现异常，可能是网络原因，允许重试
            logger.warning(f"轮询请求异常，将重试：{str(e)}")
            time.sleep(poll_interval)
            continue
        # 服务端5xx异常 → 降级重试
        status_code = poll_response.status_code
        if status_code != 200:
            if status_code >= 500 and status_code < 600:
                logger.warning(
                    f"服务端异常，将{poll_interval}s重试：状态码：{status_code}"
                )
                time.sleep(poll_interval)
                continue
            else:
                # 这里一般是客户端问题，无论如何重试都没用
                logger.error(
                    f"轮询请求失败，客户端异常{status_code}，请检查API_KEY与服务地址"
                )
                raise RuntimeError(
                    f"轮询请求失败，客户端异常{status_code}，请检查API_KEY与服务地址"
                )
        # 将响应结果转化成字典
        poll_res_data = poll_response.json()
        # 判断接口状态码是否为0
        if poll_res_data.get("code") != 0:
            logger.warning(
                f"MinerU轮询接口异常，code：{poll_res_data.get('code')}，msg：{poll_res_data.get('msg')}"
            )
            raise RuntimeError(
                f"MinerU轮询接口异常，code：{poll_res_data.get('code')}，msg：{poll_res_data.get('msg')}"
            )
        # 接口调用成功，判断pdf文件是否完成
        extract_result = poll_res_data.get("data").get("extract_result")
        if not extract_result:
            # 连返回的内容都没有，重试
            logger.debug("暂无解析结果，继续轮询...")
        # 根据任务状态，不同的处理
        state = extract_result[0].get("state")
        full_zip_url = None
        if state == "done":
            full_zip_url = extract_result[0].get("full_zip_url")
            if not full_zip_url:
                # 可能因为某种原因，为空
                logger.error("MinerU解析完成，但为解析结果")
                raise RuntimeError("MinerU解析完成，但为解析结果")
            return full_zip_url
        elif state == "failed":
            logger.error("MinerU解析任务执行失败，请检查PDF文件是否损坏或内容异常")
            raise RuntimeError(
                "MinerU解析任务执行失败，请检查PDF文件是否损坏或内容异常"
            )
        else:
            # 其他状态，如waiting-file: 等待文件上传排队提交解析任务中，pending: 排队中，running: 正在解析，converting：格式转换中
            logger.debug(f"任务处理中，当前状态：{state}，继续轮询...")
            time.sleep(poll_interval)
    # 6. 根据任务状态执行对应逻辑

@step_log("step_3_download_and_extract")
def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, stem: str) -> str:
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
    参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)

    Args:
        zip_url (str): 待下载zip文件的下载链接
        output_dir_obj (Path): 输出的文件目录地址的Path对象
        stem (str): PDF文件的原始名称（无后缀）

    Returns:
        str: 解压后，重命名后最终的md文件路径
    """
    # 1.下载zip_url对应的资源
    response = requests.get(zip_url,timeout=120)
    # 判断请求是否成功
    if response.status_code!=200:
        logger.error(f"ZIP包下载失败，状态码：{response.status_code}，响应内容：{response.text}")
        raise RuntimeError(f"ZIP包下载失败，状态码：{response.status_code}，响应内容：{response.text}")
    # 2.保存zip文件
    # 构建保存的zip文件路径
    # 文件保存和命名 output_dir_obj / 源文件名_result.zip
    zip_save_path_obj=output_dir_obj / f"{stem}_result.zip"
    # 保存,这里不需要删除旧zip文件，因为存在重名zip文件会被覆盖，即使内容不一样
    zip_save_path_obj.write_bytes(response.content)
    # 3.清理旧目录并解压zip包
    # 这里需要清理，如果同名但不同内容的zip文件解压的话，会在同名目录解压，但内容不同，
    # 比如图片，那么生成的副产物图片就不会覆盖旧图片，会保留以前的数据，覆盖只会覆盖同名的，所以要清理
    # 构建要解压的目录
    # 解压目录：output_dir_obj/源文件名/
    extract_dir_obj=output_dir_obj/stem
    # 如果目录存在的话，删除所有
    if extract_dir_obj.exists():
        # 递归删除，从内往外删，参考linux： rm -rf 非空目录
        shutil.rmtree(extract_dir_obj)
    # 不存在的话创建
    # parents：没有父目录创建，exist_ok：如果存在，不报错
    extract_dir_obj.mkdir(parents=True,exist_ok=True)
    # 利用zipfile解压文件
    with zipfile.ZipFile(zip_save_path_obj,"r") as zip_ref:
        zip_ref.extractall(extract_dir_obj)
    # 4. 处理下md文件,统一姓名,并且返回md的字符串地址
    # 获取所有md文件，判断是否转换成功
    md_file_list=list(extract_dir_obj.rglob("*.md"))

    if not md_file_list:
        logger.error("未找到PDF对应的MD文件")
        raise FileNotFoundError("未找到PDF对应的MD文件")

    # 有md文件，按照与原文件名同名文件->full.md->列表第一个md文件优先级找
    target_md_file=None
    # 优先找与原文件名同名的md文件
    for md_file in md_file_list:
        # 这里stem是不含扩展名
        if md_file.stem == stem:
            target_md_file=md_file
            break
    # 没有就找full.md文件，不区分大小写
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower()=="full.md":
                target_md_file=md_file
                break

    # 兜底，用列表的第一个md文件
    if not target_md_file:
        target_md_file = md_file_list[0]

    # 重命名成原文件名，原文件abc.pdf,转换后是full.md,那么重名名为abc.md
    # 只有不与原文件名同名，才重名
    if target_md_file.stem!=stem:
         # 进行重命名
        # target_md_file.with_name(f"{stem}.md") 修改path对象 （不涉及文件操作） 返回结果是修改后path对象
        # target_md_file.rename(target_md_file.with_name(f"{stem}.md")) 修改磁盘中的文件名称（修改名称了） return 新的路径path
        # 先修改内存的名字
        new_name=target_md_file.with_name(f"{stem}.md")
        # 再更新磁盘里的名字
        target_md_file=target_md_file.rename(new_name)
    # 返回最终的md文件的绝对路径
    return str(target_md_file.absolute())