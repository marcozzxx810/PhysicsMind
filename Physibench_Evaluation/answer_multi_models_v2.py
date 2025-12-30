#!/usr/bin/env python3
"""
多模型答案生成脚本 V2.0
读取已有的问题和图片，使用多个VLM模型同时回答，将所有答案合并到一个CSV文件中。

新功能：
- 并行行处理（可配置并发数）
- 多API key轮换（防止单个key被限流）
- 增量保存和断点续传
- 信号处理（Ctrl+C优雅退出）
- 线程安全的文件操作
"""

import os
import sys
import argparse
import csv
import base64
import time
import signal
import random
import fcntl
import threading
from pathlib import Path
from typing import Tuple, List, Dict
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image


# 线程安全的打印锁和文件锁
print_lock = threading.Lock()
file_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)


def get_random_api_key(provider: str) -> str:
    """
    获取随机的API key（如果配置了多个，用分号分隔）
    
    Args:
        provider: 'openrouter', 'chatanywhere', 或 'siliconflow'
    
    Returns:
        API key
    """
    if provider == 'openrouter':
        keys_str = os.getenv("OPENROUTER_API_KEY", "")
        if not keys_str:
            raise ValueError("OPENROUTER_API_KEY 环境变量未设置")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    elif provider == 'siliconflow':
        keys_str = os.getenv("SILICONFLOW_API_KEY", "")
        if not keys_str:
            raise ValueError("SILICONFLOW_API_KEY 环境变量未设置")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    else:  # chatanywhere
        keys_str = os.getenv("CHATANYWHERE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not keys_str:
            raise ValueError("CHATANYWHERE_API_KEY 或 OPENAI_API_KEY 环境变量未设置")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)


def encode_image_to_base64(image_path: Path, max_size_mb: float = 5.0) -> str:
    """简单压缩"""
    data = image_path.read_bytes()
    orig_size_mb = len(data) / (1024 * 1024)
    
    if orig_size_mb <= max_size_mb:
        return base64.b64encode(data).decode('utf-8')
    
    # 需要压缩
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 逐步降低质量直到符合要求
    for q in range(95, 45, -5):
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=q, optimize=True)
        comp_size_mb = buffer.tell() / (1024 * 1024)
        if comp_size_mb <= max_size_mb:
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    # 极端情况：缩小尺寸
    new_width = int(img.width * 0.7)
    new_height = int(img.height * 0.7)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_llm_config(model_url: str, model_name: str) -> Dict:
    """
    根据模型名称或URL获取对应的API配置（支持多key轮换）。
    
    Args:
        model_url (str): API类型 ('openrouter' 或 'chatanywhere')
        model_name (str): 模型名称，用于特殊模型判断
    
    Returns:
        Dict: 包含api_key, base_url的配置字典
    """
    # 检查特殊模型（DeepSeek VLM 使用 SiliconFlow）
    if 'deepseek' in model_name.lower() and 'vl' in model_name.lower():
        return {
            'api_key': get_random_api_key('siliconflow'),
            'base_url': 'https://api.siliconflow.cn/v1/',
        }
    
    # 使用OpenRouter API
    if model_url == 'openrouter':
        return {
            'api_key': get_random_api_key('openrouter'),
            'base_url': 'https://openrouter.ai/api/v1',
        }
    # 使用ChatAnywhere API
    else:
        return {
            'api_key': get_random_api_key('chatanywhere'),
            'base_url': 'https://api.chatanywhere.org/v1',
        }


def get_max_tokens(model_name: str) -> int:
    """
    根据模型名称获取对应的max_tokens。
    
    Args:
        model_name (str): 模型名称
    
    Returns:
        int: max_tokens值
    """
    if 'deepseek' in model_name.lower():
        return 20000
    else:
        return 10000


def create_llm(model_url: str, model_name: str) -> ChatOpenAI:
    """
    创建LLM实例。
    
    Args:
        model_url (str): API类型 ('openrouter' 或 'chatanywhere')
        model_name (str): 模型名称
    
    Returns:
        ChatOpenAI: LLM实例
    """
    config = get_llm_config(model_url, model_name)
    max_tokens = get_max_tokens(model_name)
    
    llm = ChatOpenAI(
        api_key=config['api_key'],
        base_url=config['base_url'],
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.3,
        timeout=600,
        default_headers={
            "HTTP-Referer": "https://github.com/ttz-cn/PhysbenchScript-boyi-guanyu-",
            "X-Title": "PhysBench Question Answering",
        }
    )
    
    return llm


def answer_question(
    llm: ChatOpenAI,
    question: str,
    image_path: Path,
    model_name: str = ""
) -> str:
    """
    使用多模态LLM回答已有的问题。
    
    Args:
        llm (ChatOpenAI): LLM实例
        question (str): 要回答的问题
        image_path (Path): 关键帧图片路径
        model_name (str): 模型名称，用于判断是否需要压缩图片
    
    Returns:
        str: 生成的答案
    """
    start_time = time.time()  # ← 添加这一行
    try:
        # 判断是否需要压缩图片（只有 Claude 模型需要）
        need_compress = 'claude' in model_name.lower() or 'anthropic' in model_name.lower() or 'deepseek' in model_name.lower()
        
        # 编码图片为base64
        if need_compress:
            base64_image = encode_image_to_base64(image_path, max_size_mb=4.5)
        else:
            # 其他模型不压缩，直接读取
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建提示词 - 直接回答问题
        prompt = f"""Please answer the following multiple choice question based on the image.

Question: {question}

Instructions:
- Look carefully at the image
- Choose the correct answer from the options provided
- Only return the letter of your answer (A, B, C, or D)
- Format: Just output "Answer: X" where X is A, B, C, or D

Your answer:"""
        
        # 创建包含图片的消息
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                },
            ],
        )
        
        # 生成回复
        response = llm.invoke([message])
        elapsed_time = time.time() - start_time
        
        # 调试信息（可选：正式环境可以注释掉）
        # safe_print(f"  DEBUG: response object: {response}")  # ← 改用 safe_print
        
        # 尝试获取 token 使用信息
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            safe_print(f"  📊 Token使用: {metadata.get('token_usage', 'N/A')}")  # ← 改用 safe_print
            safe_print(f"  📝 模型: {metadata.get('model_name', 'N/A')}")  # ← 改用 safe_print
        
        safe_print(f"  ⏱️  耗时: {elapsed_time:.2f} 秒")  # ← 改用 safe_print
        response_text = response.content.strip()
        
        # 提取答案字母
        answer = response_text.upper()
        
        # 清理答案
        if "Answer:" in answer or "ANSWER:" in answer:
            answer = answer.split(":")[-1].strip()
        
        # safe_print(f"  DEBUG: cleaned answer: {answer}")  # ← 可选的调试信息
        
        # 提取单个字母
        extracted_answer = None
        for char in answer:
            if char in ['A', 'B', 'C', 'D']:
                extracted_answer = char
                break
        
        if extracted_answer:
            return extracted_answer
        else:
            return "[PARSE_ERROR]"
            
    except Exception as e:
        safe_print(f"  错误: 回答问题时出错: {e}")
        return "[ERROR]"


def try_answer_with_retry(
    llm: ChatOpenAI,
    question: str,
    full_image_path: Path,
    max_retries: int,
    retry_delay: int,
    model_name: str = ""
) -> Tuple[bool, str]:
    """
    尝试回答问题，失败时自动重试。
    
    Args:
        llm: LLM实例
        question: 问题文本
        full_image_path: 图片路径
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        model_name: 模型名称，用于判断是否需要压缩图片
    
    Returns:
        (成功标志, 答案)
    """
    for attempt in range(max_retries):
        try:
            answer = answer_question(llm, question, full_image_path, model_name)
            
            if answer not in ["[ERROR]", "[PARSE_ERROR]"]:
                if attempt > 0:
                    safe_print(f"  ✓ 重试成功（第{attempt + 1}次尝试）")
                return True, answer
            else:
                raise Exception("生成返回错误标记")
                
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"  ⚠ 尝试 {attempt + 1} 失败: {str(e)[:80]}")
                safe_print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  ✗ 所有尝试都失败了")
    
    return False, "[ERROR]"


def save_row_incremental(row: dict, csv_path: Path, fieldnames: List[str]):
    """
    线程安全的增量保存单行数据
    
    Args:
        row: 要保存的行数据
        csv_path: CSV文件路径
        fieldnames: CSV字段名列表
    """
    with file_lock:
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            # 使用文件锁防止多线程写入冲突
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())  # 强制写入磁盘
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_existing_progress(csv_path: Path) -> Dict[str, dict]:
    """
    加载已完成的进度
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        字典：key_frame_path -> 行数据
    """
    if not csv_path.exists():
        return {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return {row['key_frame_path']: row for row in reader}
    except Exception as e:
        safe_print(f"警告: 无法加载进度文件 {csv_path}: {e}")
        return {}


def process_single_row(
    row: dict,
    llms: Dict[str, ChatOpenAI],
    model_names: List[str],
    dataset_root: Path,
    row_idx: int,
    total_rows: int,
    max_retries: int,
    retry_delay: int
) -> dict:
    """
    处理单行数据，使用所有模型回答问题。
    
    Args:
        row: CSV行数据
        llms: 模型名称到LLM实例的映射
        model_names: 模型名称列表
        dataset_root: 数据集根目录
        row_idx: 行索引
        total_rows: 总行数
        max_retries: 最大重试次数
        retry_delay: 重试延迟
    
    Returns:
        更新后的行数据
    """
    keyframe_path = row['key_frame_path']
    question = row['question']
    
    safe_print(f"\n[{row_idx}/{total_rows}] 处理: {keyframe_path}")
    
    # 检查是否有有效问题
    if not question or question.startswith('[') or len(question) < 10:
        safe_print(f"  跳过: 没有有效的问题")
        return row
    
    # 构建完整的图片路径
    full_image_path = dataset_root / keyframe_path
    
    if not full_image_path.exists():
        safe_print(f"  错误: 图片不存在: {full_image_path}")
        return row
    
    safe_print(f"  问题: {question[:80]}...")
    
    # 使用每个模型回答问题
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        safe_print(f"\n  模型: {model_name}")
        
        # 回答问题（带重试机制）
        success, answer = try_answer_with_retry(
            llms[model_name], 
            question, 
            full_image_path, 
            max_retries, 
            retry_delay,
            model_name
        )
        
        if success:
            row[f'answer_{clean_model_name}'] = answer
            safe_print(f"  ✓ 答案: {answer}")
        else:
            row[f'answer_{clean_model_name}'] = "[ERROR]"
            safe_print(f"  跳过此模型的答案")
    
    return row


def test_model_connectivity(llms: Dict[str, ChatOpenAI]) -> bool:
    """
    测试所有模型的连通性。
    
    Args:
        llms (Dict[str, ChatOpenAI]): 模型名称到LLM实例的映射
    
    Returns:
        bool: 如果所有模型都连接成功返回True，否则返回False
    """
    safe_print("开始测试模型连通性...")

    all_success = True
    results = []
    
    for model_name, llm in llms.items():
        safe_print(f"\n测试模型: {model_name}")
        safe_print(f"  正在连接...")
        
        try:
            # 发送一个简单的测试消息
            test_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Hello, this is a connectivity test. Please respond with 'OK'.",
                    }
                ]
            )
            
            # 调用模型
            response = llm.invoke([test_message])
            print(f"DEBUG: response object: {response}")
            response_text = response.content.strip()
            
            safe_print(f"  ✓ 连接成功")
            safe_print(f"  响应: {response_text[:50]}...")
            
            results.append({
                'model': model_name,
                'success': True,
            })
            
        except Exception as e:
            safe_print(f"  错误: {str(e)[:100]}...")
            all_success = False
            
            results.append({
                'model': model_name,
                'success': False,
            })
    
    # 显示汇总统计
    safe_print("\n" + "="*60)
    safe_print("连通性测试汇总:")
    
    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
    
    if all_success:
        safe_print("✓ 所有模型连通性测试通过！")
    else:
        safe_print("✗ 部分模型连通性测试失败，请检查配置")
    safe_print("="*60 + "\n")
    
    return all_success


def graceful_shutdown(signum, frame):
    """
    优雅退出处理器（Ctrl+C时触发）
    """
    safe_print("\n" + "="*60)
    safe_print("⚠️  收到中断信号 (Ctrl+C)")
    safe_print("="*60)
    safe_print("✓ 所有已完成的数据已实时保存到文件")
    safe_print("✓ 下次运行将从中断处继续（断点续传）")
    safe_print("="*60)
    safe_print("安全退出中...")
    sys.exit(0)


def process_dataset_multi_models(
    input_csv_path: Path,
    output_csv_path: Path,
    dataset_root: Path,
    model_names: List[str],
    model_url: str = 'chatanywhere',
    max_retries: int = 3,
    retry_delay: int = 1,
    parallel_rows: int = 5
):
    """
    处理CSV数据集，为每个已有问题使用多个模型生成答案（支持并行、断点续传、增量保存）。
    
    Args:
        input_csv_path (Path): 输入CSV路径（包含问题和ground truth）
        output_csv_path (Path): 输出CSV路径
        dataset_root (Path): 数据集根目录
        model_names (List[str]): 使用的模型名称列表
        model_url (str): API类型 ('openrouter' 或 'chatanywhere')
        max_retries (int): 最大重试次数
        retry_delay (int): 重试延迟（秒）
        parallel_rows (int): 并行处理的行数
    """
    # 初始化所有模型的LLM实例
    safe_print(f"初始化 {len(model_names)} 个模型...")
    llms = {}
    for model_name in model_names:
        safe_print(f"  - {model_name}")
        try:
            llms[model_name] = create_llm(model_url, model_name)
        except ValueError as e:
            safe_print(f"  错误: {e}")
            sys.exit(1)
    
    safe_print(f"\n所有模型初始化完成！\n")
    
    # 测试模型连通性
    if not test_model_connectivity(llms):
        safe_print("\n错误: 模型连通性测试失败，终止执行")
        sys.exit(1)
    
    # 读取输入CSV
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    safe_print(f"从 {input_csv_path} 读取了 {len(rows)} 个条目")
    
    # 加载已完成的进度（断点续传）
    existing_progress = load_existing_progress(output_csv_path)
    safe_print(f"已完成 {len(existing_progress)} 个条目（断点续传）")
    
    # 为每个模型初始化答案列
    fieldnames = list(rows[0].keys()) if rows else []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        if answer_col not in fieldnames:
            fieldnames.append(answer_col)
    
    # 分离已完成和未完成的行
    rows_to_process = []
    rows_completed = []
    
    for row in rows:
        keyframe_path = row['key_frame_path']
        if keyframe_path in existing_progress:
            # 使用已有的结果
            rows_completed.append(existing_progress[keyframe_path])
        else:
            # 初始化答案列
            for model_name in model_names:
                clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
                row[f'answer_{clean_model_name}'] = ''
            rows_to_process.append(row)
    
    safe_print(f"需要处理 {len(rows_to_process)} 个条目（跳过已完成的 {len(rows_completed)} 个）")
    
    if len(rows_to_process) == 0:
        safe_print(f"所有条目已完成，跳过第一轮处理")
        rows = rows_completed
    else:
        # 并行处理行（带增量保存）
        safe_print(f"\n开始第一轮处理（并行: {parallel_rows}行）...")
        
        with ThreadPoolExecutor(max_workers=parallel_rows) as executor:
            futures = {executor.submit(process_single_row, row, llms, model_names, 
                                       dataset_root, idx, len(rows_to_process),
                                       max_retries, retry_delay): idx 
                       for idx, row in enumerate(rows_to_process, 1)}
            
            processed_rows = [None] * len(rows_to_process)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    processed_rows[idx - 1] = result
                    # 增量保存
                    save_row_incremental(result, output_csv_path, fieldnames)
                    safe_print(f"[{idx}/{len(rows_to_process)}] ✓ 已保存")
                except Exception as e:
                    safe_print(f"行 {idx} 处理出错: {e}")
                    processed_rows[idx - 1] = rows_to_process[idx - 1]
        
        # 合并所有结果
        rows = rows_completed + processed_rows
    
    safe_print(f"\n{'='*60}")
    safe_print(f"第一轮处理完成")
    safe_print(f"{'='*60}")
    
    # 第一轮统计
    processed_count = {model: 0 for model in model_names}
    error_count = {model: 0 for model in model_names}
    skipped_count = 0
    
    for row in rows:
        question = row.get('question', '')
        if not question or question.startswith('[') or len(question) < 10:
            skipped_count += 1
            continue
        
        for model_name in model_names:
            clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
            answer = row.get(f'answer_{clean_model_name}', '')
            if answer and answer not in ['[ERROR]', '[PARSE_ERROR]', '']:
                processed_count[model_name] += 1
            else:
                error_count[model_name] += 1
    
    safe_print(f"跳过（无问题）: {skipped_count} 个")
    for model_name in model_names:
        safe_print(f"\n模型 {model_name}:")
        safe_print(f"  成功回答: {processed_count[model_name]} 个问题")
        safe_print(f"  失败/错误: {error_count[model_name]} 个")
    safe_print("="*60)
    
    # 第二轮：重试失败的条目
    safe_print(f"\n{'='*60}")
    safe_print(f"开始第二轮重试（处理失败的条目）")
    safe_print(f"{'='*60}")
    
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        
        # 找出该模型失败的行
        failed_rows = []
        failed_indices = []
        for idx, row in enumerate(rows):
            answer = row.get(answer_col, '')
            if not answer or answer in ['[ERROR]', '[PARSE_ERROR]', '']:
                # 确保有有效问题
                question = row.get('question', '')
                if question and not question.startswith('[') and len(question) >= 10:
                    failed_rows.append(row)
                    failed_indices.append(idx)
        
        if failed_rows:
            safe_print(f"\n模型 {model_name}: 发现 {len(failed_rows)} 个失败条目，开始第二轮重试...")
            retry_success_count = 0
            
            for retry_idx, (row_idx, row) in enumerate(zip(failed_indices, failed_rows), 1):
                keyframe_path = row['key_frame_path']
                question = row['question']
                
                safe_print(f"\n[重试 {retry_idx}/{len(failed_rows)}] {keyframe_path}")
                safe_print(f"  问题: {question[:80]}...")
                
                # 构建完整的图片路径
                full_image_path = dataset_root / keyframe_path
                
                if not full_image_path.exists():
                    safe_print(f"  错误: 图片不存在: {full_image_path}")
                    continue
                
                # 重试5次
                success, answer = try_answer_with_retry(
                    llms[model_name], 
                    question, 
                    full_image_path, 
                    5, 
                    retry_delay,
                    model_name
                )
                
                if success:
                    rows[row_idx][answer_col] = answer
                    retry_success_count += 1
                    safe_print(f"  ✓ 答案: {answer}")
                    # 更新文件
                    save_row_incremental(rows[row_idx], output_csv_path, fieldnames)
                else:
                    safe_print(f"  最终失败，跳过此条目")
            
            safe_print(f"\n" + "="*60)
            safe_print(f"模型 {model_name} 重试完成")
            safe_print(f"成功回答: {retry_success_count} 个问题")
            safe_print(f"仍然失败: {len(failed_rows) - retry_success_count} 个")
            safe_print("="*60)
            
            # 更新总计数
            processed_count[model_name] += retry_success_count
    
    # 最终写入完整CSV（确保数据完整）
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    safe_print(f"\n" + "="*60)
    safe_print(f"全部处理完成！")
    safe_print(f"跳过（无问题）: {skipped_count} 个")
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        failed_rows_count = sum(1 for row in rows if not row.get(f'answer_{clean_model_name}') or row.get(f'answer_{clean_model_name}') in ['[ERROR]', '[PARSE_ERROR]'])
        safe_print(f"\n模型 {model_name}:")
        safe_print(f"  总成功回答: {processed_count[model_name]} 个问题")
        safe_print(f"  最终失败: {failed_rows_count} 个")
    safe_print(f"\n输出文件: {output_csv_path}")
    safe_print("="*60)


def main():
    """主函数：处理命令行参数并回答问题。"""
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # kill命令
    
    parser = argparse.ArgumentParser(
        description="使用多个VLM模型回答已有问题 V2.0（支持并行、断点续传、多API轮换）"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集根目录路径"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="输入CSV文件路径（包含问题的文件）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/dataset_multi_answered.csv",
        help="输出CSV文件路径 (默认: output/dataset_multi_answered.csv)"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="使用的模型名称列表，用逗号分隔 (例如: gpt-4o,gpt-4o-mini,deepseek-chat)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="第一轮API调用失败时的最大重试次数 (默认: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=1,
        help="重试之间的延迟秒数 (默认: 1)"
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default='chatanywhere',
        help="模型URL (默认: chatanywhere)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=5,
        help="并行处理的行数 (默认: 5)"
    )
    
    args = parser.parse_args()
    
    # 解析模型列表
    model_names = [m.strip() for m in args.models.split(',')]
    
    if not model_names:
        print("错误: 至少需要指定一个模型")
        sys.exit(1)
    
    dataset_path = Path(args.dataset_path)
    input_csv_path = Path(args.csv)
    output_csv_path = Path(args.output)
    
    # 验证路径
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    if not input_csv_path.exists():
        print(f"错误: 输入CSV文件不存在: {input_csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"多模型VLM问题回答器 V2.0")
    print(f"{'='*60}")
    print(f"数据集路径: {dataset_path}")
    print(f"输入CSV: {input_csv_path}")
    print(f"输出CSV: {output_csv_path}")
    print(f"模型列表: {', '.join(model_names)}")
    print(f"模型URL: {args.model_url}")
    print(f"第一轮最大重试次数: {args.max_retries}")
    print(f"并行处理行数: {args.parallel_rows}")
    print(f"\n新功能:")
    print(f"  ✓ 并行行处理（{args.parallel_rows}行并发）")
    print(f"  ✓ 实时保存（每行处理完立即写入）")
    print(f"  ✓ 断点续传（中断后可继续）")
    print(f"  ✓ 信号处理（Ctrl+C安全退出）")
    print(f"  ✓ 多API Key轮换（防止单个key被限流）")
    print(f"  ✓ 两轮重试（第一轮{args.max_retries}次，第二轮5次）")
    print(f"{'='*60}\n")
    
    process_dataset_multi_models(
        input_csv_path,
        output_csv_path,
        dataset_path,
        model_names,
        args.model_url,
        args.max_retries,
        args.retry_delay,
        args.parallel_rows
    )


if __name__ == "__main__":
    main()



