#!/usr/bin/env python3
"""
并行多模型测试脚本
同时处理3个CSV数据集，使用22个VLM模型并行回答问题并评估性能。
"""


import os
import sys
import argparse
import csv
import base64
import time
import pandas as pd
import signal
import random
import fcntl
import json
from pathlib import Path
from typing import Dict, List, Tuple
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import threading

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from PIL import Image


# 线程安全的打印锁和文件锁
print_lock = threading.Lock()
file_lock = threading.Lock()

# 全局变量用于信号处理
current_csv_data = {}
current_output_path = None


def safe_print(*args, **kwargs):
    """线程安全的打印函数"""
    with print_lock:
        print(*args, **kwargs)


def get_api_endpoint(provider: str) -> str:
    """
    获取API提供商的端点URL（固定URL，不轮换）
    
    Args:
        provider: 'openrouter' 或 'chatanywhere'
    
    Returns:
        端点URL
    """
    if provider == 'openrouter':
        return 'https://openrouter.ai/api/v1'
    else:  # chatanywhere
        return 'https://api.chatanywhere.org/v1'


def get_random_api_key(provider: str) -> str:
    """
    获取随机的API key（如果配置了多个）
    
    Args:
        provider: 'openrouter' 或 'chatanywhere'
    
    Returns:
        API key
    """
    if provider == 'openrouter':
        # 支持多个OpenRouter API key（用分号分隔）
        keys_str = os.getenv("OPENROUTER_API_KEY", "")
        if not keys_str:
            raise ValueError("OPENROUTER_API_KEY 环境变量未设置（OpenRouter模型需要）")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    else:
        # 支持多个ChatAnywhere API key（用分号分隔）
        keys_str = os.getenv("CHATANYWHERE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not keys_str:
            raise ValueError("CHATANYWHERE_API_KEY 或 OPENAI_API_KEY 环境变量未设置（ChatAnywhere模型需要）")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)


def encode_image_to_base64(image_path: Path, max_size_mb: float = 5.0) -> str:
    """简单压缩图片"""
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


def encode_image_to_jpeg_strict(image_path: Path, max_size_mb: float = 2.0) -> str:
    """
    强制转换为JPEG格式并压缩（用于Amazon Nova、xAI Grok等有严格要求的模型）
    
    Args:
        image_path: 图片路径
        max_size_mb: 最大文件大小（MB）
    
    Returns:
        base64编码的JPEG图片
    """
    # 打开图片
    img = Image.open(image_path)
    
    # 转换为RGB（JPEG不支持透明通道）
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # 逐步降低质量直到符合要求
    for q in range(85, 30, -5):
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=q, optimize=True)
        size_mb = buffer.tell() / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    # 极端情况：大幅缩小尺寸
    scale = 0.6
    while scale > 0.2:
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = BytesIO()
        resized.save(buffer, format='JPEG', quality=60, optimize=True)
        size_mb = buffer.tell() / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        
        scale -= 0.1
    
    # 最后手段：极小质量
    buffer = BytesIO()
    img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.LANCZOS).save(
        buffer, format='JPEG', quality=40, optimize=True
    )
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_model_provider(model_name: str) -> str:
    """
    根据模型名称判断应该使用哪个API提供商。
    
    Args:
        model_name (str): 模型名称
    
    Returns:
        str: 'openrouter' 或 'chatanywhere'
    """
    # OpenRouter模型列表（与实际使用的模型列表一致）
    openrouter_models = {
        # Qwen VL系列 (3个)
        'qwen/qwen2.5-vl-72b-instruct',
        'qwen/qwen2.5-vl-32b-instruct',
        'qwen/qwen-vl-max',
        # Anthropic Claude系列 (3个)
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.7-sonnet',
        'anthropic/claude-3-haiku',
        # Google Gemini系列 (3个)
        'google/gemini-2.5-flash-image',
        'google/gemini-2.5-flash',
        'google/gemini-pro-vision',
        # OpenAI系列 (3个)
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        # 其他VLM模型 (3个)
        'x-ai/grok-vision-beta',
        'meta-llama/llama-3.2-90b-vision-instruct',
        'meta-llama/llama-3.2-11b-vision-instruct',
        'x-ai/grok-3',
    }
    
    if model_name in openrouter_models:
        return 'openrouter'
    else:
        return 'chatanywhere'


def create_llm(model_name: str) -> ChatOpenAI:
    """
    创建LLM实例，根据模型自动选择API提供商（支持多API key轮换）。
    
    Args:
        model_name (str): 模型名称
    
    Returns:
        ChatOpenAI: LLM实例
    """
    provider = get_model_provider(model_name)
    
    # 随机选择API key（防止被ban），但URL固定
    api_key = get_random_api_key(provider)
    base_url = get_api_endpoint(provider)
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        max_tokens=10000,
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
    使用多模态LLM回答问题。
    
    Args:
        llm (ChatOpenAI): LLM实例
        question (str): 要回答的问题
        image_path (Path): 关键帧图片路径
        model_name (str): 模型名称
    
    Returns:
        str: 生成的答案
    """
    try:
        model_lower = model_name.lower()
        
        # 特殊模型需要严格的JPEG格式和小尺寸
        # Amazon Nova: 只接受JPEG，不接受PNG
        # xAI Grok: 图片太大会返回413错误
        if 'nova' in model_lower or 'amazon' in model_lower:
            # Amazon Nova需要JPEG + 中等压缩
            base64_image = encode_image_to_jpeg_strict(image_path, max_size_mb=3.0)
        elif 'grok' in model_lower or 'x-ai' in model_lower:
            # Grok需要JPEG + 严格压缩
            base64_image = encode_image_to_jpeg_strict(image_path, max_size_mb=1.5)
        elif 'claude' in model_lower or 'anthropic' in model_lower or 'deepseek' in model_lower:
            # Claude/DeepSeek需要一般压缩
            base64_image = encode_image_to_base64(image_path, max_size_mb=4.5)
        else:
            # 其他模型：直接读取原始文件
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 构建提示词
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
        print(response)
        response_text = response.content.strip()
        
        # 提取答案字母
        answer = response_text.upper()
        
        # 清理答案
        if "ANSWER:" in answer:
            answer = answer.split(":")[-1].strip()
        
        # 提取单个字母
        for char in answer:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return "[PARSE_ERROR]"
            
    except Exception as e:
        safe_print(f"  错误: {e}")
        return "[ERROR]"


def answer_question_with_retry(
    llm: ChatOpenAI,
    question: str,
    image_path: Path,
    model_name: str,
    max_retries: int = 2,
    retry_delay: int = 1
) -> str:
    """
    尝试回答问题，失败时自动重试。
    
    Args:
        llm: LLM实例
        question: 问题文本
        image_path: 图片路径
        model_name: 模型名称
        max_retries: 最大重试次数（默认2次）
        retry_delay: 重试延迟秒数（默认1秒）
    
    Returns:
        答案字符串
    """
    for attempt in range(max_retries):
        try:
            answer = answer_question(llm, question, image_path, model_name)
            
            # 如果得到有效答案，直接返回
            if answer not in ["[ERROR]", "[PARSE_ERROR]"]:
                if attempt > 0:
                    safe_print(f"  ✓ 重试成功（第{attempt + 1}次尝试）")
                return answer
            else:
                # 答案无效，需要重试
                raise Exception(f"返回了错误标记: {answer}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"  ⚠️  尝试 {attempt + 1} 失败: {str(e)[:80]}")
                safe_print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  ✗ 所有尝试都失败了")
    
    return "[ERROR]"


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
    answered_csv_path: Path = None,
    fieldnames: List[str] = None
) -> dict:
    """
    处理单行数据，使用所有模型串行回答问题（避免过载）。
    
    Args:
        row: CSV行数据
        llms: 模型名称到LLM实例的映射
        model_names: 模型名称列表
        dataset_root: 数据集根目录
        row_idx: 行索引
        total_rows: 总行数
        answered_csv_path: 答案CSV路径（用于增量保存）
        fieldnames: CSV字段名（用于增量保存）
    
    Returns:
        更新后的行数据
    """
    keyframe_path = row['key_frame_path']
    question = row['question']
    
    # 检查是否有有效问题
    if not question or question.startswith('[') or len(question) < 10:
        return row
    
    # 构建完整的图片路径
    full_image_path = dataset_root / keyframe_path
    
    if not full_image_path.exists():
        safe_print(f"[{row_idx}/{total_rows}] 错误: 图片不存在: {full_image_path}")
        return row
    
    safe_print(f"[{row_idx}/{total_rows}] 处理: {keyframe_path}")
    
    # 串行处理所有模型（避免过度并发）
    for model_idx, model_name in enumerate(model_names, 1):
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        safe_print(f"  [{model_idx}/{len(model_names)}] 模型: {model_name}")
        
        try:
            answer = answer_question_with_retry(
                llms[model_name],
                question,
                full_image_path,
                model_name
            )
            row[f'answer_{clean_model_name}'] = answer
        except Exception as e:
            safe_print(f"    错误: {e}")
            row[f'answer_{clean_model_name}'] = "[ERROR]"
    
    # 处理完成后立即保存（增量保存）
    if answered_csv_path and fieldnames:
        save_row_incremental(row, answered_csv_path, fieldnames)
        safe_print(f"[{row_idx}/{total_rows}] ✓ 已保存")
    
    return row


def evaluate_model_on_dataset(df: pd.DataFrame, answer_col: str, model_name: str) -> dict:
    """
    评估单个模型在数据集上的性能（包含分类别统计）。
    
    Args:
        df: 包含问题、答案和ground truth的DataFrame
        answer_col: 答案列名
        model_name: 模型名称
    
    Returns:
        dict: 评估结果统计
    """
    # 过滤掉没有答案的行
    valid_df = df[df[answer_col].notna() & (df[answer_col] != '') & 
                  (~df[answer_col].isin(['[ERROR]', '[PARSE_ERROR]']))].copy()
    
    if len(valid_df) == 0:
        return {
            'model': model_name,
            'overall_accuracy': 0.0,
            'total': 0,
            'correct': 0,
            'errors': len(df),
            'category_accuracy': {}
        }
    
    # 标准化答案
    valid_df['answer_norm'] = valid_df[answer_col].astype(str).str.upper().str.strip()
    valid_df['gt_norm'] = valid_df['ground truth'].astype(str).str.upper().str.strip()
    
    # 计算准确率
    valid_df['correct'] = valid_df['answer_norm'] == valid_df['gt_norm']
    
    total_accuracy = valid_df['correct'].mean()
    
    # 计算分类别准确率
    category_accuracy = {}
    if 'question_category' in valid_df.columns:
        for category in sorted(valid_df['question_category'].unique()):
            cat_data = valid_df[valid_df['question_category'] == category]
            cat_acc = cat_data['correct'].mean()
            category_accuracy[category] = {
                'accuracy': cat_acc,
                'correct': cat_data['correct'].sum(),
                'total': len(cat_data)
            }
    
    return {
        'model': model_name,
        'overall_accuracy': total_accuracy,
        'total': len(valid_df),
        'correct': valid_df['correct'].sum(),
        'errors': len(df) - len(valid_df),
        'category_accuracy': category_accuracy,
        'valid_df': valid_df
    }


def process_single_csv(
    csv_name: str,
    csv_path: Path,
    dataset_root: Path,
    llms: Dict[str, ChatOpenAI],
    model_names: List[str],
    output_base_dir: Path,
    parallel_rows: int = 5
) -> dict:
    """
    处理单个CSV文件：回答问题 + 评估（支持断点续传和增量保存）。
    
    Args:
        csv_name: CSV名称（用于输出）
        csv_path: CSV文件路径
        dataset_root: 数据集根目录
        llms: 模型字典
        model_names: 模型名称列表
        output_base_dir: 输出基础目录
        parallel_rows: 并行处理的行数（模型串行处理）
    
    Returns:
        dict: 处理结果统计
    """
    safe_print(f"\n{'='*60}")
    safe_print(f"开始处理: {csv_name}")
    safe_print(f"{'='*60}")
    
    start_time = time.time()
    
    # 准备输出目录
    output_dir = output_base_dir / csv_name
    output_dir.mkdir(parents=True, exist_ok=True)
    answered_csv_path = output_dir / f"answered_{csv_name}.csv"
    
    # 读取CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    safe_print(f"读取了 {len(rows)} 个条目")
    
    # 加载已完成的进度（断点续传）
    existing_progress = load_existing_progress(answered_csv_path)
    safe_print(f"已完成 {len(existing_progress)} 个条目（断点续传）")
    
    # 为每个模型初始化答案列
    fieldnames = list(rows[0].keys()) if rows else []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        fieldnames.append(f'answer_{clean_model_name}')
    
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
        safe_print(f"所有条目已完成，跳过处理")
        rows = rows_completed
    else:
        # 并行处理行（每行内部模型串行，带增量保存）
        safe_print(f"开始并行处理（行并发: {parallel_rows}，模型串行，实时保存）...")
        
        with ThreadPoolExecutor(max_workers=parallel_rows) as executor:
            futures = {executor.submit(process_single_row, row, llms, model_names, 
                                       dataset_root, idx, len(rows_to_process),
                                       answered_csv_path, fieldnames): idx 
                       for idx, row in enumerate(rows_to_process, 1)}
            
            processed_rows = [None] * len(rows_to_process)
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    processed_rows[idx - 1] = future.result()
                except Exception as e:
                    safe_print(f"行 {idx} 处理出错: {e}")
                    processed_rows[idx - 1] = rows_to_process[idx - 1]
        
        # 合并所有结果
        rows = rows_completed + processed_rows
    
    safe_print(f"✓ 所有答案已保存到: {answered_csv_path}")
    
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
            retry_success = 0
            
            for retry_idx, (row_idx, row) in enumerate(zip(failed_indices, failed_rows), 1):
                keyframe_path = row['key_frame_path']
                question = row['question']
                full_image_path = dataset_root / keyframe_path
                
                if not full_image_path.exists():
                    continue
                
                safe_print(f"  [重试 {retry_idx}/{len(failed_rows)}] {keyframe_path}")
                
                # 第二轮也尝试2次
                answer = answer_question_with_retry(
                    llms[model_name],
                    question,
                    full_image_path,
                    model_name,
                    max_retries=2,
                    retry_delay=1
                )
                
                if answer not in ['[ERROR]', '[PARSE_ERROR]']:
                    rows[row_idx][answer_col] = answer
                    retry_success += 1
                    safe_print(f"    ✓ 第二轮成功")
                    # 更新文件
                    save_row_incremental(rows[row_idx], answered_csv_path, fieldnames)
            
            safe_print(f"  模型 {model_name} 第二轮完成: 成功 {retry_success}/{len(failed_rows)}")
    
    safe_print(f"\n{'='*60}")
    safe_print(f"第二轮重试完成")
    safe_print(f"{'='*60}\n")
    
    # 评估所有模型
    safe_print(f"\n开始评估所有模型...")
    df = pd.DataFrame(rows)
    
    evaluation_results = []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        
        result = evaluate_model_on_dataset(df, answer_col, model_name)
        evaluation_results.append(result)
        
        # 保存详细评估结果
        if result['total'] > 0:
            valid_df = result['valid_df']
            eval_csv_path = output_dir / f"evaluation_{clean_model_name}.csv"
            valid_df.to_csv(eval_csv_path, index=False)
        
        safe_print(f"  {model_name}: {result['overall_accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    elapsed_time = time.time() - start_time
    safe_print(f"\n{csv_name} 处理完成，耗时: {elapsed_time:.2f}秒")
    
    return {
        'csv_name': csv_name,
        'total_rows': len(rows),
        'evaluation_results': evaluation_results,
        'elapsed_time': elapsed_time
    }


def test_model_connectivity(llms: Dict[str, ChatOpenAI]) -> Tuple[Dict[str, ChatOpenAI], List[str]]:
    """
    测试所有模型的连通性，返回可用的模型。
    
    Args:
        llms: 模型字典
    
    Returns:
        Tuple[Dict[str, ChatOpenAI], List[str]]: (可用的模型字典, 可用的模型名称列表)
    """
    safe_print("开始测试模型连通性...")
    
    available_llms = {}
    available_models = []
    failed_models = []
    
    for model_name, llm in llms.items():
        safe_print(f"测试模型: {model_name}")
        
        try:
            test_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Hello, this is a connectivity test. Please respond with 'OK'.",
                    }
                ]
            )
            
            response = llm.invoke([test_message])
            response_text = response.content.strip()
            
            safe_print(f"  ✓ 连接成功: {response_text[:50]}...")
            available_llms[model_name] = llm
            available_models.append(model_name)
            
        except Exception as e:
            safe_print(f"  ✗ 连接失败: {str(e)[:100]}...")
            safe_print(f"  ⚠️  该模型将被跳过")
            failed_models.append(model_name)
    
    safe_print(f"\n连通性测试完成:")
    safe_print(f"  ✓ 可用模型: {len(available_models)} 个")
    safe_print(f"  ✗ 失败模型: {len(failed_models)} 个")
    
    if failed_models:
        safe_print(f"\n跳过的模型:")
        for model in failed_models:
            safe_print(f"  - {model}")
    
    safe_print()
    
    return available_llms, available_models


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


def main():
    """主函数：并行处理多个CSV并评估所有模型（支持断点续传和信号处理）。"""
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # kill命令
    
    parser = argparse.ArgumentParser(
        description="并行多模型测试脚本（支持断点续传、多API轮换）"
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
        help="要处理的CSV文件路径"
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        required=True,
        help="CSV文件名称（用于输出目录，如：apl, arrowtrajectory, nfl）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=10,
        help="并行处理的行数，模型在每行内串行处理 (默认: 25)"
    )
    
    args = parser.parse_args()
    
    # 定义15个OpenRouter VLM模型（这次全部使用OpenRouter）
    model_names = [
        # === OpenRouter组（15个）===
        
        # Qwen VL系列 (3个) - 仅使用OpenRouter上存在的
        'qwen/qwen2.5-vl-72b-instruct',
        'qwen/qwen2.5-vl-32b-instruct',
        'qwen/qwen-vl-max',
        
        # Anthropic Claude系列 (2个)
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.7-sonnet',
        
        # Google Gemini系列 (2个)
  
        'google/gemini-2.5-flash-image',
        'google/gemini-2.5-flash',
        
        # OpenAI系列 (3个)
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        
        # 其他支持Vision的VLM模型 (5个)
        # 'x-ai/grok-3',
        'meta-llama/llama-3.2-90b-vision-instruct',
        'meta-llama/llama-3.2-11b-vision-instruct',
        # 'anthropic/claude-3-haiku',
        'google/gemini-pro-vision',
    ]
    
    dataset_path = Path(args.dataset_path)
    csv_path = Path(args.csv)
    csv_name = args.csv_name
    output_base_dir = Path(args.output)
    
    # 验证路径
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"错误: CSV文件不存在: {csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"并行多模型测试脚本 v2.0")
    print(f"{'='*60}")
    print(f"数据集路径: {dataset_path}")
    print(f"CSV文件: {csv_path}")
    print(f"CSV名称: {csv_name}")
    print(f"输出目录: {output_base_dir}")
    print(f"\n模型配置:")
    print(f"  - 总模型数: {len(model_names)}")
    print(f"  - OpenRouter: {sum(1 for m in model_names if get_model_provider(m) == 'openrouter')} 个")
    print(f"  - ChatAnywhere: {sum(1 for m in model_names if get_model_provider(m) == 'chatanywhere')} 个")
    print(f"\n并发策略:")
    print(f"  - 行处理: {args.parallel_rows}行并行")
    print(f"  - 模型处理: 串行（避免过载）")
    print(f"  - 最大并发API请求: {args.parallel_rows} 个")
    print(f"\n新功能:")
    print(f"  ✓ 实时保存（每行处理完立即写入）")
    print(f"  ✓ 断点续传（中断后可继续）")
    print(f"  ✓ 信号处理（Ctrl+C安全退出）")
    print(f"  ✓ 多API Key轮换（防止单个key被限流）")
    print(f"\n其他配置:")
    print(f"  - 重试策略: 两轮重试（第一轮2次，第二轮2次）")
    print(f"  - 超时时间: 600秒")
    print()
    
    # 初始化所有模型
    print(f"初始化 {len(model_names)} 个模型...")
    llms = {}
    for model_name in model_names:
        print(f"  - {model_name}")
        try:
            llms[model_name] = create_llm(model_name)
        except ValueError as e:
            print(f"  错误: {e}")
            sys.exit(1)
    
    print(f"\n所有模型初始化完成！\n")
    
    # 测试模型连通性，只使用可用的模型
    available_llms, available_models = test_model_connectivity(llms)
    
    if len(available_models) == 0:
        print("错误: 没有可用的模型！")
        sys.exit(1)
    
    print(f"将使用 {len(available_models)} 个可用模型继续处理\n")
    
    # 处理单个CSV文件
    print(f"\n{'='*60}")
    print(f"开始处理CSV文件: {csv_name}")
    print(f"{'='*60}\n")
    
    overall_start = time.time()
    
    try:
        result = process_single_csv(
            csv_name,
            csv_path,
            dataset_path,
            available_llms,  # 只使用可用的模型
            available_models,  # 只使用可用的模型名称
            output_base_dir,
            args.parallel_rows
        )
    except Exception as e:
        print(f"错误: CSV处理失败: {e}")
        sys.exit(1)
    
    overall_elapsed = time.time() - overall_start
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"处理完成！总耗时: {overall_elapsed:.2f}秒")
    print(f"{'='*60}\n")
    
    # 打印数据集统计
    print(f"\n{csv_name.upper()} 数据集:")
    print(f"  总条目: {result['total_rows']}")
    print(f"  耗时: {result['elapsed_time']:.2f}秒")
    print(f"  各模型准确率:")
    
    # 按准确率排序
    sorted_results = sorted(result['evaluation_results'], 
                           key=lambda x: x['overall_accuracy'], 
                           reverse=True)
    
    for eval_result in sorted_results:
        acc_str = f"{eval_result['overall_accuracy']:.2%}"
        count_str = f"{eval_result['correct']}/{eval_result['total']}"
        print(f"    {eval_result['model']:<40} {acc_str:<10} ({count_str})")
        
        # 显示分类别准确率
        if 'category_accuracy' in eval_result and eval_result['category_accuracy']:
            for category, stats in sorted(eval_result['category_accuracy'].items()):
                cat_acc_str = f"{stats['accuracy']:.2%}"
                cat_count_str = f"{stats['correct']}/{stats['total']}"
                print(f"      └─ {category:<35} {cat_acc_str:<10} ({cat_count_str})")
    
    print(f"\n{'='*60}")
    print(f"所有结果已保存到: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("！Starting test_parallel_multi_models.py")
    main()

