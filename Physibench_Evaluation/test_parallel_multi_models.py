#!/usr/bin/env python3
"""
Parallel multi-model evaluation script.
Processes three CSV datasets, uses multiple VLM models to answer questions in
parallel, and evaluates their performance.
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


# Thread-safe print lock and file lock
print_lock = threading.Lock()
file_lock = threading.Lock()

# Global state used by signal handling
current_csv_data = {}
current_output_path = None


def safe_print(*args, **kwargs):
    """Thread-safe print helper."""
    with print_lock:
        print(*args, **kwargs)


def get_api_endpoint(provider: str) -> str:
    """
    Return the fixed endpoint URL for the selected API provider.
    
    Args:
        provider: 'openrouter' or 'chatanywhere'
    
    Returns:
        Endpoint URL
    """
    if provider == 'openrouter':
        return 'https://openrouter.ai/api/v1'
    else:  # chatanywhere
        return 'https://api.chatanywhere.org/v1'


def get_random_api_key(provider: str) -> str:
    """
    Get a random API key when multiple keys are configured.
    
    Args:
        provider: 'openrouter' or 'chatanywhere'
    
    Returns:
        API key
    """
    if provider == 'openrouter':
        # Multiple OpenRouter API keys can be provided, separated by semicolons.
        keys_str = os.getenv("OPENROUTER_API_KEY", "")
        if not keys_str:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set (required for OpenRouter models)")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    else:
        # Multiple ChatAnywhere API keys can be provided, separated by semicolons.
        keys_str = os.getenv("CHATANYWHERE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not keys_str:
            raise ValueError("CHATANYWHERE_API_KEY or OPENAI_API_KEY environment variable is not set (required for ChatAnywhere models)")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)


def encode_image_to_base64(image_path: Path, max_size_mb: float = 5.0) -> str:
    """Apply simple image compression when needed."""
    data = image_path.read_bytes()
    orig_size_mb = len(data) / (1024 * 1024)
    
    if orig_size_mb <= max_size_mb:
        return base64.b64encode(data).decode('utf-8')
    
    # Compression is required.
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Gradually reduce quality until the size target is met.
    for q in range(95, 45, -5):
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=q, optimize=True)
        comp_size_mb = buffer.tell() / (1024 * 1024)
        if comp_size_mb <= max_size_mb:
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    # Fallback: reduce the image dimensions.
    new_width = int(img.width * 0.7)
    new_height = int(img.height * 0.7)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def encode_image_to_jpeg_strict(image_path: Path, max_size_mb: float = 2.0) -> str:
    """
    Force conversion to JPEG and compress aggressively for strict model APIs.
    
    Args:
        image_path: Image path
        max_size_mb: Maximum file size in MB
    
    Returns:
        Base64-encoded JPEG image
    """
    # Open the image.
    img = Image.open(image_path)
    
    # Convert to RGB because JPEG does not support alpha channels.
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Gradually reduce quality until the size target is met.
    for q in range(85, 30, -5):
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=q, optimize=True)
        size_mb = buffer.tell() / (1024 * 1024)
        
        if size_mb <= max_size_mb:
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
    
    # Fallback: significantly reduce dimensions.
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
    
    # Last resort: save at very low quality.
    buffer = BytesIO()
    img.resize((int(img.width * 0.3), int(img.height * 0.3)), Image.LANCZOS).save(
        buffer, format='JPEG', quality=40, optimize=True
    )
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


def get_model_provider(model_name: str) -> str:
    """
    Determine which API provider should be used for a given model.
    
    Args:
        model_name (str): Model name
    
    Returns:
        str: 'openrouter' or 'chatanywhere'
    """
    # OpenRouter model list, matching the models actually used by this script.
    openrouter_models = {
        # Qwen VL family (3)
        'qwen/qwen2.5-vl-72b-instruct',
        'qwen/qwen2.5-vl-32b-instruct',
        'qwen/qwen-vl-max',
        # Anthropic Claude family (3)
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.7-sonnet',
        'anthropic/claude-3-haiku',
        # Google Gemini family (3)
        'google/gemini-2.5-flash-image',
        'google/gemini-2.5-flash',
        'google/gemini-pro-vision',
        # OpenAI family (3)
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        # Other VLM models
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
    Create an LLM instance and automatically select the API provider.
    
    Args:
        model_name (str): Model name
    
    Returns:
        ChatOpenAI: LLM instance
    """
    provider = get_model_provider(model_name)
    
    # Randomize API key selection while keeping the endpoint fixed.
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
    Answer a question with a multimodal LLM.
    
    Args:
        llm (ChatOpenAI): LLM instance
        question (str): Question to answer
        image_path (Path): Keyframe image path
        model_name (str): Model name
    
    Returns:
        str: Generated answer
    """
    try:
        model_lower = model_name.lower()
        
        # Some models require strict JPEG input and smaller image sizes.
        # Amazon Nova accepts JPEG only.
        # xAI Grok may return HTTP 413 for oversized images.
        if 'nova' in model_lower or 'amazon' in model_lower:
            # Amazon Nova needs JPEG with moderate compression.
            base64_image = encode_image_to_jpeg_strict(image_path, max_size_mb=3.0)
        elif 'grok' in model_lower or 'x-ai' in model_lower:
            # Grok needs JPEG with stricter compression.
            base64_image = encode_image_to_jpeg_strict(image_path, max_size_mb=1.5)
        elif 'claude' in model_lower or 'anthropic' in model_lower or 'deepseek' in model_lower:
            # Claude and DeepSeek need moderate compression.
            base64_image = encode_image_to_base64(image_path, max_size_mb=4.5)
        else:
            # Other models can use the original image directly.
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Build the prompt.
        prompt = f"""Please answer the following multiple choice question based on the image.

Question: {question}

Instructions:
- Look carefully at the image
- Choose the correct answer from the options provided
- Only return the letter of your answer (A, B, C, or D)
- Format: Just output "Answer: X" where X is A, B, C, or D

Your answer:"""
        
        # Create a message that includes the image.
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
        
        # Generate the response.
        response = llm.invoke([message])
        print(response)
        response_text = response.content.strip()
        
        # Extract the answer letter.
        answer = response_text.upper()
        
        # Normalize the answer text.
        if "ANSWER:" in answer:
            answer = answer.split(":")[-1].strip()
        
        # Extract a single answer letter.
        for char in answer:
            if char in ['A', 'B', 'C', 'D']:
                return char
        
        return "[PARSE_ERROR]"
            
    except Exception as e:
        safe_print(f"  Error: {e}")
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
    Attempt to answer a question and retry automatically on failure.
    
    Args:
        llm: LLM instance
        question: Question text
        image_path: Image path
        model_name: Model name
        max_retries: Maximum number of retries (default: 2)
        retry_delay: Retry delay in seconds (default: 1)
    
    Returns:
        Answer string
    """
    for attempt in range(max_retries):
        try:
            answer = answer_question(llm, question, image_path, model_name)
            
            # Return immediately if a valid answer is received.
            if answer not in ["[ERROR]", "[PARSE_ERROR]"]:
                if attempt > 0:
                    safe_print(f"  ✓ Retry succeeded on attempt {attempt + 1}")
                return answer
            else:
                # Invalid answer, retry.
                raise Exception(f"Model returned an error marker: {answer}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:80]}")
                safe_print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  ✗ All attempts failed")
    
    return "[ERROR]"


def save_row_incremental(row: dict, csv_path: Path, fieldnames: List[str]):
    """
    Incrementally save a single row in a thread-safe way.
    
    Args:
        row: Row data to save
        csv_path: CSV file path
        fieldnames: List of CSV field names
    """
    with file_lock:
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            # Use a file lock to avoid write conflicts across threads.
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())  # Force a disk flush.
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_existing_progress(csv_path: Path) -> Dict[str, dict]:
    """
    Load existing progress from a previous run.
    
    Args:
        csv_path: CSV file path
    
    Returns:
        Dict mapping `key_frame_path` to row data
    """
    if not csv_path.exists():
        return {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return {row['key_frame_path']: row for row in reader}
    except Exception as e:
        safe_print(f"Warning: could not load progress file {csv_path}: {e}")
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
    Process one CSV row, answering the question with each model serially.
    
    Args:
        row: CSV row data
        llms: Mapping from model name to LLM instance
        model_names: List of model names
        dataset_root: Dataset root directory
        row_idx: Row index
        total_rows: Total number of rows
        answered_csv_path: Answer CSV path used for incremental saving
        fieldnames: CSV field names used for incremental saving
    
    Returns:
        Updated row data
    """
    keyframe_path = row['key_frame_path']
    question = row['question']
    
    # Skip rows without a valid question.
    if not question or question.startswith('[') or len(question) < 10:
        return row
    
    # Build the full image path.
    full_image_path = dataset_root / keyframe_path
    
    if not full_image_path.exists():
        safe_print(f"[{row_idx}/{total_rows}] Error: image not found: {full_image_path}")
        return row
    
    safe_print(f"[{row_idx}/{total_rows}] Processing: {keyframe_path}")
    
    # Process models serially within the row to avoid overload.
    for model_idx, model_name in enumerate(model_names, 1):
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        safe_print(f"  [{model_idx}/{len(model_names)}] Model: {model_name}")
        
        try:
            answer = answer_question_with_retry(
                llms[model_name],
                question,
                full_image_path,
                model_name
            )
            row[f'answer_{clean_model_name}'] = answer
        except Exception as e:
            safe_print(f"    Error: {e}")
            row[f'answer_{clean_model_name}'] = "[ERROR]"
    
    # Save immediately after the row finishes.
    if answered_csv_path and fieldnames:
        save_row_incremental(row, answered_csv_path, fieldnames)
        safe_print(f"[{row_idx}/{total_rows}] ✓ Saved")
    
    return row


def evaluate_model_on_dataset(df: pd.DataFrame, answer_col: str, model_name: str) -> dict:
    """
    Evaluate one model on the dataset, including per-category statistics.
    
    Args:
        df: DataFrame containing questions, answers, and ground truth
        answer_col: Answer column name
        model_name: Model name
    
    Returns:
        dict: Evaluation statistics
    """
    # Filter out rows without usable answers.
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
    
    # Normalize answers.
    valid_df['answer_norm'] = valid_df[answer_col].astype(str).str.upper().str.strip()
    valid_df['gt_norm'] = valid_df['ground truth'].astype(str).str.upper().str.strip()
    
    # Compute accuracy.
    valid_df['correct'] = valid_df['answer_norm'] == valid_df['gt_norm']
    
    total_accuracy = valid_df['correct'].mean()
    
    # Compute per-category accuracy.
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
    Process one CSV file: answer questions and evaluate the results.
    
    Args:
        csv_name: CSV name used for output
        csv_path: CSV file path
        dataset_root: Dataset root directory
        llms: Model dictionary
        model_names: List of model names
        output_base_dir: Base output directory
        parallel_rows: Number of rows to process in parallel
    
    Returns:
        dict: Processing statistics
    """
    safe_print(f"\n{'='*60}")
    safe_print(f"Starting processing: {csv_name}")
    safe_print(f"{'='*60}")
    
    start_time = time.time()
    
    # Prepare output directories.
    output_dir = output_base_dir / csv_name
    output_dir.mkdir(parents=True, exist_ok=True)
    answered_csv_path = output_dir / f"answered_{csv_name}.csv"
    
    # Read the CSV.
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    safe_print(f"Read {len(rows)} entries")
    
    # Load progress from any previous run.
    existing_progress = load_existing_progress(answered_csv_path)
    safe_print(f"Loaded {len(existing_progress)} completed entries from previous progress")
    
    # Initialize answer columns for each model.
    fieldnames = list(rows[0].keys()) if rows else []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        fieldnames.append(f'answer_{clean_model_name}')
    
    # Split rows into completed and pending sets.
    rows_to_process = []
    rows_completed = []
    
    for row in rows:
        keyframe_path = row['key_frame_path']
        if keyframe_path in existing_progress:
            # Reuse the existing result.
            rows_completed.append(existing_progress[keyframe_path])
        else:
            # Initialize answer columns.
            for model_name in model_names:
                clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
                row[f'answer_{clean_model_name}'] = ''
            rows_to_process.append(row)
    
    safe_print(f"{len(rows_to_process)} entries need processing ({len(rows_completed)} already completed)")
    
    if len(rows_to_process) == 0:
        safe_print(f"All entries are already complete; skipping processing")
        rows = rows_completed
    else:
        # Process rows in parallel, with models handled serially inside each row.
        safe_print(f"Starting parallel processing (parallel rows: {parallel_rows}, serial models, real-time saving)...")
        
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
                    safe_print(f"Row {idx} failed: {e}")
                    processed_rows[idx - 1] = rows_to_process[idx - 1]
        
        # Merge all results.
        rows = rows_completed + processed_rows
    
    safe_print(f"✓ All answers have been saved to: {answered_csv_path}")
    
    # Second pass: retry failed entries.
    safe_print(f"\n{'='*60}")
    safe_print(f"Starting second-pass retries for failed entries")
    safe_print(f"{'='*60}")
    
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        
        # Find rows that failed for this model.
        failed_rows = []
        failed_indices = []
        for idx, row in enumerate(rows):
            answer = row.get(answer_col, '')
            if not answer or answer in ['[ERROR]', '[PARSE_ERROR]', '']:
                # Only retry rows with a valid question.
                question = row.get('question', '')
                if question and not question.startswith('[') and len(question) >= 10:
                    failed_rows.append(row)
                    failed_indices.append(idx)
        
        if failed_rows:
            safe_print(f"\nModel {model_name}: found {len(failed_rows)} failed entries, starting second-pass retries...")
            retry_success = 0
            
            for retry_idx, (row_idx, row) in enumerate(zip(failed_indices, failed_rows), 1):
                keyframe_path = row['key_frame_path']
                question = row['question']
                full_image_path = dataset_root / keyframe_path
                
                if not full_image_path.exists():
                    continue
                
                safe_print(f"  [Retry {retry_idx}/{len(failed_rows)}] {keyframe_path}")
                
                # The second pass also uses two attempts.
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
                    safe_print(f"    ✓ Second pass succeeded")
                    # Update the progress file.
                    save_row_incremental(rows[row_idx], answered_csv_path, fieldnames)
            
            safe_print(f"  Model {model_name} second pass complete: {retry_success}/{len(failed_rows)} succeeded")
    
    safe_print(f"\n{'='*60}")
    safe_print(f"Second-pass retries complete")
    safe_print(f"{'='*60}\n")
    
    # Evaluate all models.
    safe_print(f"\nStarting evaluation for all models...")
    df = pd.DataFrame(rows)
    
    evaluation_results = []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        
        result = evaluate_model_on_dataset(df, answer_col, model_name)
        evaluation_results.append(result)
        
        # Save detailed evaluation results.
        if result['total'] > 0:
            valid_df = result['valid_df']
            eval_csv_path = output_dir / f"evaluation_{clean_model_name}.csv"
            valid_df.to_csv(eval_csv_path, index=False)
        
        safe_print(f"  {model_name}: {result['overall_accuracy']:.2%} ({result['correct']}/{result['total']})")
    
    elapsed_time = time.time() - start_time
    safe_print(f"\n{csv_name} processing complete, elapsed time: {elapsed_time:.2f}s")
    
    return {
        'csv_name': csv_name,
        'total_rows': len(rows),
        'evaluation_results': evaluation_results,
        'elapsed_time': elapsed_time
    }


def test_model_connectivity(llms: Dict[str, ChatOpenAI]) -> Tuple[Dict[str, ChatOpenAI], List[str]]:
    """
    Test connectivity for all models and return the available subset.
    
    Args:
        llms: Model dictionary
    
    Returns:
        Tuple[Dict[str, ChatOpenAI], List[str]]: Available model dictionary and names
    """
    safe_print("Starting model connectivity test...")
    
    available_llms = {}
    available_models = []
    failed_models = []
    
    for model_name, llm in llms.items():
        safe_print(f"Testing model: {model_name}")
        
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
            
            safe_print(f"  ✓ Connection successful: {response_text[:50]}...")
            available_llms[model_name] = llm
            available_models.append(model_name)
            
        except Exception as e:
            safe_print(f"  ✗ Connection failed: {str(e)[:100]}...")
            safe_print(f"  ⚠️  This model will be skipped")
            failed_models.append(model_name)
    
    safe_print(f"\nConnectivity test complete:")
    safe_print(f"  ✓ Available models: {len(available_models)}")
    safe_print(f"  ✗ Failed models: {len(failed_models)}")
    
    if failed_models:
        safe_print(f"\nSkipped models:")
        for model in failed_models:
            safe_print(f"  - {model}")
    
    safe_print()
    
    return available_llms, available_models


def graceful_shutdown(signum, frame):
    """
    Graceful shutdown handler triggered by Ctrl+C.
    """
    safe_print("\n" + "="*60)
    safe_print("⚠️  Interrupt signal received (Ctrl+C)")
    safe_print("="*60)
    safe_print("✓ All completed data has already been saved to disk")
    safe_print("✓ The next run will resume from the interruption point")
    safe_print("="*60)
    safe_print("Exiting safely...")
    sys.exit(0)


def main():
    """Parse command-line arguments, process one CSV, and evaluate all models."""
    
    # Register signal handlers.
    signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # kill signal
    
    parser = argparse.ArgumentParser(
        description="Parallel multi-model evaluation script (supports resume support and API key rotation)"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="CSV file path to process"
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        required=True,
        help="CSV name used for output directories, for example: apl, arrowtrajectory, nfl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=10,
        help="Number of rows to process in parallel; models are handled serially within each row (default: 25)"
    )
    
    args = parser.parse_args()
    
    # Define the OpenRouter VLM models used in this run.
    model_names = [
        # === OpenRouter group ===
        
        # Qwen VL family (3) - only models available on OpenRouter
        'qwen/qwen2.5-vl-72b-instruct',
        'qwen/qwen2.5-vl-32b-instruct',
        'qwen/qwen-vl-max',
        
        # Anthropic Claude family (2)
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-3.7-sonnet',
        
        # Google Gemini family (2)
  
        'google/gemini-2.5-flash-image',
        'google/gemini-2.5-flash',
        
        # OpenAI family (3)
        'openai/gpt-4o',
        'openai/gpt-4o-mini',
        'openai/gpt-4-turbo',
        
        # Other vision-capable VLM models
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
    
    # Validate paths.
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    if not csv_path.exists():
        print(f"Error: CSV file does not exist: {csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Parallel multi-model evaluation script v2.0")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"CSV file: {csv_path}")
    print(f"CSV name: {csv_name}")
    print(f"Output directory: {output_base_dir}")
    print(f"\nModel configuration:")
    print(f"  - Total models: {len(model_names)}")
    print(f"  - OpenRouter: {sum(1 for m in model_names if get_model_provider(m) == 'openrouter')}")
    print(f"  - ChatAnywhere: {sum(1 for m in model_names if get_model_provider(m) == 'chatanywhere')}")
    print(f"\nConcurrency strategy:")
    print(f"  - Row processing: {args.parallel_rows} rows in parallel")
    print(f"  - Model processing: serial within each row")
    print(f"  - Maximum concurrent API requests: {args.parallel_rows}")
    print(f"\nFeatures:")
    print(f"  ✓ Real-time saving (write immediately after each row)")
    print(f"  ✓ Resume support after interruption")
    print(f"  ✓ Signal handling for safe Ctrl+C exit")
    print(f"  ✓ Multiple API key rotation")
    print(f"\nOther settings:")
    print(f"  - Retry strategy: two passes (2 attempts in the first pass, 2 in the second)")
    print(f"  - Timeout: 600 seconds")
    print()
    
    # Initialize all models.
    print(f"Initializing {len(model_names)} models...")
    llms = {}
    for model_name in model_names:
        print(f"  - {model_name}")
        try:
            llms[model_name] = create_llm(model_name)
        except ValueError as e:
            print(f"  Error: {e}")
            sys.exit(1)
    
    print(f"\nAll models initialized\n")
    
    # Test connectivity and continue with available models only.
    available_llms, available_models = test_model_connectivity(llms)
    
    if len(available_models) == 0:
        print("Error: no available models")
        sys.exit(1)
    
    print(f"Continuing with {len(available_models)} available models\n")
    
    # Process the CSV file.
    print(f"\n{'='*60}")
    print(f"Starting CSV processing: {csv_name}")
    print(f"{'='*60}\n")
    
    overall_start = time.time()
    
    try:
        result = process_single_csv(
            csv_name,
            csv_path,
            dataset_path,
            available_llms,  # Use only available models.
            available_models,  # Use only available model names.
            output_base_dir,
            args.parallel_rows
        )
    except Exception as e:
        print(f"Error: CSV processing failed: {e}")
        sys.exit(1)
    
    overall_elapsed = time.time() - overall_start
    
    # Print a summary.
    print(f"\n{'='*60}")
    print(f"Processing complete. Total elapsed time: {overall_elapsed:.2f}s")
    print(f"{'='*60}\n")
    
    # Print dataset statistics.
    print(f"\n{csv_name.upper()} dataset:")
    print(f"  Total entries: {result['total_rows']}")
    print(f"  Elapsed time: {result['elapsed_time']:.2f}s")
    print(f"  Accuracy by model:")
    
    # Sort by accuracy.
    sorted_results = sorted(result['evaluation_results'], 
                           key=lambda x: x['overall_accuracy'], 
                           reverse=True)
    
    for eval_result in sorted_results:
        acc_str = f"{eval_result['overall_accuracy']:.2%}"
        count_str = f"{eval_result['correct']}/{eval_result['total']}"
        print(f"    {eval_result['model']:<40} {acc_str:<10} ({count_str})")
        
        # Show per-category accuracy.
        if 'category_accuracy' in eval_result and eval_result['category_accuracy']:
            for category, stats in sorted(eval_result['category_accuracy'].items()):
                cat_acc_str = f"{stats['accuracy']:.2%}"
                cat_count_str = f"{stats['correct']}/{stats['total']}"
                print(f"      └─ {category:<35} {cat_acc_str:<10} ({cat_count_str})")
    
    print(f"\n{'='*60}")
    print(f"All results have been saved to: {output_base_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Starting test_parallel_multi_models.py")
    main()
