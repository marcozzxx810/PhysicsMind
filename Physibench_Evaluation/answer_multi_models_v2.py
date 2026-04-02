#!/usr/bin/env python3
"""
Multi-model answer generation script V2.0.
Reads existing questions and images, uses multiple VLM models to answer them,
and merges all answers into a single CSV file.

New features:
- Parallel row processing (configurable concurrency)
- Multiple API key rotation (prevents a single key from being rate-limited)
- Incremental saving and resume support
- Signal handling (graceful exit on Ctrl+C)
- Thread-safe file operations
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


# Thread-safe print lock and file lock
print_lock = threading.Lock()
file_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print helper."""
    with print_lock:
        print(*args, **kwargs)


def get_random_api_key(provider: str) -> str:
    """
    Get a random API key when multiple keys are configured.
    
    Args:
        provider: 'openrouter', 'chatanywhere', or 'siliconflow'
    
    Returns:
        API key
    """
    if provider == 'openrouter':
        keys_str = os.getenv("OPENROUTER_API_KEY", "")
        if not keys_str:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    elif provider == 'siliconflow':
        keys_str = os.getenv("SILICONFLOW_API_KEY", "")
        if not keys_str:
            raise ValueError("SILICONFLOW_API_KEY environment variable is not set")
        keys = [k.strip() for k in keys_str.split(';') if k.strip()]
        return random.choice(keys)
    else:  # chatanywhere
        keys_str = os.getenv("CHATANYWHERE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not keys_str:
            raise ValueError("CHATANYWHERE_API_KEY or OPENAI_API_KEY environment variable is not set")
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


def get_llm_config(model_url: str, model_name: str) -> Dict:
    """
    Get the API configuration for the requested model.
    
    Args:
        model_url (str): API type ('openrouter' or 'chatanywhere')
        model_name (str): Model name, used for special-case routing
    
    Returns:
        Dict: Configuration dictionary containing `api_key` and `base_url`
    """
    # Special case: route DeepSeek VLM models through SiliconFlow.
    if 'deepseek' in model_name.lower() and 'vl' in model_name.lower():
        return {
            'api_key': get_random_api_key('siliconflow'),
            'base_url': 'https://api.siliconflow.cn/v1/',
        }
    
    # Use the OpenRouter API.
    if model_url == 'openrouter':
        return {
            'api_key': get_random_api_key('openrouter'),
            'base_url': 'https://openrouter.ai/api/v1',
        }
    # Use the ChatAnywhere API.
    else:
        return {
            'api_key': get_random_api_key('chatanywhere'),
            'base_url': 'https://api.chatanywhere.org/v1',
        }


def get_max_tokens(model_name: str) -> int:
    """
    Determine the `max_tokens` value for a model.
    
    Args:
        model_name (str): Model name
    
    Returns:
        int: `max_tokens` value
    """
    if 'deepseek' in model_name.lower():
        return 20000
    else:
        return 10000


def create_llm(model_url: str, model_name: str) -> ChatOpenAI:
    """
    Create an LLM instance.
    
    Args:
        model_url (str): API type ('openrouter' or 'chatanywhere')
        model_name (str): Model name
    
    Returns:
        ChatOpenAI: LLM instance
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
    Answer an existing question with a multimodal LLM.
    
    Args:
        llm (ChatOpenAI): LLM instance
        question (str): Question to answer
        image_path (Path): Keyframe image path
        model_name (str): Model name, used to determine compression rules
    
    Returns:
        str: Generated answer
    """
    start_time = time.time()
    try:
        # Determine whether the image needs compression.
        need_compress = 'claude' in model_name.lower() or 'anthropic' in model_name.lower() or 'deepseek' in model_name.lower()
        
        # Encode the image as base64.
        if need_compress:
            base64_image = encode_image_to_base64(image_path, max_size_mb=4.5)
        else:
            # Other models can use the original image directly.
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Build the prompt and ask for a direct answer.
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
        elapsed_time = time.time() - start_time
        
        # Debug logging can be enabled here if needed.
        # safe_print(f"  DEBUG: response object: {response}")
        
        # Try to log token usage information.
        if hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            safe_print(f"  📊 Token usage: {metadata.get('token_usage', 'N/A')}")
            safe_print(f"  📝 Model: {metadata.get('model_name', 'N/A')}")
        
        safe_print(f"  ⏱️  Elapsed time: {elapsed_time:.2f} seconds")
        response_text = response.content.strip()
        
        # Extract the answer letter.
        answer = response_text.upper()
        
        # Normalize the answer text.
        if "Answer:" in answer or "ANSWER:" in answer:
            answer = answer.split(":")[-1].strip()
        
        # safe_print(f"  DEBUG: cleaned answer: {answer}")
        
        # Extract a single answer letter.
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
        safe_print(f"  Error: failed to answer the question: {e}")
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
    Attempt to answer a question and retry automatically on failure.
    
    Args:
        llm: LLM instance
        question: Question text
        full_image_path: Image path
        max_retries: Maximum number of retries
        retry_delay: Retry delay in seconds
        model_name: Model name, used to determine compression rules
    
    Returns:
        (success flag, answer)
    """
    for attempt in range(max_retries):
        try:
            answer = answer_question(llm, question, full_image_path, model_name)
            
            if answer not in ["[ERROR]", "[PARSE_ERROR]"]:
                if attempt > 0:
                    safe_print(f"  ✓ Retry succeeded on attempt {attempt + 1}")
                return True, answer
            else:
                raise Exception("Model returned an error marker")
                
        except Exception as e:
            if attempt < max_retries - 1:
                safe_print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:80]}")
                safe_print(f"  Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                safe_print(f"  ✗ All attempts failed")
    
    return False, "[ERROR]"


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
    max_retries: int,
    retry_delay: int
) -> dict:
    """
    Process one CSV row by answering the question with all models.
    
    Args:
        row: CSV row data
        llms: Mapping from model name to LLM instance
        model_names: List of model names
        dataset_root: Dataset root directory
        row_idx: Row index
        total_rows: Total number of rows
        max_retries: Maximum number of retries
        retry_delay: Retry delay
    
    Returns:
        Updated row data
    """
    keyframe_path = row['key_frame_path']
    question = row['question']
    
    safe_print(f"\n[{row_idx}/{total_rows}] Processing: {keyframe_path}")
    
    # Skip rows without a valid question.
    if not question or question.startswith('[') or len(question) < 10:
        safe_print(f"  Skipping: no valid question")
        return row
    
    # Build the full image path.
    full_image_path = dataset_root / keyframe_path
    
    if not full_image_path.exists():
        safe_print(f"  Error: image not found: {full_image_path}")
        return row
    
    safe_print(f"  Question: {question[:80]}...")
    
    # Query each model in sequence.
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        safe_print(f"\n  Model: {model_name}")
        
        # Answer the question with retry support.
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
            safe_print(f"  ✓ Answer: {answer}")
        else:
            row[f'answer_{clean_model_name}'] = "[ERROR]"
            safe_print(f"  Skipping this model's answer")
    
    return row


def test_model_connectivity(llms: Dict[str, ChatOpenAI]) -> bool:
    """
    Test connectivity for all configured models.
    
    Args:
        llms (Dict[str, ChatOpenAI]): Mapping from model name to LLM instance
    
    Returns:
        bool: `True` if every model connects successfully, otherwise `False`
    """
    safe_print("Starting model connectivity test...")

    all_success = True
    results = []
    
    for model_name, llm in llms.items():
        safe_print(f"\nTesting model: {model_name}")
        safe_print(f"  Connecting...")
        
        try:
            # Send a simple test message.
            test_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Hello, this is a connectivity test. Please respond with 'OK'.",
                    }
                ]
            )
            
            # Invoke the model.
            response = llm.invoke([test_message])
            print(f"DEBUG: response object: {response}")
            response_text = response.content.strip()
            
            safe_print(f"  ✓ Connection successful")
            safe_print(f"  Response: {response_text[:50]}...")
            
            results.append({
                'model': model_name,
                'success': True,
            })
            
        except Exception as e:
            safe_print(f"  Error: {str(e)[:100]}...")
            all_success = False
            
            results.append({
                'model': model_name,
                'success': False,
            })
    
    # Display summary statistics.
    safe_print("\n" + "="*60)
    safe_print("Connectivity test summary:")
    
    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failure"
    
    if all_success:
        safe_print("✓ All model connectivity tests passed")
    else:
        safe_print("✗ Some model connectivity tests failed. Please check the configuration")
    safe_print("="*60 + "\n")
    
    return all_success


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
    Process a CSV dataset and generate answers for each existing question with
    multiple models.
    
    Args:
        input_csv_path (Path): Input CSV path containing questions and ground truth
        output_csv_path (Path): Output CSV path
        dataset_root (Path): Dataset root directory
        model_names (List[str]): List of model names to use
        model_url (str): API type ('openrouter' or 'chatanywhere')
        max_retries (int): Maximum number of retries
        retry_delay (int): Retry delay in seconds
        parallel_rows (int): Number of rows to process in parallel
    """
    # Initialize LLM instances for all models.
    safe_print(f"Initializing {len(model_names)} models...")
    llms = {}
    for model_name in model_names:
        safe_print(f"  - {model_name}")
        try:
            llms[model_name] = create_llm(model_url, model_name)
        except ValueError as e:
            safe_print(f"  Error: {e}")
            sys.exit(1)
    
    safe_print(f"\nAll models initialized\n")
    
    # Test model connectivity.
    if not test_model_connectivity(llms):
        safe_print("\nError: model connectivity test failed, aborting")
        sys.exit(1)
    
    # Read the input CSV.
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    safe_print(f"Read {len(rows)} entries from {input_csv_path}")
    
    # Load progress from any previous run.
    existing_progress = load_existing_progress(output_csv_path)
    safe_print(f"Loaded {len(existing_progress)} completed entries from previous progress")
    
    # Initialize answer columns for each model.
    fieldnames = list(rows[0].keys()) if rows else []
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        answer_col = f'answer_{clean_model_name}'
        if answer_col not in fieldnames:
            fieldnames.append(answer_col)
    
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
        safe_print(f"All entries are already complete; skipping the first pass")
        rows = rows_completed
    else:
        # Process rows in parallel with incremental saves.
        safe_print(f"\nStarting first pass (parallel rows: {parallel_rows})...")
        
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
                    # Save progress incrementally.
                    save_row_incremental(result, output_csv_path, fieldnames)
                    safe_print(f"[{idx}/{len(rows_to_process)}] ✓ Saved")
                except Exception as e:
                    safe_print(f"Row {idx} failed: {e}")
                    processed_rows[idx - 1] = rows_to_process[idx - 1]
        
        # Merge all results.
        rows = rows_completed + processed_rows
    
    safe_print(f"\n{'='*60}")
    safe_print(f"First pass complete")
    safe_print(f"{'='*60}")
    
    # First-pass statistics.
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
    
    safe_print(f"Skipped (no question): {skipped_count}")
    for model_name in model_names:
        safe_print(f"\nModel {model_name}:")
        safe_print(f"  Successful answers: {processed_count[model_name]}")
        safe_print(f"  Failed/errors: {error_count[model_name]}")
    safe_print("="*60)
    
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
            retry_success_count = 0
            
            for retry_idx, (row_idx, row) in enumerate(zip(failed_indices, failed_rows), 1):
                keyframe_path = row['key_frame_path']
                question = row['question']
                
                safe_print(f"\n[Retry {retry_idx}/{len(failed_rows)}] {keyframe_path}")
                safe_print(f"  Question: {question[:80]}...")
                
                # Build the full image path.
                full_image_path = dataset_root / keyframe_path
                
                if not full_image_path.exists():
                    safe_print(f"  Error: image not found: {full_image_path}")
                    continue
                
                # Retry up to five times during the second pass.
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
                    safe_print(f"  ✓ Answer: {answer}")
                    # Update the progress file.
                    save_row_incremental(rows[row_idx], output_csv_path, fieldnames)
                else:
                    safe_print(f"  Final failure; skipping this entry")
            
            safe_print(f"\n" + "="*60)
            safe_print(f"Retry pass complete for model {model_name}")
            safe_print(f"Successful answers: {retry_success_count}")
            safe_print(f"Still failing: {len(failed_rows) - retry_success_count}")
            safe_print("="*60)
            
            # Update total counts.
            processed_count[model_name] += retry_success_count
    
    # Rewrite the final CSV to ensure the output is complete.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    safe_print(f"\n" + "="*60)
    safe_print(f"All processing complete")
    safe_print(f"Skipped (no question): {skipped_count}")
    for model_name in model_names:
        clean_model_name = model_name.replace('-', '_').replace('.', '_').replace('/', '_')
        failed_rows_count = sum(1 for row in rows if not row.get(f'answer_{clean_model_name}') or row.get(f'answer_{clean_model_name}') in ['[ERROR]', '[PARSE_ERROR]'])
        safe_print(f"\nModel {model_name}:")
        safe_print(f"  Total successful answers: {processed_count[model_name]}")
        safe_print(f"  Final failures: {failed_rows_count}")
    safe_print(f"\nOutput file: {output_csv_path}")
    safe_print("="*60)


def main():
    """Parse command-line arguments and answer questions."""
    
    # Register signal handlers.
    signal.signal(signal.SIGINT, graceful_shutdown)   # Ctrl+C
    signal.signal(signal.SIGTERM, graceful_shutdown)  # kill signal
    
    parser = argparse.ArgumentParser(
        description="Answer existing questions with multiple VLM models V2.0 (supports parallel processing, resume support, and API key rotation)"
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
        help="Input CSV file path containing questions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/dataset_multi_answered.csv",
        help="Output CSV file path (default: output/dataset_multi_answered.csv)"
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of model names (for example: gpt-4o,gpt-4o-mini,deepseek-chat)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for first-pass API failures (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=1,
        help="Delay between retries in seconds (default: 1)"
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default='chatanywhere',
        help="Model API provider (default: chatanywhere)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=5,
        help="Number of rows to process in parallel (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Parse the model list.
    model_names = [m.strip() for m in args.models.split(',')]
    
    if not model_names:
        print("Error: at least one model must be specified")
        sys.exit(1)
    
    dataset_path = Path(args.dataset_path)
    input_csv_path = Path(args.csv)
    output_csv_path = Path(args.output)
    
    # Validate paths.
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    if not input_csv_path.exists():
        print(f"Error: input CSV file does not exist: {input_csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Multi-model VLM question answerer V2.0")
    print(f"{'='*60}")
    print(f"Dataset path: {dataset_path}")
    print(f"Input CSV: {input_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Model list: {', '.join(model_names)}")
    print(f"Model provider: {args.model_url}")
    print(f"First-pass max retries: {args.max_retries}")
    print(f"Parallel rows: {args.parallel_rows}")
    print(f"\nFeatures:")
    print(f"  ✓ Parallel row processing ({args.parallel_rows} rows)")
    print(f"  ✓ Real-time saving (write immediately after each row)")
    print(f"  ✓ Resume support after interruption")
    print(f"  ✓ Signal handling for safe Ctrl+C exit")
    print(f"  ✓ Multiple API key rotation")
    print(f"  ✓ Two retry passes ({args.max_retries} first pass, 5 second pass)")
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
