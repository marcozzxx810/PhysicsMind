#!/usr/bin/env python3
"""
Generate multiple-choice questions for the simulated suspended center-of-gravity
experiment.
Uses a multimodal LLM to analyze video keyframes and generate:
1. Center-of-gravity position questions
2. Rotation-direction questions
"""

import os
import sys
import argparse
import csv
import base64
from pathlib import Path
from typing import Tuple
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# Prompt template for center-of-gravity position questions
CENTER_OF_GRAVITY_PROMPT = """You are a physics experiment analysis expert. Please carefully observe this image, which is the first frame from the "{experiment_name}" experiment.

**Task**: Generate a single-choice question about the center of gravity position of the object in the image.

**Requirements**:
1. Carefully observe the object's shape and mass distribution in the image
2. Determine where the center of gravity is located relative to the object
3. Generate a clear multiple-choice question with 4 options (A/B/C/D)
4. Options should be:
   - A) Upper part of the object (above geometric center)
   - B) Lower part of the object (below geometric center)
   - C) Center of the object (at geometric center)
   - D) Outside the object (not within the object boundary)
5. The question must **only be answerable by observing the image**, not by common sense guessing
6. All content must be in English
7. Format your response as: "Question: [question text] A) [option A] B) [option B] C) [option C] D) [option D] Answer: [correct letter]"

**Example**:
Question: In the image, where is the center of gravity of the object located?
A) Upper part of the object (above geometric center)
B) Lower part of the object (below geometric center)
C) Center of the object (at geometric center)
D) Outside the object (not within the object boundary)
Answer: B

Now please generate the question:
"""


# Prompt template for rotation-direction questions
ROTATION_DIRECTION_PROMPT = """You are a physics experiment analysis expert. Please carefully observe this image, which is the first frame from the "{experiment_name}" experiment.

**Task**: Generate a detailed single-choice question about how the object will rotate when released.

**Requirements**:
1. Carefully observe the object's shape, center of gravity, and current suspended position in the image
2. Determine how the object will rotate when released
3. Generate a clear multiple-choice question with 4 options (A/B/C/D)
4. Options must include: Clockwise rotation, Counterclockwise rotation, No rotation (remain stationary), and Oscillate without net rotation
5. The question must **describe the object's current state in detail** and ask about the rotation direction
6. All content must be in English
7. Format your response as: "Question: [question text] A) [option A] B) [option B] C) [option C] D) [option D] Answer: [correct letter]"

**Example**:
Question: In the image, when the suspended object is released, in which direction will the object rotate?
A) Clockwise rotation
B) Counterclockwise (anticlockwise) rotation
C) No rotation (remain stationary)
D) Oscillate without net rotation
Answer: A

Now please generate the question:
"""


def encode_image_to_base64(image_path: Path) -> Tuple[str, str]:
    """
    Encode an image as base64 and return its MIME type.
    
    Args:
        image_path (Path): Image file path
    
    Returns:
        Tuple[str, str]: Base64 image string and MIME type
    """
    # Determine the MIME type from the file extension.
    ext = image_path.suffix.lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_type_map.get(ext, 'image/jpeg')  # Default to JPEG.
    
    with open(image_path, 'rb') as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    return base64_data, mime_type


def generate_question_answer(
    llm: ChatOpenAI,
    prompt_template: str,
    image_path: Path,
    experiment_name: str
) -> Tuple[str, str]:
    """
    Generate a question and answer with a multimodal LLM.
    
    Args:
        llm (ChatOpenAI): LLM instance
        prompt_template (str): Prompt template
        image_path (Path): Keyframe image path
        experiment_name (str): Experiment name
    
    Returns:
        Tuple[str, str]: Generated question and answer
    """
    try:
        # Encode the image as base64 and get its MIME type.
        base64_image, mime_type = encode_image_to_base64(image_path)
        
        # Format the prompt.
        formatted_prompt = prompt_template.format(experiment_name=experiment_name)
        
        # Create a message that includes the image.
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": formatted_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                },
            ],
        )
        
        # Generate the response.
        response = llm.invoke([message])
        response_text = response.content.strip()
        
        # Check for an empty or unusably short response.
        if not response_text or len(response_text) < 10:
            print(f"  Error: model returned an empty or too-short response")
            return "[ERROR: Empty Response]", "[ERROR]"
        
        # Parse responses of the form "Question: ... Answer: ...".
        if "Question:" in response_text and "Answer:" in response_text:
            parts = response_text.split("Answer:")
            question_part = parts[0].replace("Question:", "").strip()
            answer_part = parts[1].strip()
            
            # Preserve the question format until final cleanup.
            question = question_part
            answer = answer_part
        else:
            # Fallback parsing.
            lines = response_text.strip().split('\n')
            question = lines[0] if lines else "[Generated Question]"
            answer = lines[1] if len(lines) > 1 else "[Generated Answer]"
        
        # Extract a single answer letter (A/B/C/D).
        answer_upper = answer.upper()
        extracted_answer = None
        
        for char in ['A', 'B', 'C', 'D']:
            if char in answer_upper:
                extracted_answer = char
                break
        
        # Normalize the question to a single line.
        question = question.replace('\n', ' ').strip()
        
        if extracted_answer:
            return question, extracted_answer
        else:
            print(f"  Warning: could not extract an answer letter from the response")
            print(f"  Raw answer: {answer}")
            return question, "[PARSE_ERROR]"
            
    except Exception as e:
        print(f"  Error: failed to generate the question: {e}")
        return "[ERROR]", "[ERROR]"


def process_dataset(
    input_csv_path: Path,
    output_csv_path: Path,
    dataset_root: Path,
    model_name: str = "openai/gpt-4o",
    max_retries: int = 2,
    retry_delay: int = 1
):
    """
    Process a CSV dataset and generate a question and answer for each entry.
    
    Args:
        input_csv_path (Path): Input CSV path
        output_csv_path (Path): Output CSV path
        dataset_root (Path): Dataset root directory
        model_name (str): Model name to use
        max_retries (int): Maximum number of retries
        retry_delay (int): Retry delay in seconds
    """
    # Initialize the LLM.
    print(f"Initializing model: {model_name}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Run: export OPENROUTER_API_KEY='your-api-key'")
        sys.exit(1)
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        max_tokens=10000,
        temperature=0.3,
        timeout=600,
        default_headers={
            "HTTP-Referer": "https://github.com/ttz-cn/PhysbenchScript-boyi-guanyu-",
            "X-Title": "PhysBench CG Question Generator",
        }
    )
    
    # Read the input CSV.
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Read {len(rows)} entries from {input_csv_path}")
    
    # Process each row.
    processed_count = 0
    error_count = 0
    
    for idx, row in enumerate(rows, 1):
        experiment_name = row['experiment_name']
        video_path = row['video_path']
        keyframe_path = row['key_frame_path']
        question_category = row['question_category']
        
        print(f"\n[{idx}/{len(rows)}] Processing: {video_path}")
        print(f"  Category: {question_category}")
        
        # Build the full image path.
        full_image_path = dataset_root / keyframe_path
        
        if not full_image_path.exists():
            print(f"  Error: image not found: {full_image_path}")
            error_count += 1
            continue
        
        # Select the prompt template based on the question category.
        if question_category == 'center_of_gravity':
            prompt_template = CENTER_OF_GRAVITY_PROMPT
        elif question_category == 'rotation_direction':
            prompt_template = ROTATION_DIRECTION_PROMPT
        else:
            print(f"  Warning: unknown question category: {question_category}")
            continue
        
        # Generate the question and answer with retry support.
        success = False
        for attempt in range(max_retries):
            try:
                question, answer = generate_question_answer(
                    llm,
                    prompt_template,
                    full_image_path,
                    experiment_name
                )
                
                if question != "[ERROR]" and answer != "[ERROR]" and answer != "[PARSE_ERROR]":
                    row['question'] = question
                    # Leave `ground truth` empty for manual verification.
                    processed_count += 1
                    success = True
                    print(f"  ✓ Question generated successfully")
                    print(f"  Generated answer (for reference only): {answer}")
                    print(f"  ⚠️  The `ground truth` field is intentionally left blank for manual verification")
                    break
                else:
                    raise Exception("Model returned an error marker")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠ Attempt {attempt + 1} failed: {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"  ✗ All attempts failed")
                    error_count += 1
        
        if not success:
            print(f"  Skipping this entry")
    
    # Write the output CSV.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\n" + "="*60)
    print(f"Processing complete")
    print(f"Questions generated successfully: {processed_count}")
    print(f"Failures/errors: {error_count}")
    print(f"Output file: {output_csv_path}")
    print(f"\n⚠️  Important:")
    print(f"  - All questions have been generated, but the `ground truth` column is still empty")
    print(f"  - Manually review each question and fill in the correct answer")
    print(f"  - The generated answer is only a reference and may be incorrect")
    print("="*60)


def main():
    """Parse command-line arguments and generate questions."""
    parser = argparse.ArgumentParser(
        description="Generate multiple-choice questions for the simulated suspended center-of-gravity experiment"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory (Dataset_V2)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV file path generated by build_csv_cg_sim.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/PhysBench/output/apl/dataset_cg_sim_with_questions.csv",
        help="Output CSV file path (default: /root/PhysBench/output/apl/dataset_cg_sim_with_questions.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name to use (default: openai/gpt-4o)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum number of retries for API failures (default: 2)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=1,
        help="Delay between retries in seconds (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Check required environment variables.
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set")
        print("Run: export OPENROUTER_API_KEY='your-api-key'")
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
    
    print(f"Suspended center-of-gravity question generator")
    print(f"Dataset path: {dataset_path}")
    print(f"Input CSV: {input_csv_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Model: {args.model}")
    print(f"Max retries: {args.max_retries}")
    print()
    
    process_dataset(
        input_csv_path,
        output_csv_path,
        dataset_path,
        args.model,
        args.max_retries,
        args.retry_delay
    )


if __name__ == "__main__":
    main()


