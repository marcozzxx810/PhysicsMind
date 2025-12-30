#!/usr/bin/env python3
"""
为悬挂法测物体重心模拟实验生成单选题。
使用多模态LLM分析视频关键帧，生成：
1. 重心位置选择题（靠上/靠下/中间/不在物体内）
2. 旋转方向选择题（顺时针/逆时针/不旋转/摆动）
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


# 重心位置问题的提示词模板
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


# 旋转方向问题的提示词模板
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
    将图片编码为base64字符串，并返回图片的MIME类型。
    
    Args:
        image_path (Path): 图片文件路径
    
    Returns:
        Tuple[str, str]: (Base64编码的图片字符串, MIME类型)
    """
    # 根据文件扩展名确定MIME类型
    ext = image_path.suffix.lower()
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    mime_type = mime_type_map.get(ext, 'image/jpeg')  # 默认使用jpeg
    
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
    使用多模态LLM生成问题和答案。
    
    Args:
        llm (ChatOpenAI): LLM实例
        prompt_template (str): 提示词模板
        image_path (Path): 关键帧图片路径
        experiment_name (str): 实验名称
    
    Returns:
        Tuple[str, str]: (生成的问题, 生成的答案)
    """
    try:
        # 编码图片为base64并获取MIME类型
        base64_image, mime_type = encode_image_to_base64(image_path)
        
        # 格式化提示词
        formatted_prompt = prompt_template.format(experiment_name=experiment_name)
        
        # 创建包含图片的消息
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
        
        # 生成回复
        response = llm.invoke([message])
        response_text = response.content.strip()
        
        # 检查是否为空响应
        if not response_text or len(response_text) < 10:
            print(f"  错误: 模型返回了空响应或过短的响应")
            return "[ERROR: Empty Response]", "[ERROR]"
        
        # 解析回复 - 格式为 "Question: ... Answer: ..."
        if "Question:" in response_text and "Answer:" in response_text:
            parts = response_text.split("Answer:")
            question_part = parts[0].replace("Question:", "").strip()
            answer_part = parts[1].strip()
            
            # 保持问题的多行格式（包含选项），只在最后清理
            question = question_part
            answer = answer_part
        else:
            # Fallback 解析
            lines = response_text.strip().split('\n')
            question = lines[0] if lines else "[Generated Question]"
            answer = lines[1] if len(lines) > 1 else "[Generated Answer]"
        
        # 提取单个字母答案 (A/B/C/D)
        answer_upper = answer.upper()
        extracted_answer = None
        
        for char in ['A', 'B', 'C', 'D']:
            if char in answer_upper:
                extracted_answer = char
                break
        
        # 清理问题 - 将换行符替换为空格，使其成为单行
        question = question.replace('\n', ' ').strip()
        
        if extracted_answer:
            return question, extracted_answer
        else:
            print(f"  警告: 无法从回复中提取答案字母")
            print(f"  原始答案: {answer}")
            return question, "[PARSE_ERROR]"
            
    except Exception as e:
        print(f"  错误: 生成问题时出错: {e}")
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
    处理CSV数据集，为每个条目生成问题和答案。
    
    Args:
        input_csv_path (Path): 输入CSV路径
        output_csv_path (Path): 输出CSV路径
        dataset_root (Path): 数据集根目录
        model_name (str): 使用的模型名称
        max_retries (int): 最大重试次数
        retry_delay (int): 重试延迟（秒）
    """
    # 初始化LLM
    print(f"初始化模型: {model_name}")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("错误: 未设置 OPENROUTER_API_KEY 环境变量")
        print("请运行: export OPENROUTER_API_KEY='your-api-key'")
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
    
    # 读取输入CSV
    rows = []
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"从 {input_csv_path} 读取了 {len(rows)} 个条目")
    
    # 处理每一行
    processed_count = 0
    error_count = 0
    
    for idx, row in enumerate(rows, 1):
        experiment_name = row['experiment_name']
        video_path = row['video_path']
        keyframe_path = row['key_frame_path']
        question_category = row['question_category']
        
        print(f"\n[{idx}/{len(rows)}] 处理: {video_path}")
        print(f"  类别: {question_category}")
        
        # 构建完整的图片路径
        full_image_path = dataset_root / keyframe_path
        
        if not full_image_path.exists():
            print(f"  错误: 图片不存在: {full_image_path}")
            error_count += 1
            continue
        
        # 根据问题类别选择提示词
        if question_category == 'center_of_gravity':
            prompt_template = CENTER_OF_GRAVITY_PROMPT
        elif question_category == 'rotation_direction':
            prompt_template = ROTATION_DIRECTION_PROMPT
        else:
            print(f"  警告: 未知的问题类别: {question_category}")
            continue
        
        # 生成问题和答案（带重试机制）
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
                    # 注意：不要填充ground truth，留空给用户手动填写
                    # row['ground truth'] 保持为空
                    processed_count += 1
                    success = True
                    print(f"  ✓ 问题生成成功")
                    print(f"  生成的答案（仅供参考）: {answer}")
                    print(f"  ⚠️  ground truth 留空，需要手动验证填写")
                    break
                else:
                    raise Exception("生成返回错误标记")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  ⚠ 尝试 {attempt + 1} 失败: {e}")
                    print(f"  等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    print(f"  ✗ 所有尝试都失败了")
                    error_count += 1
        
        if not success:
            print(f"  跳过此条目")
    
    # 写入输出CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        if rows:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\n" + "="*60)
    print(f"处理完成！")
    print(f"成功生成: {processed_count} 个问题")
    print(f"失败/错误: {error_count} 个")
    print(f"输出文件: {output_csv_path}")
    print(f"\n⚠️  重要提示:")
    print(f"  - 所有问题已生成，但 ground truth 列为空")
    print(f"  - 请手动检查每个问题并填写正确答案")
    print(f"  - 生成的答案仅供参考，不一定正确")
    print("="*60)


def main():
    """主函数：处理命令行参数并生成问题。"""
    parser = argparse.ArgumentParser(
        description="为悬挂法测物体重心模拟实验生成单选题"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集根目录路径（Dataset_V2）"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="输入CSV文件路径（由build_csv_cg_sim.py生成）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/PhysBench/output/apl/dataset_cg_sim_with_questions.csv",
        help="输出CSV文件路径 (默认: /root/PhysBench/output/apl/dataset_cg_sim_with_questions.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="使用的模型名称 (默认: openai/gpt-4o)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="API调用失败时的最大重试次数 (默认: 2)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=1,
        help="重试之间的延迟秒数 (默认: 1)"
    )
    
    args = parser.parse_args()
    
    # 检查环境变量
    if not os.getenv("OPENROUTER_API_KEY"):
        print("错误: 未设置 OPENROUTER_API_KEY 环境变量")
        print("请运行: export OPENROUTER_API_KEY='your-api-key'")
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
    
    print(f"悬挂法测物体重心 - 问题生成器")
    print(f"数据集路径: {dataset_path}")
    print(f"输入CSV: {input_csv_path}")
    print(f"输出CSV: {output_csv_path}")
    print(f"模型: {args.model}")
    print(f"最大重试次数: {args.max_retries}")
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



