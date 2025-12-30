#!/usr/bin/env python3
"""
为悬挂法测物体重心模拟实验生成CSV
为每个视频生成2个问题：
1. 重心位置判断（center_of_gravity）
2. 旋转方向判断（rotation_direction）
"""

import os
import sys
import argparse
import csv
from pathlib import Path


def get_video_keyframe_pairs(experiment_path):
    """
    获取实验目录下的视频和关键帧配对（从variants_sim和key_frames_sim）。
    
    Args:
        experiment_path (Path): 实验目录路径
    
    Returns:
        list: (视频文件, 关键帧文件)元组列表，使用相对路径
    """
    variants_sim_path = experiment_path / "variants_sim"
    key_frames_sim_path = experiment_path / "key_frames_sim"
    
    if not variants_sim_path.exists() or not key_frames_sim_path.exists():
        return []
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'}
    
    # 获取所有视频文件
    video_files = {}
    for video_file in variants_sim_path.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # 跳过隐藏文件
            if not video_file.name.startswith('._'):
                video_files[video_file.stem] = video_file
    
    # 获取所有关键帧文件
    keyframe_files = {}
    for keyframe_file in key_frames_sim_path.iterdir():
        if keyframe_file.is_file() and keyframe_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            keyframe_files[keyframe_file.stem] = keyframe_file
    
    # 匹配视频和关键帧
    pairs = []
    for video_stem, video_file in sorted(video_files.items()):
        if video_stem in keyframe_files:
            keyframe_file = keyframe_files[video_stem]
            pairs.append((video_file, keyframe_file))
        else:
            print(f"警告: 视频 {video_file.name} 没有找到对应的关键帧")
    
    return pairs


def build_cg_sim_csv(dataset_path, output_csv_path, experiment_name="悬挂法测物体重心"):
    """
    为重心模拟实验构建CSV文件。
    
    Args:
        dataset_path (Path): 数据集根目录
        output_csv_path (Path): 输出CSV路径
        experiment_name (str): 实验名称
    """
    dataset_path = Path(dataset_path)
    experiment_path = dataset_path / experiment_name
    
    if not experiment_path.exists():
        print(f"错误: 实验目录不存在: {experiment_path}")
        sys.exit(1)
    
    # 获取视频-关键帧配对
    pairs = get_video_keyframe_pairs(experiment_path)
    
    if not pairs:
        print(f"错误: 在 {experiment_name} 中没有找到视频-关键帧配对")
        sys.exit(1)
    
    print(f"找到 {len(pairs)} 个视频-关键帧配对")
    
    # 准备CSV数据
    rows = []
    
    for video_file, keyframe_file in pairs:
        # 使用相对于数据集根目录的路径
        video_rel_path = video_file.relative_to(dataset_path)
        keyframe_rel_path = keyframe_file.relative_to(dataset_path)
        
        # 为每个视频创建2行（2个问题）
        question_categories = [
            ('center_of_gravity', '重心位置'),
            ('rotation_direction', '旋转方向')
        ]
        
        for category, category_cn in question_categories:
            row = {
                'experiment_name': experiment_name,
                'video_path': str(video_rel_path),
                'key_frame_path': str(keyframe_rel_path),
                'question_type': 'single_choice',
                'question_category': category,
                'question': '',  # 将由问题生成脚本填充
                'ground truth': ''  # 用户手动填写
            }
            rows.append(row)
    
    # 写入CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['experiment_name', 'video_path', 'key_frame_path', 
                     'question_type', 'question_category', 'question', 'ground truth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*60}")
    print(f"CSV文件已创建: {output_csv_path}")
    print(f"总行数: {len(rows)}")
    print(f"  - 视频数: {len(pairs)}")
    print(f"  - 每个视频2个问题（重心位置 + 旋转方向）")
    print(f"{'='*60}")


def main():
    """主函数：处理命令行参数并构建CSV。"""
    parser = argparse.ArgumentParser(
        description="为悬挂法测物体重心模拟实验构建CSV"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集根目录路径（Dataset_V2）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/PhysBench/output/apl/dataset_cg_sim.csv",
        help="输出CSV文件路径 (默认: /root/PhysBench/output/apl/dataset_cg_sim.csv)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    output_csv_path = Path(args.output)
    
    # 验证路径
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    print(f"悬挂法测物体重心 - CSV构建器")
    print(f"数据集路径: {dataset_path}")
    print(f"输出CSV: {output_csv_path}")
    print()
    
    build_cg_sim_csv(dataset_path, output_csv_path)


if __name__ == "__main__":
    main()



