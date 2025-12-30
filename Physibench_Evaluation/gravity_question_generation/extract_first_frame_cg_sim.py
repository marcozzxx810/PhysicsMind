#!/usr/bin/env python3
"""
从悬挂法测物体重心的模拟视频中提取第一帧。
处理 variants_sim 文件夹中的视频，保存帧到 key_frames_sim。
"""

import os
import sys
import argparse
from pathlib import Path
import cv2


def extract_first_frame(video_path, output_path):
    """
    从视频文件中提取第一帧。
    
    Args:
        video_path (Path): 输入视频文件路径
        output_path (Path): 保存提取帧的路径
    
    Returns:
        bool: 成功返回True，否则返回False
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return False
        
        # 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"错误: 视频中没有帧 {video_path}")
            cap.release()
            return False
        
        # 设置到第一帧（索引0）
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # 读取第一帧
        ret, frame = cap.read()
        
        if not ret:
            print(f"错误: 无法从视频读取帧 {video_path}")
            cap.release()
            return False
        
        # 保存帧
        success = cv2.imwrite(str(output_path), frame)
        
        if success:
            print(f"提取第一帧 (0/{total_frames}) 从 {video_path.name} -> {output_path.name}")
        else:
            print(f"错误: 无法保存帧到 {output_path}")
        
        cap.release()
        return success
        
    except Exception as e:
        print(f"处理 {video_path} 时出错: {str(e)}")
        return False


def process_cg_sim_experiment(dataset_path):
    """
    处理悬挂法测物体重心实验的variants_sim文件夹中的所有视频。
    
    Args:
        dataset_path (Path): 数据集根目录路径
    
    Returns:
        int: 成功处理的视频数量
    """
    dataset_path = Path(dataset_path)
    experiment_path = dataset_path / "悬挂法测物体重心"
    variants_sim_path = experiment_path / "variants_sim"
    
    # 检查variants_sim文件夹是否存在
    if not variants_sim_path.exists():
        print(f"错误: variants_sim文件夹不存在: {variants_sim_path}")
        return 0
    
    # 创建key_frames_sim输出目录
    key_frames_sim_path = experiment_path / "key_frames_sim"
    key_frames_sim_path.mkdir(exist_ok=True)
    print(f"创建输出目录: {key_frames_sim_path}")
    
    # 支持的视频扩展名
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'}
    
    # 查找所有视频文件
    video_files = []
    for file_path in variants_sim_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            # 跳过隐藏文件（以._开头的）
            if not file_path.name.startswith('._'):
                video_files.append(file_path)
    
    # 按文件名排序
    video_files.sort()
    
    if not video_files:
        print(f"在 {variants_sim_path} 中没有找到视频文件")
        return 0
    
    print(f"在 {variants_sim_path} 中找到 {len(video_files)} 个视频文件")
    
    successful_count = 0
    
    # 处理每个视频文件
    for video_file in video_files:
        # 创建输出文件名（将视频扩展名替换为.jpg）
        output_filename = video_file.stem + ".jpg"
        output_path = key_frames_sim_path / output_filename
        
        # 提取第一帧
        if extract_first_frame(video_file, output_path):
            successful_count += 1
    
    print(f"\n成功处理 {successful_count}/{len(video_files)} 个视频")
    return successful_count


def main():
    """主函数：处理命令行参数并提取帧。"""
    parser = argparse.ArgumentParser(
        description="从悬挂法测物体重心模拟视频中提取第一帧"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集根目录路径（Dataset_V2）"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # 检查数据集路径是否存在
    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    if not dataset_path.is_dir():
        print(f"错误: 数据集路径不是目录: {dataset_path}")
        sys.exit(1)
    
    print(f"悬挂法测物体重心 - 模拟视频第一帧提取器")
    print(f"数据集路径: {dataset_path}")
    print()
    
    total_processed = process_cg_sim_experiment(dataset_path)
    
    print(f"\n{'='*60}")
    print(f"总共成功处理: {total_processed} 个视频")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()



