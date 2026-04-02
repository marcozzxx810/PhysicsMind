#!/usr/bin/env python3
"""
Build a CSV file for the simulated suspended center-of-gravity experiment.
Generate two questions for each video:
1. Center-of-gravity position (`center_of_gravity`)
2. Rotation direction (`rotation_direction`)
"""

import os
import sys
import argparse
import csv
from pathlib import Path


# Dataset folder name on disk for the suspended center-of-gravity experiment.
EXPERIMENT_DIR_NAME = "\u60ac\u6302\u6cd5\u6d4b\u7269\u4f53\u91cd\u5fc3"


def get_video_keyframe_pairs(experiment_path):
    """
    Pair videos with keyframes under the experiment directory.
    
    Args:
        experiment_path (Path): Experiment directory path
    
    Returns:
        list: Tuples of video files and keyframe files
    """
    variants_sim_path = experiment_path / "variants_sim"
    key_frames_sim_path = experiment_path / "key_frames_sim"
    
    if not variants_sim_path.exists() or not key_frames_sim_path.exists():
        return []
    
    # Supported video formats.
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Collect all video files.
    video_files = {}
    for video_file in variants_sim_path.iterdir():
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # Skip hidden files.
            if not video_file.name.startswith('._'):
                video_files[video_file.stem] = video_file
    
    # Collect all keyframe files.
    keyframe_files = {}
    for keyframe_file in key_frames_sim_path.iterdir():
        if keyframe_file.is_file() and keyframe_file.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            keyframe_files[keyframe_file.stem] = keyframe_file
    
    # Match videos to keyframes.
    pairs = []
    for video_stem, video_file in sorted(video_files.items()):
        if video_stem in keyframe_files:
            keyframe_file = keyframe_files[video_stem]
            pairs.append((video_file, keyframe_file))
        else:
            print(f"Warning: no matching keyframe found for video {video_file.name}")
    
    return pairs


def build_cg_sim_csv(dataset_path, output_csv_path, experiment_name=EXPERIMENT_DIR_NAME):
    """
    Build a CSV file for the center-of-gravity simulation experiment.
    
    Args:
        dataset_path (Path): Dataset root directory
        output_csv_path (Path): Output CSV path
        experiment_name (str): Experiment directory name
    """
    dataset_path = Path(dataset_path)
    experiment_path = dataset_path / experiment_name
    
    if not experiment_path.exists():
        print(f"Error: experiment directory does not exist: {experiment_path}")
        sys.exit(1)
    
    # Collect video/keyframe pairs.
    pairs = get_video_keyframe_pairs(experiment_path)
    
    if not pairs:
        print(f"Error: no video/keyframe pairs found under {experiment_name}")
        sys.exit(1)
    
    print(f"Found {len(pairs)} video/keyframe pairs")
    
    # Prepare CSV rows.
    rows = []
    
    for video_file, keyframe_file in pairs:
        # Use paths relative to the dataset root.
        video_rel_path = video_file.relative_to(dataset_path)
        keyframe_rel_path = keyframe_file.relative_to(dataset_path)
        
        # Create two rows per video.
        question_categories = [
            'center_of_gravity',
            'rotation_direction',
        ]
        
        for category in question_categories:
            row = {
                'experiment_name': experiment_name,
                'video_path': str(video_rel_path),
                'key_frame_path': str(keyframe_rel_path),
                'question_type': 'single_choice',
                'question_category': category,
                'question': '',  # Filled later by the question-generation script.
                'ground truth': ''  # Filled manually.
            }
            rows.append(row)
    
    # Write the CSV.
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['experiment_name', 'video_path', 'key_frame_path', 
                     'question_type', 'question_category', 'question', 'ground truth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n{'='*60}")
    print(f"CSV file created: {output_csv_path}")
    print(f"Total rows: {len(rows)}")
    print(f"  - Videos: {len(pairs)}")
    print(f"  - Two questions per video (center of gravity position + rotation direction)")
    print(f"{'='*60}")


def main():
    """Parse command-line arguments and build the CSV file."""
    parser = argparse.ArgumentParser(
        description="Build a CSV file for the simulated suspended center-of-gravity experiment"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory (Dataset_V2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/PhysBench/output/apl/dataset_cg_sim.csv",
        help="Output CSV file path (default: /root/PhysBench/output/apl/dataset_cg_sim.csv)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    output_csv_path = Path(args.output)
    
    # Validate paths.
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    print(f"Suspended center-of-gravity CSV builder")
    print(f"Dataset path: {dataset_path}")
    print(f"Output CSV: {output_csv_path}")
    print()
    
    build_cg_sim_csv(dataset_path, output_csv_path)


if __name__ == "__main__":
    main()
