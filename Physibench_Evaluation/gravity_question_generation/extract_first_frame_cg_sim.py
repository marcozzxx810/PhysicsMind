#!/usr/bin/env python3
"""
Extract the first frame from simulated videos in the suspended
center-of-gravity experiment.
Processes videos in `variants_sim` and saves frames to `key_frames_sim`.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2


# Dataset folder name on disk for the suspended center-of-gravity experiment.
EXPERIMENT_DIR_NAME = "\u60ac\u6302\u6cd5\u6d4b\u7269\u4f53\u91cd\u5fc3"


def extract_first_frame(video_path, output_path):
    """
    Extract the first frame from a video file.
    
    Args:
        video_path (Path): Input video file path
        output_path (Path): Path for the extracted frame
    
    Returns:
        bool: `True` on success, otherwise `False`
    """
    try:
        # Open the video file.
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: could not open video file {video_path}")
            return False
        
        # Get the total number of frames.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: video contains no frames {video_path}")
            cap.release()
            return False
        
        # Seek to the first frame (index 0).
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read the first frame.
        ret, frame = cap.read()
        
        if not ret:
            print(f"Error: could not read a frame from {video_path}")
            cap.release()
            return False
        
        # Save the frame.
        success = cv2.imwrite(str(output_path), frame)
        
        if success:
            print(f"Extracted first frame (0/{total_frames}) from {video_path.name} -> {output_path.name}")
        else:
            print(f"Error: could not save frame to {output_path}")
        
        cap.release()
        return success
        
    except Exception as e:
        print(f"Error while processing {video_path}: {str(e)}")
        return False


def process_cg_sim_experiment(dataset_path):
    """
    Process all videos under `variants_sim` for the suspended
    center-of-gravity experiment.
    
    Args:
        dataset_path (Path): Dataset root directory path
    
    Returns:
        int: Number of successfully processed videos
    """
    dataset_path = Path(dataset_path)
    experiment_path = dataset_path / EXPERIMENT_DIR_NAME
    variants_sim_path = experiment_path / "variants_sim"
    
    # Check whether the `variants_sim` directory exists.
    if not variants_sim_path.exists():
        print(f"Error: variants_sim directory does not exist: {variants_sim_path}")
        return 0
    
    # Create the `key_frames_sim` output directory.
    key_frames_sim_path = experiment_path / "key_frames_sim"
    key_frames_sim_path.mkdir(exist_ok=True)
    print(f"Created output directory: {key_frames_sim_path}")
    
    # Supported video extensions.
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Find all video files.
    video_files = []
    for file_path in variants_sim_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            # Skip hidden files that start with `._`.
            if not file_path.name.startswith('._'):
                video_files.append(file_path)
    
    # Sort by filename.
    video_files.sort()
    
    if not video_files:
        print(f"No video files found in {variants_sim_path}")
        return 0
    
    print(f"Found {len(video_files)} video files in {variants_sim_path}")
    
    successful_count = 0
    
    # Process each video file.
    for video_file in video_files:
        # Create the output filename by replacing the extension with `.jpg`.
        output_filename = video_file.stem + ".jpg"
        output_path = key_frames_sim_path / output_filename
        
        # Extract the first frame.
        if extract_first_frame(video_file, output_path):
            successful_count += 1
    
    print(f"\nSuccessfully processed {successful_count}/{len(video_files)} videos")
    return successful_count


def main():
    """Parse command-line arguments and extract frames."""
    parser = argparse.ArgumentParser(
        description="Extract the first frame from simulated videos in the suspended center-of-gravity experiment"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory (Dataset_V2)"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    
    # Validate the dataset path.
    if not dataset_path.exists():
        print(f"Error: dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    if not dataset_path.is_dir():
        print(f"Error: dataset path is not a directory: {dataset_path}")
        sys.exit(1)
    
    print(f"Suspended center-of-gravity first-frame extractor")
    print(f"Dataset path: {dataset_path}")
    print()
    
    total_processed = process_cg_sim_experiment(dataset_path)
    
    print(f"\n{'='*60}")
    print(f"Total successfully processed videos: {total_processed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

