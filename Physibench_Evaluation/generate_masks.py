#!/usr/bin/env python3
"""
Generate segmentation masks from videos using SAM.
Reads point annotations from annotations.json and extracts masks from first frame.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def extract_first_frame(video_path):
    """Extract first frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read first frame: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def generate_mask(predictor, image, points):
    """Generate mask using SAM with point prompts."""
    predictor.set_image(image)
    input_points = np.array(points)
    input_labels = np.ones(len(input_points))
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )
    
    return masks[np.argmax(scores)]


def process_directory(video_dir, checkpoint, model_type="vit_h", device="cuda"):
    """Process all videos in directory and generate masks."""
    video_dir = Path(video_dir)
    anno_path = video_dir / "annotations.json"
    
    if not anno_path.exists():
        raise FileNotFoundError(f"annotations.json not found in {video_dir}")
    
    with open(anno_path) as f:
        annotations = json.load(f)
    
    output_dir = video_dir / "seg_masks"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading SAM model: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)
    
    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Processing {len(videos)} videos...")
    
    for video_path in videos:
        video_name = video_path.name
        if video_name not in annotations:
            continue
        
        objects = annotations[video_name].get('objects', [])
        if not objects or not objects[0].get('point'):
            continue
        
        points = objects[0]['point']
        
        try:
            frame = extract_first_frame(video_path)
            mask = generate_mask(predictor, frame, points)
            mask_img = (mask * 255).astype(np.uint8)
            
            out_path = output_dir / f"{video_path.stem}.png"
            cv2.imwrite(str(out_path), mask_img)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
    
    print(f"Masks saved to {output_dir}")


def download_sam_checkpoint(model_type="vit_h"):
    """Download SAM checkpoint if not exists."""
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    
    filename = urls[model_type].split("/")[-1]
    ckpt_path = ckpt_dir / filename
    
    if ckpt_path.exists():
        return str(ckpt_path)
    
    print(f"Downloading SAM {model_type} checkpoint...")
    import urllib.request
    urllib.request.urlretrieve(urls[model_type], ckpt_path)
    print(f"Saved to {ckpt_path}")
    return str(ckpt_path)


def main():
    parser = argparse.ArgumentParser(description="Generate masks using SAM")
    parser.add_argument("--video_dir", required=True, help="Video directory with annotations.json")
    parser.add_argument("--checkpoint", help="SAM checkpoint path (auto-downloads if not provided)")
    parser.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    checkpoint = args.checkpoint
    if not checkpoint or not os.path.exists(checkpoint):
        checkpoint = download_sam_checkpoint(args.model_type)
    
    process_directory(args.video_dir, checkpoint, args.model_type, args.device)


if __name__ == "__main__":
    main()

