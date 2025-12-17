#!/usr/bin/env python3
"""
Generate trajectory tracks from videos using CoTracker.
Reads query points from annotations.json.
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np


def load_video(video_path, max_size=1000):
    """Load video with optional downsampling."""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    scale = min(max_size / w, max_size / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (new_w, new_h))
        frames.append(frame)
    cap.release()
    
    if not frames:
        raise ValueError(f"Cannot read video: {video_path}")
    
    return np.array(frames), scale


def load_queries(anno_path, video_name):
    """Load query points from annotations."""
    with open(anno_path) as f:
        data = json.load(f)
    
    if video_name not in data:
        raise ValueError(f"Video {video_name} not in annotations")
    
    queries = data[video_name].get('queries', [])
    if not queries:
        raise ValueError(f"No queries for {video_name}")
    
    return np.array(queries)


_model = None
_checkpoint = None


def get_model(checkpoint_path, device, use_online=False):
    """Load CoTracker model (cached). Auto-downloads if no checkpoint provided."""
    global _model, _checkpoint
    
    if _model is not None and _checkpoint == checkpoint_path:
        return _model
    
    from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        if use_online:
            model = CoTrackerOnlinePredictor(checkpoint=checkpoint_path, window_len=16)
        else:
            model = CoTrackerPredictor(checkpoint=checkpoint_path)
    else:
        print("Downloading CoTracker from torch.hub...")
        name = "cotracker3_online" if use_online else "cotracker3_offline"
        model = torch.hub.load("facebookresearch/co-tracker", name)
    
    model = model.to(device).eval()
    _model = model
    _checkpoint = checkpoint_path
    return model


def generate_tracks(video_path, queries, output_path, checkpoint=None, 
                   device="cuda", max_size=1000, use_online=False):
    """Generate tracks for a single video."""
    video, scale = load_video(video_path, max_size)
    
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(device)
    
    queries_scaled = queries.copy().astype(np.float32)
    queries_scaled[:, 1:] *= scale
    queries_tensor = torch.from_numpy(queries_scaled)[None].to(device)
    
    model = get_model(checkpoint, device, use_online)
    
    with torch.no_grad():
        tracks, _ = model(video_tensor, queries=queries_tensor)
    
    tracks_np = tracks[0].cpu().numpy()
    tracks_np[..., :2] /= scale
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tracks_np)
    
    del video_tensor, queries_tensor
    torch.cuda.empty_cache()
    
    return tracks_np


def process_single(video_path, anno_path, output_dir, **kwargs):
    """Process a single video."""
    video_path = Path(video_path)
    video_name = video_path.name
    
    queries = load_queries(anno_path, video_name)
    output_path = Path(output_dir) / f"{video_path.stem}.npy"
    
    if output_path.exists():
        print(f"Skipping {video_name} (exists)")
        return
    
    print(f"Processing {video_name}")
    generate_tracks(video_path, queries, output_path, **kwargs)
    print(f"Saved {output_path}")


def process_batch(video_dir, output_dir, **kwargs):
    """Process all videos in directory."""
    video_dir = Path(video_dir)
    anno_path = video_dir / "annotations.json"
    
    if not anno_path.exists():
        raise FileNotFoundError(f"annotations.json not found")
    
    videos = sorted(video_dir.glob("*.mp4"))
    print(f"Processing {len(videos)} videos...")
    
    for video_path in videos:
        try:
            process_single(video_path, anno_path, output_dir, **kwargs)
        except Exception as e:
            print(f"Error {video_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate tracks using CoTracker")
    subparsers = parser.add_subparsers(dest="mode")
    
    single = subparsers.add_parser("single")
    single.add_argument("--video", required=True)
    single.add_argument("--annotations", required=True)
    single.add_argument("--output_dir", required=True)
    single.add_argument("--checkpoint")
    single.add_argument("--device", default="cuda")
    single.add_argument("--max_size", type=int, default=1000)
    single.add_argument("--online", action="store_true")
    
    batch = subparsers.add_parser("batch")
    batch.add_argument("--video_dir", required=True)
    batch.add_argument("--output_dir", required=True)
    batch.add_argument("--checkpoint")
    batch.add_argument("--device", default="cuda")
    batch.add_argument("--max_size", type=int, default=1000)
    batch.add_argument("--online", action="store_true")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        process_single(args.video, args.annotations, args.output_dir,
                      checkpoint=args.checkpoint, device=args.device,
                      max_size=args.max_size, use_online=args.online)
    elif args.mode == "batch":
        process_batch(args.video_dir, args.output_dir,
                     checkpoint=args.checkpoint, device=args.device,
                     max_size=args.max_size, use_online=args.online)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

