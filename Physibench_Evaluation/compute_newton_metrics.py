#!/usr/bin/env python3
"""
Newton's First Law Experiment - 5 trajectory metrics.
Handles resolution normalization between GT and predicted tracks.
"""

import numpy as np
import cv2
import json
import argparse
from pathlib import Path


def get_video_resolution(video_path):
    """Get video width and height."""
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def normalize_tracks(tracks, width, height):
    """Normalize pixel coordinates to [0,1]."""
    normalized = tracks.copy().astype(np.float64)
    normalized[..., 0] /= width
    normalized[..., 1] /= height
    return normalized


def interpolate_tracks(gt, pred):
    """Interpolate both tracks to the shorter length."""
    gt_len, pred_len = len(gt), len(pred)
    target_len = min(gt_len, pred_len)
    
    gt_t = np.linspace(0, 1, gt_len)
    pred_t = np.linspace(0, 1, pred_len)
    common_t = np.linspace(0, 1, target_len)
    
    gt_interp = np.zeros((target_len, gt.shape[1], 2))
    pred_interp = np.zeros((target_len, pred.shape[1], 2))
    
    for n in range(gt.shape[1]):
        for d in range(2):
            gt_interp[:, n, d] = np.interp(common_t, gt_t, gt[:, n, d])
            pred_interp[:, n, d] = np.interp(common_t, pred_t, pred[:, n, d])
    
    return gt_interp, pred_interp


def trajectory_rmse(gt, pred):
    """RMSE of position errors (Eq. 4)."""
    sq_dist = np.sum((gt - pred) ** 2, axis=2)
    return float(np.sqrt(np.mean(sq_dist)))


def final_position_error(gt, pred):
    """Final position error normalized by GT path length (Eq. 5-6)."""
    T, N, _ = gt.shape
    fpe_list = []
    
    for n in range(N):
        final_dist = np.linalg.norm(gt[-1, n] - pred[-1, n])
        path_len = np.sum(np.linalg.norm(np.diff(gt[:, n], axis=0), axis=1))
        fpe = final_dist / path_len if path_len > 1e-8 else 0.0
        fpe_list.append(fpe)
    
    return float(np.mean(fpe_list))


def speed_similarity(gt, pred):
    """Cosine similarity of velocity vectors (Eq. 7-8)."""
    gt_vel = np.diff(gt, axis=0)
    pred_vel = np.diff(pred, axis=0)
    
    gt_norm = np.linalg.norm(gt_vel, axis=2)
    pred_norm = np.linalg.norm(pred_vel, axis=2)
    dot = np.sum(gt_vel * pred_vel, axis=2)
    
    denom = gt_norm * pred_norm
    valid = denom > 1e-8
    
    cos_sim = np.zeros_like(dot)
    cos_sim[valid] = dot[valid] / denom[valid]
    cos_sim[(gt_norm < 1e-8) & (pred_norm < 1e-8)] = 1.0
    
    return float(np.mean(cos_sim))


def acceleration_similarity(gt, pred):
    """Cosine similarity of acceleration vectors (Eq. 9-10)."""
    if len(gt) < 3:
        return 0.0
    
    gt_vel = np.diff(gt, axis=0)
    pred_vel = np.diff(pred, axis=0)
    gt_acc = np.diff(gt_vel, axis=0)
    pred_acc = np.diff(pred_vel, axis=0)
    
    gt_norm = np.linalg.norm(gt_acc, axis=2)
    pred_norm = np.linalg.norm(pred_acc, axis=2)
    dot = np.sum(gt_acc * pred_acc, axis=2)
    
    denom = gt_norm * pred_norm
    valid = denom > 1e-8
    
    cos_sim = np.zeros_like(dot)
    cos_sim[valid] = dot[valid] / denom[valid]
    cos_sim[(gt_norm < 1e-8) & (pred_norm < 1e-8)] = 1.0
    
    return float(np.mean(cos_sim))


def directional_consistency(gt, pred):
    """Direction consistency based on overall displacement (Eq. 11-13)."""
    T, N, _ = gt.shape
    scores = []
    
    for n in range(N):
        d_gt = gt[-1, n] - gt[0, n]
        d_pred = pred[-1, n] - pred[0, n]
        
        norm_gt = np.linalg.norm(d_gt)
        norm_pred = np.linalg.norm(d_pred)
        
        if norm_gt < 1e-8 and norm_pred < 1e-8:
            scores.append(1.0)
        elif norm_gt < 1e-8 or norm_pred < 1e-8:
            scores.append(0.0)
        else:
            cos_theta = np.clip(np.dot(d_gt, d_pred) / (norm_gt * norm_pred), -1, 1)
            theta = np.degrees(np.arccos(cos_theta))
            scores.append((180 - theta) / 180)
    
    return float(np.mean(scores))


def compute_metrics(gt_tracks, pred_tracks, gt_res, pred_res):
    """Compute all 5 metrics with resolution normalization."""
    gt_w, gt_h = gt_res
    pred_w, pred_h = pred_res
    
    # Ensure 3D shape (T, N, 2)
    if gt_tracks.ndim == 2:
        gt_tracks = gt_tracks[:, np.newaxis, :]
    if pred_tracks.ndim == 2:
        pred_tracks = pred_tracks[:, np.newaxis, :]
    
    # Normalize to [0,1]
    gt_norm = normalize_tracks(gt_tracks, gt_w, gt_h)
    pred_norm = normalize_tracks(pred_tracks, pred_w, pred_h)
    
    # Align temporal length
    gt_interp, pred_interp = interpolate_tracks(gt_norm, pred_norm)
    
    return {
        "trajectory_rmse": trajectory_rmse(gt_interp, pred_interp),
        "final_position_error": final_position_error(gt_interp, pred_interp),
        "speed_similarity": speed_similarity(gt_interp, pred_interp),
        "acceleration_similarity": acceleration_similarity(gt_interp, pred_interp),
        "directional_consistency": directional_consistency(gt_interp, pred_interp)
    }


def evaluate_single(gt_video, gt_track, pred_video, pred_track):
    """Evaluate a single video pair."""
    gt_res = get_video_resolution(gt_video)
    pred_res = get_video_resolution(pred_video)
    
    gt_tracks = np.load(gt_track)
    pred_tracks = np.load(pred_track)
    
    return compute_metrics(gt_tracks, pred_tracks, gt_res, pred_res)


def evaluate_batch(gt_video_dir, gt_track_dir, pred_video_dir, pred_track_dir, output_file=None):
    """Evaluate all matching video pairs in directories."""
    gt_video_dir = Path(gt_video_dir)
    gt_track_dir = Path(gt_track_dir)
    pred_video_dir = Path(pred_video_dir)
    pred_track_dir = Path(pred_track_dir)
    
    gt_tracks = {p.stem: p for p in gt_track_dir.glob("*.npy")}
    pred_tracks = {p.stem: p for p in pred_track_dir.glob("*.npy")}
    
    common = set(gt_tracks.keys()) & set(pred_tracks.keys())
    if not common:
        print("No matching tracks found")
        return None
    
    print(f"Found {len(common)} matching track pairs")
    
    results = []
    for name in sorted(common):
        gt_video = gt_video_dir / f"{name}.mp4"
        pred_video = pred_video_dir / f"{name}.mp4"
        
        if not gt_video.exists() or not pred_video.exists():
            continue
        
        try:
            metrics = evaluate_single(gt_video, gt_tracks[name], pred_video, pred_tracks[name])
            metrics["video"] = name
            results.append(metrics)
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    if not results:
        return None
    
    # Compute averages
    keys = ["trajectory_rmse", "final_position_error", "speed_similarity", 
            "acceleration_similarity", "directional_consistency"]
    avg = {k: float(np.mean([r[k] for r in results])) for k in keys}
    
    summary = {"average": avg, "num_samples": len(results), "results": results}
    
    print(f"\nResults ({len(results)} videos):")
    print(f"  Trajectory RMSE:         {avg['trajectory_rmse']:.6f}")
    print(f"  Final Position Error:    {avg['final_position_error']:.6f}")
    print(f"  Speed Similarity:        {avg['speed_similarity']:.6f}")
    print(f"  Acceleration Similarity: {avg['acceleration_similarity']:.6f}")
    print(f"  Directional Consistency: {avg['directional_consistency']:.6f}")
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute Newton's First Law trajectory metrics")
    subparsers = parser.add_subparsers(dest="mode")
    
    # Single video mode
    single = subparsers.add_parser("single", help="Evaluate single video pair")
    single.add_argument("--gt_video", required=True)
    single.add_argument("--gt_track", required=True)
    single.add_argument("--pred_video", required=True)
    single.add_argument("--pred_track", required=True)
    single.add_argument("--output")
    
    # Batch mode
    batch = subparsers.add_parser("batch", help="Evaluate all videos in directory")
    batch.add_argument("--gt_video_dir", required=True)
    batch.add_argument("--gt_track_dir", required=True)
    batch.add_argument("--pred_video_dir", required=True)
    batch.add_argument("--pred_track_dir", required=True)
    batch.add_argument("--output")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        metrics = evaluate_single(args.gt_video, args.gt_track, args.pred_video, args.pred_track)
        print(json.dumps(metrics, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
    elif args.mode == "batch":
        evaluate_batch(args.gt_video_dir, args.gt_track_dir, 
                      args.pred_video_dir, args.pred_track_dir, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

