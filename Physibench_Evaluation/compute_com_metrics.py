#!/usr/bin/env python3
"""
Center of Mass Experiment - IoU and Center Distance metrics.
Handles resolution alignment between GT and predicted masks.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from scipy.ndimage import center_of_mass


def resize_mask(mask, target_shape):
    """Resize mask to target shape using nearest neighbor interpolation."""
    if mask.shape == target_shape:
        return mask
    return cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)


def compute_iou(gt_mask, pred_mask):
    """
    Compute IoU between two binary masks.
    IoU = intersection / union
    """
    gt_binary = (gt_mask > 127).astype(np.uint8)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_center_distance(gt_mask, pred_mask, gt_shape, pred_shape):
    """
    Compute normalized center distance between mask centroids.
    Coordinates are normalized to [0,1] before computing distance.
    """
    gt_binary = (gt_mask > 127).astype(np.uint8)
    pred_binary = (pred_mask > 127).astype(np.uint8)
    
    if not np.any(gt_binary) or not np.any(pred_binary):
        return None
    
    # center_of_mass returns (row, col) = (y, x)
    gt_cy, gt_cx = center_of_mass(gt_binary)
    pred_cy, pred_cx = center_of_mass(pred_binary)
    
    if np.isnan(gt_cx) or np.isnan(pred_cx):
        return None
    
    # Normalize to [0,1] using original resolution
    gt_h, gt_w = gt_shape
    pred_h, pred_w = pred_shape
    
    gt_norm = np.array([gt_cx / gt_w, gt_cy / gt_h])
    pred_norm = np.array([pred_cx / pred_w, pred_cy / pred_h])
    
    # Euclidean distance in normalized space, scaled to pixels at 1920x1080
    norm_dist = np.linalg.norm(gt_norm - pred_norm)
    pixel_dist = norm_dist * np.sqrt(1920**2 + 1080**2)
    
    return float(pixel_dist)


def find_matching_masks(gt_dir, pred_dir):
    """Find matching mask pairs between GT and prediction directories."""
    gt_masks = {p.stem: p for p in Path(gt_dir).glob("*.png")}
    pred_masks = {p.stem: p for p in Path(pred_dir).glob("*.png")}
    
    matches = {}
    for gt_stem in gt_masks:
        base = gt_stem.replace("_seg_mask", "")
        for pred_stem in pred_masks:
            pred_base = pred_stem.replace("_seg_mask", "")
            if base == pred_base:
                matches[gt_stem] = pred_stem
                break
    
    return matches, gt_masks, pred_masks


def evaluate(gt_mask_dir, pred_mask_dir, output_file=None):
    """
    Evaluate IoU and center distance metrics.
    Masks are aligned to the same resolution before IoU computation.
    """
    matches, gt_paths, pred_paths = find_matching_masks(gt_mask_dir, pred_mask_dir)
    
    if not matches:
        print(f"No matching masks found between {gt_mask_dir} and {pred_mask_dir}")
        return None
    
    print(f"Found {len(matches)} matching mask pairs")
    
    results = []
    iou_scores = []
    distances = []
    
    for gt_stem, pred_stem in matches.items():
        gt_path = gt_paths[gt_stem]
        pred_path = pred_paths[pred_stem]
        
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        
        if gt_mask is None or pred_mask is None:
            continue
        
        gt_shape = gt_mask.shape
        pred_shape = pred_mask.shape
        
        # Align resolution: resize pred to GT size for IoU
        pred_aligned = resize_mask(pred_mask, gt_shape)
        
        iou = compute_iou(gt_mask, pred_aligned)
        dist = compute_center_distance(gt_mask, pred_mask, gt_shape, pred_shape)
        
        results.append({
            "file": gt_stem,
            "iou": iou,
            "center_distance": dist
        })
        
        iou_scores.append(iou)
        if dist is not None:
            distances.append(dist)
    
    valid_distances = [d for d in distances if not np.isnan(d)]
    
    summary = {
        "avg_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
        "avg_center_distance": float(np.mean(valid_distances)) if valid_distances else 0.0,
        "num_samples": len(results),
        "results": results
    }
    
    print(f"\nResults:")
    print(f"  Avg IoU: {summary['avg_iou']:.4f} ({summary['avg_iou']*100:.2f}%)")
    print(f"  Avg Center Distance: {summary['avg_center_distance']:.2f} px")
    print(f"  Samples: {summary['num_samples']}")
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compute IoU and Center Distance metrics")
    parser.add_argument("--gt_dir", required=True, help="GT mask directory")
    parser.add_argument("--pred_dir", required=True, help="Predicted mask directory")
    parser.add_argument("--output", help="Output JSON file")
    
    args = parser.parse_args()
    evaluate(args.gt_dir, args.pred_dir, args.output)


if __name__ == "__main__":
    main()

