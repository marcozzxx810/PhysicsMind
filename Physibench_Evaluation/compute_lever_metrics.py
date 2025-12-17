#!/usr/bin/env python3
"""
Lever Balance Experiment - Direction Accuracy metric.
Compares lever tilt direction between GT and predicted videos using VLM.
"""

import os
import cv2
import json
import base64
import argparse
import time
from pathlib import Path

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def extract_last_frame(video_path, save_path=None):
    """Extract the last frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Cannot read last frame: {video_path}")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), frame)
    
    return frame


def encode_image(image):
    """Encode image to base64."""
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode()
    _, buf = cv2.imencode('.jpg', image)
    return base64.b64encode(buf).decode()


PROMPT = """Analyze these two lever experiment images.
Image 1: Ground truth (real experiment)
Image 2: AI-generated prediction

Determine the lever tilt direction in each:
- "left": left side down
- "right": right side down  
- "balanced": level

Reply with JSON only: {"gt": "left/right/balanced", "pred": "left/right/balanced"}"""


def query_openai(gt_frame, pred_frame, api_key):
    """Query GPT-4V for direction comparison."""
    client = openai.OpenAI(api_key=api_key)
    gt_b64 = encode_image(gt_frame)
    pred_b64 = encode_image(pred_frame)
    
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{gt_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pred_b64}"}}
            ]
        }],
        max_tokens=100
    )
    
    text = resp.choices[0].message.content.strip()
    try:
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        result = json.loads(text)
        result["match"] = result["gt"] == result["pred"]
        return result
    except:
        return {"gt": "unknown", "pred": "unknown", "match": False}


def query_anthropic(gt_frame, pred_frame, api_key):
    """Query Claude for direction comparison."""
    client = anthropic.Anthropic(api_key=api_key)
    gt_b64 = encode_image(gt_frame)
    pred_b64 = encode_image(pred_frame)
    
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": gt_b64}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": pred_b64}}
            ]
        }]
    )
    
    text = resp.content[0].text.strip()
    try:
        if "{" in text:
            text = text[text.find("{"):text.rfind("}")+1]
        result = json.loads(text)
        result["match"] = result["gt"] == result["pred"]
        return result
    except:
        return {"gt": "unknown", "pred": "unknown", "match": False}


def evaluate(gt_video_dir, pred_video_dir, api="openai", api_key=None, output_file=None):
    """Compute direction accuracy across all video pairs."""
    gt_dir = Path(gt_video_dir)
    pred_dir = Path(pred_video_dir)
    
    gt_videos = {p.stem: p for p in gt_dir.glob("*.mp4")}
    pred_videos = {p.stem: p for p in pred_dir.glob("*.mp4")}
    common = set(gt_videos.keys()) & set(pred_videos.keys())
    
    if not common:
        print("No matching videos found")
        return None
    
    print(f"Found {len(common)} matching video pairs")
    
    if api == "openai":
        if not HAS_OPENAI:
            raise ImportError("pip install openai")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        query_fn = lambda g, p: query_openai(g, p, api_key)
    else:
        if not HAS_ANTHROPIC:
            raise ImportError("pip install anthropic")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        query_fn = lambda g, p: query_anthropic(g, p, api_key)
    
    if not api_key:
        raise ValueError(f"API key required for {api}")
    
    results = []
    correct = 0
    
    for name in sorted(common):
        try:
            gt_frame = extract_last_frame(gt_videos[name])
            pred_frame = extract_last_frame(pred_videos[name])
            result = query_fn(gt_frame, pred_frame)
            result["video"] = name
            results.append(result)
            if result["match"]:
                correct += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Error {name}: {e}")
            results.append({"video": name, "match": False, "error": str(e)})
    
    accuracy = correct / len(results) if results else 0.0
    summary = {
        "direction_accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "results": results
    }
    
    print(f"\nDirection Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct: {correct}/{len(results)}")
    
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_file}")
    
    return summary


def batch_extract_frames(video_dir, output_dir):
    """Extract last frames from all videos in directory."""
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video in video_dir.glob("*.mp4"):
        out_path = output_dir / f"{video.stem}.png"
        try:
            extract_last_frame(video, out_path)
        except Exception as e:
            print(f"Error {video.name}: {e}")
    
    print(f"Extracted frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compute lever direction accuracy")
    subparsers = parser.add_subparsers(dest="cmd")
    
    eval_p = subparsers.add_parser("evaluate")
    eval_p.add_argument("--gt_dir", required=True)
    eval_p.add_argument("--pred_dir", required=True)
    eval_p.add_argument("--api", choices=["openai", "anthropic"], default="openai")
    eval_p.add_argument("--api_key")
    eval_p.add_argument("--output")
    
    ext_p = subparsers.add_parser("extract")
    ext_p.add_argument("--video_dir", required=True)
    ext_p.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    
    if args.cmd == "evaluate":
        evaluate(args.gt_dir, args.pred_dir, args.api, args.api_key, args.output)
    elif args.cmd == "extract":
        batch_extract_frames(args.video_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

