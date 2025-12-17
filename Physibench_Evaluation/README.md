# Physibench Evaluation

Evaluation scripts for three physics video generation experiments.

## Structure

```
├── generate_masks.py         # Extract segmentation masks (SAM)
├── generate_tracks.py        # Extract trajectories (CoTracker)
├── compute_com_metrics.py    # Center of Mass: IoU, Center Distance
├── compute_newton_metrics.py # Newton's First Law: 5 trajectory metrics
├── compute_lever_metrics.py  # Lever Balance: Direction Accuracy
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Models

**SAM:** Auto-downloads on first run, or manually:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

**CoTracker:** Auto-downloads via torch.hub on first run.

## Annotation Format

Mask generation:
```json
{"video.mp4": {"objects": [{"point": [[x, y]]}]}}
```

Track generation:
```json
{"video.mp4": {"queries": [[frame, x, y], ...]}}
```
