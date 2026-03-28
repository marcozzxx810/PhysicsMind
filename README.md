# PhysicsMind: Sim and Real Mechanics Benchmarking for Physical Reasoning and Prediction in Foundational VLMs and World Models

<p align="left">
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-25XX.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://your-domain.com/physicsmind">
    <img src="https://img.shields.io/badge/Project Page-PhysicsMind-blue" alt="Project Website">
  </a>
  <a href="#dataset">
    <img src="https://img.shields.io/badge/Dataset-PhysicsMind-yellow" alt="Dataset">
  </a>
</p>

<p align="left">
    <a href="#">Chak-Wing Mak<sup>1,*</sup></a> • <a href="#">Guanyu Zhu<sup>2,*</sup></a> • <a href="#">Boyi Zhang<sup>3,*</sup></a> • <a href="#">Hongji Li<sup>4</sup></a> • <a href="#">Xiaowei Chi<sup>5</sup></a> • <a href="#">Kevin Zhang<sup>1</sup></a> • <a href="#">Yichen Wu<sup>6</sup></a> • <a href="#">Yangfan He<sup>7</sup></a> • <a href="#">Chun-Kai Fan<sup>1</sup></a> • <a href="#">Wentao Lu<sup>8</sup></a> • <a href="#">Kuangzhi Ge<sup>1</sup></a> • <a href="#">Xinyu Fang<sup>9</sup></a> • <a href="#">Hongyang He<sup>10</sup></a> • <a href="#">Kuan Lu<sup>11</sup></a> • <a href="#">Tianxiang Xu<sup>1</sup></a> • <a href="#">Li Zhang<sup>12</sup></a> • <a href="#">Yongxin Ni<sup>13</sup></a> • <a href="#">Youhua Li<sup>14</sup></a> • <a href="#">Shanghang Zhang<sup>1,†</sup></a>
</p>

<sup>1</sup> Peking University  <sup>2</sup> South China Agricultural University  <sup>3</sup> University of the Chinese Academy of Sciences  <sup>4</sup> MBZUAI  <sup>5</sup> HKUST  <sup>6</sup> NUS  <sup>7</sup> UNC Chapel Hill  <sup>8</sup> USTC  <sup>9</sup> HFUT  <sup>10</sup> Manifold.AI  <sup>11</sup> Cornell  <sup>12</sup> PolyU  <sup>13</sup> Westlake University  <sup>14</sup> CityU

\* Equal Contribution | † Corresponding Author

---

## Overview

[**To-Do**](#to-do) | [**Dataset**](#dataset) | [**Evaluation**](#evaluation) | [**Citation**](#citation)

Modern multimodal large language models (MLLMs) and video world models excel at visual reasoning but often fail to grasp underlying physical laws. Existing benchmarks rely heavily on synthetic templates or perceptual quality metrics that do not measure physical adherence.

We introduce **PhysicsMind**, a unified benchmark combining real-world experiments and controlled simulations to evaluate **physical reasoning** and **physically plausible generation**.

We focus on three canonical mechanics scenarios:
1.  **Center of Mass**: Reasoning about mass distribution and geometric stability.
2.  **Lever Equilibrium**: Understanding torque balance and rotational dynamics.
3.  **Newton's First Law**: Predicting inertial motion and contact forces.

**PhysicsMind** evaluates models on two complementary tasks:
* **VQA (Visual Question Answering)**: Can the model reason about physical states from images?
* **Video Generation**: Can the model generate future trajectories that obey physical constraints?

---

## Dataset

The PhysicsMind dataset is organized by physical scenario and modality (Real vs. Simulation).

```
PhysicsMind/
├── CenterOfMass/
│   ├── Real/
│   │   ├── videos/       # 4K resolution recordings
│   │   └── annotations/  # Mass labels, ground truth masks
│   └── Sim/
│       ├── videos/       # 2D physics engine outputs
│       └── annotations/
├── LeverEquilibrium/
│   ├── Sim/              # Controlled torque balance setups
│   └── annotations/
└── NewtonFirstLaw/
    ├── Real/             # Inertia experiments (e.g., paper pull)
    └── annotations/
```

Each sample includes:
* **Video**: High-resolution clip (MP4).
* **Question/Prompt**: For VQA and Video Generation tasks.
* **Physics Ground Truth**: Numerical values (mass, position), segmentation masks, and trajectory data.

---

## Evaluation

We provide a unified evaluation toolkit to benchmark both VLMs and World Models.

### Physibench Evaluation (Video Generation)

We include a standalone toolkit under `Physibench_Evaluation/` for physics-aware evaluation of video generation in three canonical mechanics scenarios:

- **Center of Mass**
- **Lever Equilibrium**
- **Newton's First Law**

**Directory structure**

```text
Physibench_Evaluation/
├── generate_masks.py         # Extract segmentation masks (SAM)
├── generate_tracks.py        # Extract trajectories (CoTracker)
├── compute_com_metrics.py    # Center of Mass: IoU, Center Distance
├── compute_newton_metrics.py # Newton's First Law: 5 trajectory metrics
├── compute_lever_metrics.py  # Lever Balance: Direction Accuracy
└── requirements.txt
```

**Setup**

```bash
cd Physibench_Evaluation
pip install -r requirements.txt
```

**Models**

- **SAM**: auto-downloads on first run, or manually:

  ```bash
  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  ```

- **CoTracker**: auto-downloads via `torch.hub` on first run.

**Annotation format**

- Mask generation:

  ```json
  {"video.mp4": {"objects": [{"point": [[x, y]]}]}}
  ```

- Track generation:

  ```json
  {"video.mp4": {"queries": [[frame, x, y], ...]}}
  ```

These scripts compute PhysicsMind's physics-aware metrics for generated videos and can be plugged into any video generation pipeline that outputs MP4s and the corresponding annotations in the above formats.

### VQA Evaluation
Run the VQA benchmark to test reasoning capabilities on static frames or short clips. The evaluation reports accuracy across specific subtypes (e.g., "Rotation Prediction", "Equilibrium State").

```bash
# Example command (scripts coming soon)
python scripts/eval_vqa.py --model gpt-4o --task lever_equilibrium
```

### Video Generation Evaluation
Evaluate generated videos using our **Physics-Aware Metrics**, which measure:
* **Trajectory Consistency**: RMSE against ground truth motion.
* **Geometric Fidelity**: Segmentation Mask IoU and Center of Mass offset.
* **Physical Validity**: Check if the final state (e.g., lever tilt) matches the laws of physics).

```bash
# Example command (scripts coming soon)
python scripts/eval_video_gen.py --pred_path ./results/sora2 --metric center_of_mass
```

---

## Citation

If you find **PhysicsMind** useful for your research, please consider citing our paper:

```bibtex
@article{PhysicsMind2026,
  title={PhysicsMind: Sim and Real Mechanics Benchmarking for Physical Reasoning and Prediction in Foundational VLMs and World Models},
  author={Mak, Chak-Wing and Zhu, Guanyu and Zhang, Boyi and Li, Hongji and Chi, Xiaowei and Zhang, Kevin and Wu, Yichen and He, Yangfan and Fan, Chun-Kai and Lu, Wentao and Ge, Kuangzhi and Fang, Xinyu and He, Hongyang and Lu, Kuan and Xu, Tianxiang and Zhang, Li and Ni, Yongxin and Li, Youhua and Zhang, Shanghang},
  journal={Under Review},
  year={2026}
}
```
