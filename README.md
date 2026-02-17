# Real-Time Object Detection and Tracking on Aerial Video

This project implements an end-to-end system for detecting and tracking vehicles and pedestrians in aerial drone footage under real-world noise and resource constraints. The emphasis is on execution, system design, and practical tradeoffs rather than research novelty.

---

## Problem Statement

Aerial surveillance footage presents unique challenges including small object sizes, occlusion, motion blur, and camera movement. The objective of this project is to build a robust detection and tracking pipeline that operates reliably under these conditions and runs on limited hardware.

---

## Dataset

**VisDrone-DET (2019)**

- Train and validation splits
- ~25 percent subset sampled for faster iteration while preserving real-world noise
- Manual conversion from VisDrone annotation format to YOLO format

**Classes used**
- Pedestrian
- Car
- Van
- Truck
- Bus

---

## System Architecture

```mermaid
flowchart LR
    A[VisDrone-DET Dataset] --> B[Annotation Conversion]
    B --> C[YOLO Format Dataset]

    C --> D[YOLOv8 Training]
    D --> E[Trained Detector]

    E --> F[Frame-wise Detection]
    F --> G[ByteTrack Association]
    G --> H[Tracked Objects with IDs]

    H --> I[Annotated Video Output]
    H --> J[Per-frame Tracking Logs]

    style D fill:#e3f2fd
    style G fill:#e8f5e9
````

---

## Model and Pipeline

* **Detector:** YOLOv8n (pretrained, fine-tuned on VisDrone)
* **Tracker:** ByteTrack
* **Framework:** PyTorch with Ultralytics
* **Inference:** Frame-by-frame detection followed by IoU-based association

**Pipeline**

1. Dataset filtering and annotation conversion
2. YOLOv8 fine-tuning on aerial imagery
3. Detection evaluation using mAP
4. Multi-object tracking using ByteTrack
5. Video inference with persistent tracking IDs
6. FPS measurement under CPU-only constraints

---

## Evaluation

**Detection**

* mAP@50 ≈ 0.35
* mAP@50–95 ≈ 0.22

<img width="2400" height="1200" alt="image" src="https://github.com/user-attachments/assets/92665985-8a37-4331-bf0a-a92e9eae88e5" />


**Tracking**

* Stable tracking IDs across most scenes
* Occasional ID switches under heavy occlusion

**Performance**

* ~12 FPS on Ryzen CPU
* Resolution: 384×640
* `vid_stride=2` used to balance speed and temporal consistency

<img width="1920" height="1098" alt="image" src="https://github.com/user-attachments/assets/c7b04f93-66ba-4ad4-baa5-50f4b76c2c75" />

---

## Failure Analysis

**Observed failures**

* Occlusion causes ID switches when vehicles overlap
* Small objects missed at higher altitudes
* Motion blur during rapid camera movement lowers confidence

**Tradeoffs**

* YOLOv8n chosen for speed over peak accuracy
* ByteTrack preferred over DeepSORT to avoid re-identification overhead
* CPU-only inference prioritizes portability over raw throughput

---

## Usage

Run video inference with tracking:

```bash
python infer.py --video data/sample.mp4 --model runs/detect/train/weights/best.pt
```

**Outputs**

* Annotated video with bounding boxes and tracking IDs
* Per-frame tracking logs

---

## Repository Notes

* Raw datasets, trained weights, and large videos are excluded from the repository
* A short demo video is linked separately


