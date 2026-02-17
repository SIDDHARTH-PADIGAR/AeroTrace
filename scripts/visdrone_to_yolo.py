## this scriptdoes the following:
# Reads DET annotations
# Filters classes
# Converts boxes
# Copies images
# Optionally subsamples

import os
import shutil
import random
from tqdm import tqdm
import cv2

# =========================
# CONFIG
# =========================

RAW_ROOT = "data/visdrone/raw"
OUT_ROOT = "data/visdrone"

SPLITS = {
    "VisDrone-DET-train": "train",
    "VisDrone-DET-val": "val"
}

SUBSET_RATIO = 0.25  # 25% of data

# VisDrone class_id -> YOLO class_id
CLASS_MAP = {
    1: 0,  # pedestrian
    4: 1,  # car
    5: 2,  # van
    6: 3,  # truck
    9: 4   # bus
}

# =========================
# UTILS
# =========================

def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(f"{OUT_ROOT}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_ROOT}/labels/{split}", exist_ok=True)

def convert_bbox(x, y, w, h, img_w, img_h):
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h
    return xc, yc, bw, bh

# =========================
# MAIN
# =========================

def process_split(raw_name, out_split):
    img_dir = os.path.join(RAW_ROOT, raw_name, "images")
    ann_dir = os.path.join(RAW_ROOT, raw_name, "annotations")

    images = sorted(os.listdir(img_dir))
    random.shuffle(images)

    keep_count = int(len(images) * SUBSET_RATIO)
    images = images[:keep_count]

    print(f"{out_split}: using {len(images)} images")

    for img_name in tqdm(images):
        img_path = os.path.join(img_dir, img_name)
        ann_path = os.path.join(ann_dir, img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        yolo_lines = []

        with open(ann_path, "r") as f:
            for line in f.readlines():
                x, y, bw, bh, _, cls_id, *_ = map(float, line.strip().split(","))
                cls_id = int(cls_id)

                if cls_id not in CLASS_MAP:
                    continue

                if bw <= 0 or bh <= 0:
                    continue

                xc, yc, nw, nh = convert_bbox(x, y, bw, bh, w, h)
                yolo_cls = CLASS_MAP[cls_id]

                yolo_lines.append(
                    f"{yolo_cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"
                )

        if not yolo_lines:
            continue

        out_img = f"{OUT_ROOT}/images/{out_split}/{img_name}"
        out_lbl = f"{OUT_ROOT}/labels/{out_split}/{img_name.replace('.jpg', '.txt')}"

        shutil.copy(img_path, out_img)

        with open(out_lbl, "w") as f:
            f.write("\n".join(yolo_lines))


if __name__ == "__main__":
    random.seed(42)
    ensure_dirs()

    for raw_split, out_split in SPLITS.items():
        process_split(raw_split, out_split)

    print("Conversion complete.")
