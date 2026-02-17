import os
import random
import cv2

IMG_DIR = "data/visdrone/images/train"
LBL_DIR = "data/visdrone/labels/train"

CLASS_NAMES = {
    0: "pedestrian",
    1: "car",
    2: "van",
    3: "truck",
    4: "bus"
}

SAMPLES = 10

images = random.sample(os.listdir(IMG_DIR), SAMPLES)

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    lbl_path = os.path.join(LBL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path, "r") as f:
        for line in f.readlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            cls = int(cls)

            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                CLASS_NAMES[cls],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

    cv2.imshow("verification", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
