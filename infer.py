import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--model", required=True, help="Path to trained weights")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    model = YOLO(args.model)

    model.track(
        source=args.video,
        conf=args.conf,
        tracker="bytetrack.yaml",
        save=True,
        save_txt=True,
        vid_stride=args.stride,
        device="cpu",
        persist=True
    )

if __name__ == "__main__":
    main()
