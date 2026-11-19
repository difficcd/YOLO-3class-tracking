"""
학습된 모델로 동영상 객체 감지 + 추적 (ByteTrack)
박스/ID 위에 객체 중심점 궤적(trajectory)을 같이 그리고, 궤적은 csv 로 따로 저장

usage: python track_video.py --weights weights/best.pt --src input.mp4 --out results/tracking.mp4
"""
import argparse
import csv
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

COLORS = {0: (80, 200, 80), 1: (60, 60, 230), 2: (230, 160, 40)}  # like, dislike, palm (BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="results/tracking.mp4")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=0, help="앞에서부터 n초만 처리 (0 = 전체)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * args.seconds) if args.seconds else None
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = defaultdict(lambda: deque(maxlen=40))
    rows = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and f >= max_frames):
            break
        r = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz,
                        tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                                   r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                trails[tid].append((cx, cy))
                rows.append([f, tid, model.names[c], round(s, 3), cx, cy])
                col = COLORS.get(c, (255, 255, 255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                label = f"#{tid} {model.names[c]} {s:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - th - 8), (int(x1) + tw + 6, int(y1)), col, -1)
                cv2.putText(frame, label, (int(x1) + 3, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                pts = list(trails[tid])
                for a, b in zip(pts, pts[1:]):
                    cv2.line(frame, a, b, col, 3)
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    with open(args.out.rsplit(".", 1)[0] + "_trajectory.csv", "w", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["frame", "track_id", "class", "conf", "cx", "cy"])
        wr.writerows(rows)
    print(f"{f} frames -> {args.out}")


if __name__ == "__main__":
    main()
