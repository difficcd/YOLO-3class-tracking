# detect + track (bytetrack), draw trajectory of box center
# python track_video.py --weights weights/best.pt --src input.mp4 --out results/tracking.mp4

import argparse
import csv
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# BGR - like, dislike, palm
COLORS = {0: (80, 200, 80), 1: (60, 60, 230), 2: (230, 160, 40)}


def draw (frame, box, tid, name, conf, col, trail):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    label = f"#{tid} {name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # trajectory
    pts = list(trail)
    for a, b in zip(pts, pts[1:]):
        cv2.line(frame, a, b, col, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11s_best.pt")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="results/tracking.mp4")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    model = YOLO(args.weights) # import trained YOLO (weights)
    cap = cv2.VideoCapture(args.src) # cv2 (OpenCV) video => frame
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    trails = defaultdict(lambda: deque(maxlen=40))
    rows = []
    f = 0

    while True:
        ok, frame = cap.read() # one frame
        if not ok:
            break

        r = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz,
                        tracker="bytetrack.yaml", verbose=False)[0] # model answer
        # YOLO + bytetrack (tracker option)

        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.tolist()
            ids = r.boxes.id.int().tolist()
            clss = r.boxes.cls.int().tolist()
            confs = r.boxes.conf.tolist()

            for box, tid, c, s in zip(boxes, ids, clss, confs):
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2) 
                # tensor => list => all box (x) amid box (o)

                trails[tid].append((cx, cy))
                rows.append([f, tid, model.names[c], round(s, 3), cx, cy])
                draw(frame, box, tid, model.names[c], s, COLORS.get(c, (255, 255, 255)), trails[tid])
                # print(f, tid, model.names[c], s)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()

    # trajectory csv
    with open(args.out.rsplit(".", 1)[0] + "_trajectory.csv", "w", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["frame", "track_id", "class", "conf", "cx", "cy"])
        wr.writerows(rows)

    print(f"{f} frames -> {args.out}")


if __name__ == "__main__":
    main()
