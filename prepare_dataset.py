"""
HaGRID(sample 30k 384p) -> YOLO 형식 변환
like / dislike / palm 3개 부류만 사용, no_gesture 박스는 제외
train/val/test 는 user_id 기준으로 분리 (같은 사람이 양쪽에 들어가지 않게)

usage: python prepare_dataset.py --src <hagrid-sample-30k-384p 경로> --dst datasets/gesture3
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["like", "dislike", "palm"]
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="datasets/gesture3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    samples = []  # (img_path, yolo_lines, user_id)
    for cls in CLASSES:
        ann = json.loads((src / "ann_train_val" / f"{cls}.json").read_text())
        img_dir = src / "hagrid_30k" / f"train_val_{cls}"
        for img_id, a in ann.items():
            img = img_dir / f"{img_id}.jpg"
            if not img.exists():
                continue
            lines = []
            for (x, y, w, h), lab in zip(a["bboxes"], a["labels"]):
                if lab not in CLASSES:
                    continue
                cx, cy = x + w / 2, y + h / 2
                lines.append(f"{CLASSES.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            if lines:
                samples.append((img, lines, a["user_id"]))

    # user_id 단위 분리
    users = sorted({s[2] for s in samples})
    random.Random(args.seed).shuffle(users)
    n = len(users)
    n_tr, n_va = int(n * SPLIT["train"]), int(n * SPLIT["val"])
    user_split = {}
    for i, u in enumerate(users):
        user_split[u] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

    if dst.exists():
        shutil.rmtree(dst)
    stat = defaultdict(Counter)
    for img, lines, u in samples:
        sp = user_split[u]
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst / "images" / sp / img.name)
        (dst / "labels" / sp / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
        stat[sp]["images"] += 1
        for l in lines:
            stat[sp][CLASSES[int(l.split()[0])]] += 1

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )

    print(f"users: {n}")
    for sp in SPLIT:
        print(sp, dict(stat[sp]))


if __name__ == "__main__":
    main()
