# hagrid (sample 30k 384p) -> yolo format
# python prepare_dataset.py --src <hagrid-sample-30k-384p> --dst datasets/gesture3

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["like", "dislike", "palm"]
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}

# read  ann_train_val/{like, dislike, plam}.json 
def load_samples(src):
    samples = []  # (img, yolo lines, user_id)

    for cls in CLASSES:
        ann = json.loads((src / "ann_train_val" / f"{cls}.json").read_text())
        img_dir = src / "hagrid_30k" / f"train_val_{cls}"

        for img_id, a in ann.items():
            img = img_dir / f"{img_id}.jpg"
            if not img.exists():
                continue

            lines = []
            for (x, y, w, h), lab in zip(a["bboxes"], a["labels"]):
                if lab not in CLASSES:  # no_gesture skip
                    continue
                # top-left xywh -> center xywh (HaGRID => YOLO)
                cx, cy = x + w / 2,  y + h / 2
                lines.append(f"{CLASSES.index(lab)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                # class names -  0: like  1: dislike  2: palm

            if lines:
                samples.append((img, lines, a["user_id"]))
    return samples


# split by user_id (same person not in train & val)
def split_users(samples, seed):
    users = sorted({s[2] for s in samples}) 
    random.Random(seed).shuffle(users)        # seed : arg

    n = len(users)
    n_tr, n_va = int(n * SPLIT["train"]), int(n * SPLIT["val"])
    res = {}
    for i, u in enumerate(users):
        res[u] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="datasets/gesture3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    samples = load_samples(src)
    user_split = split_users(samples, args.seed)

    if dst.exists():   
        shutil.rmtree(dst) # optional init

    stat = defaultdict(Counter) # train, val, test img count (images:all)
    for img, lines, u in samples:
        sp = user_split[u]
        (dst / "images" / sp).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, dst / "images" / sp / img.name) # save metadata
        (dst / "labels" / sp / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")

        stat[sp]["images"] += 1
        for l in lines:
            stat[sp][CLASSES[int(l.split()[0])]] += 1 

    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        "names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    ) # export data

    print(f"users: {len(user_split)}")
    for sp in SPLIT:
        print(sp, dict(stat[sp]))
    # debug


if __name__ == "__main__":
    main()
