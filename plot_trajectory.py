# plot box center trajectory per track id (from track_video.py csv)
# python plot_trajectory.py results/tracking_trajectory.csv results/trajectory.png

import csv
import sys
from collections import defaultdict

import matplotlib.pyplot as plt

COLORS = {"like": "tab:green", "dislike": "tab:red", "palm": "tab:blue"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/tracking_trajectory.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/trajectory.png"

    tracks = defaultdict(list)  # (tid, cls) -> [(x, y)]
    for r in csv.DictReader(open(src)):
        tracks[(r["track_id"], r["class"])].append((int(r["cx"]), int(r["cy"])))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (tid, cls), pts in tracks.items():
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", ms=2, color=COLORS.get(cls, "gray"), label=f"#{tid} {cls} ({len(pts)})")

    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)  # image coord
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("track center trajectory")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
