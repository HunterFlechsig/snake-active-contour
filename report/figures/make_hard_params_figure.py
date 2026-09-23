"""Assemble default vs --skimage panels from headless replay snapshots."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "images"
RESULTS = ROOT / "results"


def load(test, name):
    bgr = cv2.imread(str(RESULTS / f"test_{test}" / name))
    if bgr is None:
        raise SystemExit(f"missing results/test_{test}/{name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    col_titles = (
        r"Live defaults" "\n" r"$\alpha=0.08$, $\beta=0.40$, $\gamma=0.10$",
        r"--skimage" "\n" r"$\alpha=0.015$, $\beta=10$, $\gamma=0.001$",
    )
    rows = (
        ("Horseshoe, step 500", load(40, "iter_0500.png"), load(41, "iter_0500.png")),
        ("Hand, step 2000", load(42, "iter_2000.png"), load(43, "iter_2000.png")),
        ("Quarter, step 200", load(44, "iter_0200.png"), load(45, "iter_0200.png")),
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 9.6))
    for row, (label, left, right) in enumerate(rows):
        for col, image in enumerate((left, right)):
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(col_titles[col] + "\n" + label, fontsize=10)
            else:
                ax.set_title(label, fontsize=10)
    fig.tight_layout()
    path = OUT / "hard_params.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
