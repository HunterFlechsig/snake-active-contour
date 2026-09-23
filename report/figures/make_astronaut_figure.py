"""Astronaut head: same start, this report's constants vs scikit-image."""

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian
from skimage.segmentation import active_contour

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physics import evolve_snake
import defaults

OUT = Path(__file__).resolve().parents[1] / "images"
IMAGE = ROOT / "images" / "ast.jpg"

# Circle around the head, sitting in the collar.
CENTER = np.array([214.0, 108.0])
RADIUS = 92.0
N = 400


def circle_xy(center, radius, n):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([
        center[0] + radius * np.cos(s),
        center[1] + radius * np.sin(s),
    ])


def overlay_snake(ax, snake, color, lw=1.3, ms=4):
    closed = np.vstack([snake, snake[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw)
    ax.scatter(
        snake[:, 0], snake[:, 1],
        s=ms, c=color, zorder=3, edgecolors="white", linewidths=0.15,
    )


def panel(ax, image, snake, title):
    ax.imshow(image, cmap="gray", origin="upper")
    overlay_snake(ax, snake, color="#d62728", lw=1.2, ms=2)
    ax.set_xlim(55, 370)
    ax.set_ylim(250, 0)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def main():
    image = cv2.imread(str(IMAGE), cv2.IMREAD_GRAYSCALE)
    snake0 = circle_xy(CENTER, RADIUS, N)
    print("image", image.shape, "n", len(snake0))

    ours, ours_steps = evolve_snake(
        snake0,
        image,
        alpha=defaults.ALPHA,
        beta=defaults.BETA,
        gamma=defaults.GAMMA,
        kappa=defaults.KAPPA,
        w_line=defaults.W_LINE,
        w_edge=defaults.W_EDGE,
        w_term=defaults.W_TERM,
        sigmas=(defaults.SIGMA,),
        max_iterations=2500,
        convergence=defaults.AUTO_CONVERGENCE,
        max_px_move=defaults.MAX_PX_MOVE,
    )
    print("ours steps", ours_steps)

    init_rc = np.column_stack([snake0[:, 1], snake0[:, 0]])
    blur = gaussian(image.astype(np.float64) / 255.0, sigma=3, preserve_range=False)
    sk = active_contour(
        blur,
        init_rc,
        alpha=defaults.SKIMAGE_ALPHA,
        beta=defaults.SKIMAGE_BETA,
        gamma=defaults.SKIMAGE_GAMMA,
        max_num_iter=2500,
    )
    sk_xy = np.column_stack([sk[:, 1], sk[:, 0]])
    print("skimage mean r", float(np.mean(np.linalg.norm(sk_xy - CENTER, axis=1))))

    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.15))
    panel(axes[0], image, snake0, "Initial circle")
    panel(
        axes[1],
        image,
        ours,
        r"This implementation"
        "\n"
        rf"$\alpha={defaults.ALPHA:g}$, $\beta={defaults.BETA:g}$, $\gamma={defaults.GAMMA:g}$",
    )
    panel(
        axes[2],
        image,
        sk_xy,
        r"scikit-image"
        "\n"
        rf"$\alpha={defaults.SKIMAGE_ALPHA:g}$, $\beta={defaults.SKIMAGE_BETA:g}$, $\gamma={defaults.SKIMAGE_GAMMA:g}$",
    )
    fig.tight_layout()
    path = OUT / "astronaut.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
