"""Figures for the methods running example: a synthetic disk, square, and stroke."""

from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physics import (
    build_implicit_operator_inverse,
    compute_image_force_field,
    constraint_forces,
    initialize_snake,
    initialize_snake_from_polyline,
    step_snake,
    _build_stiffness_matrix,
)

OUT = Path(__file__).resolve().parents[1] / "images"

ALPHA = 0.08
BETA = 0.40
GAMMA = 0.10
KAPPA = 1.00
SIGMA = 3.0
MAX_PX_MOVE = 1.0
SPRING_K = 0.08
VOLCANO_K = 900.0
VOLCANO_RADIUS = 20.0

# Canvas and shape. The transect is the horizontal midline: through the disk,
# the square, and a short black stroke off to the right.
H, W = 400, 560
DISK_C = np.array([140.0, 185.0])
DISK_R = 62.0
SQUARE = (178, 129, 302, 241)  # x0, y0, x1, y1
STROKE = (430, 180, 520, 190)  # x0, y0, x1, y1, centered on the transect
TRANSECT = 185


def make_image():
    image = np.full((H, W), 40, np.uint8)
    cv2.circle(image, (int(DISK_C[0]), int(DISK_C[1])), int(DISK_R), 255, -1)
    x0, y0, x1, y1 = SQUARE
    cv2.rectangle(image, (x0, y0), (x1, y1), 255, -1)
    sx0, sy0, sx1, sy1 = STROKE
    cv2.rectangle(image, (sx0, sy0), (sx1, sy1), 0, -1)
    return image


def derivative_matrices(n):
    eye = np.eye(n)
    second = np.roll(eye, -1, axis=0) + np.roll(eye, 1, axis=0) - 2.0 * eye
    fourth = (
        np.roll(eye, -2, axis=0)
        + np.roll(eye, 2, axis=0)
        - 4.0 * np.roll(eye, -1, axis=0)
        - 4.0 * np.roll(eye, 1, axis=0)
        + 6.0 * eye
    )
    return second, fourth


def image_terms(image, sigma):
    """Line, edge, and termination fields, before the weighted mix."""

    image_f = image.astype(np.float64) / 255.0
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1
    blur = cv2.GaussianBlur(image_f, (ksize, ksize), sigma)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
    grad_xx = cv2.Sobel(blur, cv2.CV_64F, 2, 0)
    grad_yy = cv2.Sobel(blur, cv2.CV_64F, 0, 2)
    grad_xy = cv2.Sobel(blur, cv2.CV_64F, 1, 1)
    eps = 1e-8
    denom = (grad_x**2 + grad_y**2 + eps) ** 1.5
    term = (
        grad_yy * grad_x**2
        - 2 * grad_xy * grad_x * grad_y
        + grad_xx * grad_y**2
    ) / denom
    edge = -(grad_x**2 + grad_y**2)
    return blur, edge, term


def overlay_snake(ax, snake, color, lw=1.2, ms=10):
    closed = np.vstack([snake, snake[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw)
    ax.scatter(
        snake[:, 0], snake[:, 1],
        s=ms, c=color, zorder=3, edgecolors="white", linewidths=0.25,
    )


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def blob_circle():
    """A circle just outside the disk-square union, missing the stroke."""

    corners = np.array([
        [SQUARE[0], SQUARE[1]],
        [SQUARE[2], SQUARE[1]],
        [SQUARE[2], SQUARE[3]],
        [SQUARE[0], SQUARE[3]],
        DISK_C + [-DISK_R, 0],
        DISK_C + [DISK_R, 0],
        DISK_C + [0, -DISK_R],
        DISK_C + [0, DISK_R],
    ])
    center = np.array([190.0, 185.0])
    radius = np.max(np.linalg.norm(corners - center, axis=1)) + 16.0
    rim = (center[0] + radius, center[1])
    return center, radius, initialize_snake([tuple(center), rim], density=0.5)


def main():
    image = make_image()
    line, edge, term = image_terms(image, SIGMA)
    print(
        "terms",
        "line", float(line.min()), float(line.max()),
        "edge", float(edge.min()), float(edge.max()),
        "term", float(term.min()), float(term.max()),
    )

    center, radius, circle = blob_circle()
    print("circle n", len(circle), "center", center, "radius", round(radius, 1))

    polyline_clicks = np.array([
        [55.0, 185.0],
        [105.0, 100.0],
        [240.0, 95.0],
        [340.0, 140.0],
        [340.0, 235.0],
        [190.0, 280.0],
        [75.0, 260.0],
    ])
    polyline = initialize_snake_from_polyline(polyline_clicks, density=0.5)
    print("polyline n", len(polyline), "clicks", len(polyline_clicks))
    print(
        "circle bounds",
        float(circle[:, 0].min()), float(circle[:, 0].max()),
        float(circle[:, 1].min()), float(circle[:, 1].max()),
        "image", W, H,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, snake, clicks, title in (
        (axes[0], circle, np.array([center, [center[0] + radius, center[1]]]), "Circle"),
        (axes[1], polyline, polyline_clicks, "Polyline"),
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
        overlay_snake(ax, snake, color="#d62728", ms=7)
        ax.scatter(
            clicks[:, 0], clicks[:, 1],
            s=42, facecolors="#ffd23f", edgecolors="black", linewidths=0.7, zorder=5,
        )
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    save(fig, "init.png")

    A = _build_stiffness_matrix(16, ALPHA, BETA)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    n = 16
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    ax = axes[0]
    closed = np.vstack([ring, ring[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color="0.75", lw=1)
    ax.scatter(ring[:, 0], ring[:, 1], s=28, c="0.55", zorder=3)
    neighbors = {
        0: "#d62728",
        1: "#1f77b4",
        15: "#1f77b4",
        2: "#7f7f7f",
        14: "#7f7f7f",
    }
    for i, color in neighbors.items():
        ax.scatter(ring[i, 0], ring[i, 1], s=70, c=color, zorder=4)
        ax.annotate(
            str(i),
            ring[i],
            textcoords="offset points",
            xytext=(8, 6) if i not in (14, 15) else (-14, -12),
            fontsize=9,
            color=color,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_axis_off()
    ax.set_title("who bead 0 is tied to")

    ax = axes[1]
    im = ax.imshow(A, cmap="coolwarm", origin="upper")
    ax.add_patch(
        plt.Rectangle((-0.5, -0.5), n, 1, fill=False, edgecolor="#d62728", lw=1.4)
    )
    ax.set_xlabel("bead $j$")
    ax.set_ylabel("bead $i$")
    ax.set_xticks([0, 1, 2, 14, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_title("row 0 of $A$ (red box)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coupling")
    fig.tight_layout()
    save(fig, "stiffness_matrix.png")

    # Uneven ring: gap centered at the top, one outward kink at the bottom.
    m = 28
    gap = 0.85
    theta = np.linspace(gap / 2, 2 * np.pi - gap / 2, m, endpoint=False) + np.pi / 2
    xy = np.column_stack([np.cos(theta), np.sin(theta)])
    kink = int(np.argmin(xy[:, 1]))
    xy[kink] *= 1.5
    D2, D4 = derivative_matrices(m)
    pull_alpha = D2 @ xy
    pull_beta = -(D4 @ xy)
    print(
        "gap closes",
        float(np.dot(pull_alpha[0], xy[-1] - xy[0])),
        "kink radial beta",
        float(np.dot(pull_beta[kink], xy[kink] / np.linalg.norm(xy[kink]))),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    for ax, pull, title in (
        (axes[0], pull_alpha, r"tension ($\alpha$): the gap closes"),
        (axes[1], pull_beta, r"bending ($\beta$): the kink flattens"),
    ):
        closed = np.vstack([xy, xy[:1]])
        mag = np.linalg.norm(pull, axis=1)
        longest = max(float(mag.max()), 1e-9)
        disp = pull * (0.7 / longest)
        ax.plot(closed[:, 0], closed[:, 1], color="0.75", lw=1)
        ax.scatter(xy[:, 0], xy[:, 1], s=18, c="0.35", zorder=3)
        for p, d, m in zip(xy, disp, mag):
            if m < 0.12 * longest:
                continue
            ax.annotate(
                "",
                xy=p + d,
                xytext=p,
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.0),
                zorder=5,
            )
        ax.scatter(xy[[0, -1], 0], xy[[0, -1], 1], s=42, c="#d62728", zorder=4)
        ax.scatter(xy[kink, 0], xy[kink, 1], s=42, c="#1f77b4", zorder=4)
        ax.set_aspect("equal")
        ax.set_xlim(-2.15, 2.15)
        ax.set_ylim(-2.35, 2.05)
        ax.set_axis_off()
        ax.set_title(title)
    fig.tight_layout()
    save(fig, "internal_forces.png")

    term_lim = 0.12
    print("term abs percentiles", np.percentile(np.abs(term), [90, 95, 98, 99]))
    print(
        "transect term disk/square/stroke",
        float(term[TRANSECT, int(DISK_C[0])]),
        float(term[TRANSECT, (SQUARE[0] + SQUARE[2]) // 2]),
        float(term[TRANSECT, STROKE[0]]),
        float(term[TRANSECT, STROKE[2]]),
    )
    fields = (
        (line, 0.0, 1.0, r"$E_{\mathrm{line}}$"),
        (edge, float(edge.min()), 0.0, r"$E_{\mathrm{edge}}$"),
        (term, -term_lim, term_lim, r"$E_{\mathrm{term}}$"),
    )
    fig, axes = plt.subplots(
        2, 3, figsize=(7.8, 5.4), gridspec_kw={"height_ratios": [1.15, 1]},
    )
    row = TRANSECT
    xs = np.arange(W)
    disk_dx = np.sqrt(max(DISK_R**2 - (row - DISK_C[1])**2, 0))
    # Outer crossings only: the disk's left edge, the square's right edge,
    # and the two ends of the stroke. The other two sides lie inside the blob.
    marks = (
        (DISK_C[0] - disk_dx, "--"),
        (SQUARE[2], ":"),
        (STROKE[0], "-."),
        (STROKE[2], "-."),
    )
    for col, (field, lo, hi, title) in enumerate(fields):
        ax = axes[0, col]
        ax.imshow(field, cmap="gray", vmin=lo, vmax=hi, origin="upper")
        ax.axhline(row, color="#e4572e", lw=0.7)
        ax.set_title(title)
        ax.set_axis_off()
        ax = axes[1, col]
        ax.plot(xs, field[row], color="black", lw=1.0)
        for x, style in marks:
            ax.axvline(x, color="0.55", lw=0.7, ls=style)
        ax.set_xlim(0, W - 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col == 0:
            ax.set_ylabel("energy")
        ax.set_xlabel("column")
    fig.tight_layout()
    save(fig, "image_energy.png")
    print(
        "transect term at corners/ends",
        "square L/R", float(term[row, SQUARE[0]]), float(term[row, SQUARE[2]]),
        "stroke L/R", float(term[row, STROKE[0]]), float(term[row, STROKE[2]]),
        "disk mid", float(term[row, int(DISK_C[0])]),
    )

    # Constraints on the initial circle. Pin sits just outside the rightmost
    # bead. The volcano sits just outside the contour, above the bottom edge.
    bead = int(np.argmax(circle[:, 0]))
    pin = circle[bead] + np.array([42.0, -28.0])
    outward = circle - center
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)
    bottom = int(np.argmax(circle[:, 1]))
    volcano = circle[bottom] + outward[bottom] * 26.0
    springs = [{
        "snake": 0,
        "i": bead,
        "other_snake": None,
        "j": None,
        "anchor": pin,
    }]
    volcanoes = [volcano]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
    overlay_snake(ax, circle, color="#d62728", ms=8)
    ax.plot(
        [circle[bead, 0], pin[0]], [circle[bead, 1], pin[1]],
        color="#1f77b4", lw=1.2, zorder=4,
    )
    ax.scatter([pin[0]], [pin[1]], s=40, c="#1f77b4", zorder=5, marker="s")
    volcano_ring = plt.Circle(
        volcano, VOLCANO_RADIUS, fill=False, ec="#ff7f0e", lw=1.2, zorder=4,
    )
    ax.add_patch(volcano_ring)
    ax.scatter([volcano[0]], [volcano[1]], s=28, c="#ff7f0e", zorder=5)
    spring_only = constraint_forces(
        0, [circle], springs, [], SPRING_K, VOLCANO_K, VOLCANO_RADIUS,
    )
    volcano_only = constraint_forces(
        0, [circle], [], volcanoes, SPRING_K, VOLCANO_K, VOLCANO_RADIUS,
    )
    def draw_force_arrows(ax, pts, vec, color, maxlen=26.0, keep=0.28):
        mag = np.linalg.norm(vec, axis=1)
        peak = float(mag.max()) if len(mag) else 0.0
        if peak < 1e-8:
            return
        for p, v, m in zip(pts, vec, mag):
            if m < keep * peak:
                continue
            d = v / peak * maxlen
            ax.annotate(
                "",
                xy=(p[0] + d[0], p[1] + d[1]),
                xytext=(p[0], p[1]),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.15),
                zorder=6,
            )

    draw_force_arrows(ax, circle, -spring_only, "#1f77b4", keep=0.2)
    draw_force_arrows(ax, circle, -volcano_only, "#ff7f0e")
    print("pin", pin, "volcano", volcano)
    ax.set_axis_off()
    fig.tight_layout(pad=0.2)
    save(fig, "constraints.png")
    print(
        "constraint |spring| max", float(np.max(np.linalg.norm(spring_only, axis=1))),
        "|volcano| max", float(np.max(np.linalg.norm(volcano_only, axis=1))),
    )

    force_field = compute_image_force_field(image, 0.0, 1.0, 0.0, SIGMA)
    inv = build_implicit_operator_inverse(len(circle), ALPHA, BETA, GAMMA)
    snake = circle.copy()
    frames = {0: snake.copy()}
    show = [0, 280, 700, 850]
    for i in range(1, show[-1] + 1):
        previous = snake
        snake = step_snake(
            snake, inv, GAMMA, KAPPA, force_field, MAX_PX_MOVE, image.shape,
        )
        if i in show or i % 100 == 0:
            move = float(np.max(np.linalg.norm(snake - previous, axis=1)))
            frames[i] = snake.copy()
            print(f"iter {i:3d}  max move {move:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for ax, step in zip(axes.ravel(), show):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
        overlay_snake(ax, frames[step], color="#d62728", lw=1.1, ms=4)
        ax.set_xlim(15, 355)
        ax.set_ylim(370, 10)
        ax.set_title(f"step {step}")
        ax.set_axis_off()
    fig.tight_layout()
    save(fig, "iteration.png")

    # Same start, different alpha, beta, blur, and image weights.
    # The notch between the disk and the square is washed out at sigma=3.
    parameter_cases = [
        (
            r"$\alpha=0.08$, $\beta=0.40$" "\n" r"$\sigma=3$, edge",
            0.08, 0.40, 0.0, 1.0, 0.0, 3.0,
        ),
        (
            r"$\alpha=0.08$, $\beta=0.02$" "\n" r"$\sigma=3$, edge",
            0.08, 0.02, 0.0, 1.0, 0.0, 3.0,
        ),
        (
            r"$\alpha=0.08$, $\beta=0.40$" "\n" r"$\sigma=1$, edge",
            0.08, 0.40, 0.0, 1.0, 0.0, 1.0,
        ),
        (
            r"$\alpha=0.08$, $\beta=0.02$" "\n" r"$\sigma=1$, edge",
            0.08, 0.02, 0.0, 1.0, 0.0, 1.0,
        ),
        (
            r"$\alpha=0.40$, $\beta=0.02$" "\n" r"$\sigma=1$, edge",
            0.40, 0.02, 0.0, 1.0, 0.0, 1.0,
        ),
        (
            r"$\alpha=0.08$, $\beta=0.02$" "\n" r"$\sigma=1$, edge+term",
            0.08, 0.02, 0.0, 1.0, 1.0, 1.0,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.8))
    for ax, (label, alpha, beta, w_line, w_edge, w_term, sigma) in zip(
        axes.ravel(), parameter_cases,
    ):
        force = compute_image_force_field(image, w_line, w_edge, w_term, sigma)
        inv_case = build_implicit_operator_inverse(len(circle), alpha, beta, GAMMA)
        snake = circle.copy()
        for i in range(1, 1001):
            previous = snake
            snake = step_snake(
                snake, inv_case, GAMMA, KAPPA, force, MAX_PX_MOVE, image.shape,
            )
            if i > 400 and np.max(np.linalg.norm(snake - previous, axis=1)) < 0.02:
                break
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, origin="upper")
        overlay_snake(ax, snake, color="#d62728", lw=1.05, ms=3)
        ax.set_xlim(55, 340)
        ax.set_ylim(290, 80)
        ax.set_title(label, fontsize=9)
        ax.set_axis_off()
    fig.tight_layout()
    save(fig, "parameters.png")
    print("done")


if __name__ == "__main__":
    main()
