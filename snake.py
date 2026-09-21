import cv2
import math
import numpy as np

from display_helpers import *


# =====================================================
# Snake Initialization
# =====================================================

def initialize_snake(initial_circle):

    DENSITY = 0.5

    center = initial_circle[0]
    radius = math.dist(initial_circle[0], initial_circle[1])

    num_points = math.ceil(math.pi * radius * DENSITY)

    snake = []

    for i in range(num_points):

        angle = (2 * math.pi * i) / num_points

        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)

        snake.append([x, y])

    return np.array(snake, dtype=np.float64)


# =====================================================
# Internal Forces
# =====================================================

def compute_internal_forces(snake, alpha, beta):

    n = len(snake)

    def get_neighbors(i):

        return (
            snake[(i - 2) % n],
            snake[(i - 1) % n],
            snake[i],
            snake[(i + 1) % n],
            snake[(i + 2) % n],
        )

    def compute_tension(before, current, after):

        return alpha * (
            before
            - 2 * current
            + after
        )

    def compute_bending(before_before, before, current, after, after_after):

        return -beta * (
            before_before
            - 4 * before
            + 6 * current
            - 4 * after
            + after_after
        )

    forces = []

    for i in range(n):

        before_before, before, current, after, after_after = get_neighbors(i)

        tension = compute_tension(before, current, after)

        bending = compute_bending(
            before_before,
            before,
            current,
            after,
            after_after,
        )

        forces.append(tension + bending)

    return np.array(forces)


# =====================================================
# Image Energy
# =====================================================

def compute_image_energy(image, w_line, w_edge, w_term):

    blur = cv2.GaussianBlur(image, (5, 5), 0)

    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

    line_energy = blur.astype(np.float64)

    edge_energy = -(grad_x**2 + grad_y**2)

    grad_xx = cv2.Sobel(blur, cv2.CV_64F, 2, 0)
    grad_yy = cv2.Sobel(blur, cv2.CV_64F, 0, 2)
    grad_xy = cv2.Sobel(blur, cv2.CV_64F, 1, 1)

    eps = 1e-8

    denom = (grad_x**2 + grad_y**2 + eps) ** 1.5

    term_energy = (
        grad_yy * grad_x**2
        - 2 * grad_xy * grad_x * grad_y
        + grad_xx * grad_y**2
    ) / denom

    energy = (
        w_line * line_energy
        + w_edge * edge_energy
        + w_term * term_energy
    )

    return energy


# =====================================================
# Image Force Field
# =====================================================

def compute_image_force_field(image, w_line, w_edge, w_term):

    energy = compute_image_energy(
        image,
        w_line,
        w_edge,
        w_term,
    )

    force_x = -cv2.Sobel(energy, cv2.CV_64F, 1, 0)
    force_y = -cv2.Sobel(energy, cv2.CV_64F, 0, 1)

    return force_x, force_y


# =====================================================
# External Forces
# =====================================================

def compute_external_forces(force_field, snake):

    force_x, force_y = force_field

    forces = []

    height, width = force_x.shape

    for point in snake:

        x = int(round(point[0]))
        y = int(round(point[1]))

        # Clamp to image bounds
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        forces.append([
            force_x[y, x],
            force_y[y, x],
        ])

    return np.array(forces)


# =====================================================
# Snake Update
# =====================================================

def update_snake(snake, internal_forces, external_forces, step_size):

    return snake + step_size * (
        internal_forces + external_forces
    )


# =====================================================
# Main
# =====================================================

def main():

    image = load_image()

    ALPHA = 1.0
    BETA = 1.0

    W_LINE = 1.0
    W_EDGE = 1.0
    W_TERM = 1.0

    STEP_SIZE = 0.1

    force_field = compute_image_force_field(
        image,
        W_LINE,
        W_EDGE,
        W_TERM,
    )

    display_image(image)

    initial_circle = get_points(image)

    snake = initialize_snake(initial_circle)

    display_snake(snake, image)

    # TODO:
    # Replace with convergence criterion
    for _ in range(200):

        internal = compute_internal_forces(
            snake,
            ALPHA,
            BETA,
        )

        external = compute_external_forces(
            force_field,
            snake,
        )

        snake = update_snake(
            snake,
            internal,
            external,
            STEP_SIZE,
        )

        display_snake(snake, image)

    cv2.waitKey(0)
    close_window()


if __name__ == "__main__":
    main()