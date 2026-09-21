"""Kass snake: init, stiffness, image energy, constraints, update, resample, clip."""

import math
from collections import deque

import cv2
import numpy as np


def initialize_snake(initial_circle, density=0.5):

    center = initial_circle[0]
    radius = math.dist(initial_circle[0], initial_circle[1])

    num_points = max(8, math.ceil(math.pi * radius * density))

    snake = []

    for i in range(num_points):

        angle = (2 * math.pi * i) / num_points

        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)

        snake.append([x, y])

    return np.array(snake, dtype=np.float64)


def initialize_snake_from_polyline(points, density=0.5):
    """Densify a user-clicked closed outline into a working snake."""

    pts = np.asarray(points, dtype=np.float64)
    closed = np.vstack([pts, pts[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]

    n = max(len(pts), int(math.ceil(perimeter * density / 2.0)))
    samples = np.linspace(0.0, perimeter, n, endpoint=False)
    x = np.interp(samples, cumulative, closed[:, 0])
    y = np.interp(samples, cumulative, closed[:, 1])

    return np.column_stack([x, y])


def _build_stiffness_matrix(n, alpha, beta):
    """Circulant pentadiagonal matrix A for a closed snake, h = 1.

    A v is the gradient of discrete internal energy, matching Kass:
    A = -α D2 + β D4, with periodic wrapping.
    """

    eye = np.eye(n)

    second_derivative = (
        np.roll(eye, -1, axis=0)
        + np.roll(eye, 1, axis=0)
        - 2.0 * eye
    )

    fourth_derivative = (
        np.roll(eye, -2, axis=0)
        + np.roll(eye, 2, axis=0)
        - 4.0 * np.roll(eye, -1, axis=0)
        - 4.0 * np.roll(eye, 1, axis=0)
        + 6.0 * eye
    )

    return -alpha * second_derivative + beta * fourth_derivative


def build_implicit_operator_inverse(n, alpha, beta, gamma):
    stiffness = _build_stiffness_matrix(n, alpha, beta)
    return np.linalg.inv(stiffness + gamma * np.eye(n))


def compute_image_energy(image, w_line, w_edge, w_term, sigma=1.0):

    image_f = image.astype(np.float64) / 255.0

    ksize = max(3, int(2 * round(3 * sigma) + 1))
    if ksize % 2 == 0:
        ksize += 1

    blur = cv2.GaussianBlur(image_f, (ksize, ksize), sigma)

    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1)

    line_energy = blur

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

    max_abs = np.max(np.abs(energy))
    if max_abs > 0:
        energy = energy / max_abs

    return energy


def compute_image_force_field(image, w_line, w_edge, w_term, sigma=1.0):

    energy = compute_image_energy(
        image,
        w_line,
        w_edge,
        w_term,
        sigma,
    )

    force_y, force_x = np.gradient(energy)
    return force_x, force_y


def _sample_bilinear(field, snake):

    height, width = field.shape

    x = np.clip(snake[:, 0], 0.0, width - 1.0)
    y = np.clip(snake[:, 1], 0.0, height - 1.0)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = x - x0
    dy = y - y0

    top = field[y0, x0] * (1.0 - dx) + field[y0, x1] * dx
    bottom = field[y1, x0] * (1.0 - dx) + field[y1, x1] * dx

    return top * (1.0 - dy) + bottom * dy


def sample_image_forces(force_field, snake):

    force_x, force_y = force_field

    return np.column_stack([
        _sample_bilinear(force_x, snake),
        _sample_bilinear(force_y, snake),
    ])


def constraint_forces(snake, springs, volcanoes, spring_k, volcano_k, volcano_radius):
    """∇E_con from Snake Pit springs and the clipped 1/r volcano.

    Spring energy is (k/2)|x1 - x2|^2. Volcano energy is k / max(r, r0),
    which is high at the mouse and therefore repulsive.
    """

    forces = np.zeros_like(snake)

    for spring in springs:
        i = spring["i"]
        if spring.get("j") is not None:
            delta = snake[i] - snake[spring["j"]]
            forces[i] += spring_k * delta
            forces[spring["j"]] -= spring_k * delta
        else:
            forces[i] += spring_k * (snake[i] - spring["anchor"])

    for volcano in volcanoes:
        delta = snake - np.asarray(volcano, dtype=np.float64)
        radius = np.linalg.norm(delta, axis=1, keepdims=True)
        outside = radius > volcano_radius
        radius_safe = np.maximum(radius, 1e-6)
        volcano_grad = -volcano_k * delta / (radius_safe ** 3)
        forces += np.where(outside, volcano_grad, 0.0)

    return forces


def update_snake(snake, inv_operator, gamma, kappa, image_forces, max_px_move):
    """Kass eqs. (19)–(20): v_t = (A + γI)^{-1} (γ v_{t-1} - κ ∇E)."""

    predicted = inv_operator @ (gamma * snake - kappa * image_forces)
    delta = predicted - snake

    return snake + max_px_move * np.tanh(delta)


def clip_snake_to_image(snake, image_shape):

    height, width = image_shape[:2]
    clipped = snake.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, width - 1.0)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, height - 1.0)
    return clipped


def resample_snake(snake):
    """Re-space points uniformly along the closed contour; keep count fixed."""

    n = len(snake)
    closed = np.vstack([snake, snake[:1]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    perimeter = cumulative[-1]

    if perimeter < 1e-6:
        return snake.copy()

    samples = np.linspace(0.0, perimeter, n, endpoint=False)
    x = np.interp(samples, cumulative, closed[:, 0])
    y = np.interp(samples, cumulative, closed[:, 1])

    return np.column_stack([x, y])


def has_converged(history, snake, threshold):

    if len(history) < history.maxlen:
        return False

    distances = [
        np.max(np.abs(previous - snake))
        for previous in history
    ]

    return min(distances) < threshold


def step_snake(
    snake,
    inv_operator,
    gamma,
    kappa,
    force_field,
    max_px_move,
    image_shape,
    springs=(),
    volcanoes=(),
    spring_k=0.0,
    volcano_k=0.0,
    volcano_radius=20.0,
):
    """One implicit Euler step: sample ∇E, add constraints, update, clip."""

    image_forces = sample_image_forces(force_field, snake)
    extra_forces = constraint_forces(
        snake,
        springs,
        volcanoes,
        spring_k,
        volcano_k,
        volcano_radius,
    )
    total_forces = kappa * image_forces + extra_forces
    updated = update_snake(
        snake,
        inv_operator,
        gamma,
        1.0,
        total_forces,
        max_px_move,
    )
    return clip_snake_to_image(updated, image_shape)


def evolve_snake(
    snake,
    image,
    alpha,
    beta,
    gamma,
    kappa,
    w_line,
    w_edge,
    w_term,
    sigmas,
    max_iterations,
    convergence,
    max_px_move,
    resample_every,
    on_iteration=None,
):

    snake = snake.astype(np.float64, copy=True)
    inv_operator = build_implicit_operator_inverse(len(snake), alpha, beta, gamma)

    total_iterations = 0

    for sigma in sigmas:

        force_field = compute_image_force_field(
            image,
            w_line,
            w_edge,
            w_term,
            sigma,
        )

        snake = resample_snake(snake)
        history = deque(maxlen=10)

        for iteration in range(max_iterations):

            previous = snake
            snake = step_snake(
                snake,
                inv_operator,
                gamma,
                kappa,
                force_field,
                max_px_move,
                image.shape,
            )

            displacement = np.max(np.linalg.norm(snake - previous, axis=1))
            total_iterations += 1

            if on_iteration is not None:
                on_iteration(total_iterations, snake, sigma, displacement)

            if has_converged(history, snake, convergence):
                snake = resample_snake(snake)
                break

            history.append(snake.copy())

            if resample_every and (iteration + 1) % resample_every == 0:
                snake = resample_snake(snake)
                history.clear()

    return snake, total_iterations
