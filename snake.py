"""Load the image, wire the modules, run the live loop, quit."""

import cv2

from display import (
    close_window,
    display_image,
    format_status,
    redraw_image,
    render_overlay,
)
from interaction import (
    SnakePit,
    get_initial_contour,
    handle_key,
    install_pit_mouse,
    make_pit_buttons,
    print_runtime_help,
)
from physics import (
    build_implicit_operator_inverse,
    compute_image_force_field,
    initialize_snake,
    initialize_snake_from_polyline,
    step_snake,
)


def load_image():
    return cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE)


def main():

    image = load_image()

    ALPHA = 0.08
    BETA = 0.40
    GAMMA = 0.10
    KAPPA = 1.00

    W_LINE = 0.0
    W_EDGE = 1.0
    W_TERM = 0.0

    SIGMA = 3.0
    MAX_PX_MOVE = 1.0
    SPRING_K = 0.08
    VOLCANO_K = 900.0
    VOLCANO_RADIUS = 20.0

    display_image(image)

    points, mode = get_initial_contour(image)
    if points is None:
        close_window()
        return

    if mode == "circle":
        snake = initialize_snake(points)
    else:
        snake = initialize_snake_from_polyline(points)

    pit = SnakePit()
    holder = {"snake": snake}
    buttons = make_pit_buttons(image.shape)
    install_pit_mouse(pit, holder, buttons)

    print_runtime_help()

    force_field = compute_image_force_field(
        image, W_LINE, W_EDGE, W_TERM, SIGMA,
    )
    inv_operator = build_implicit_operator_inverse(len(snake), ALPHA, BETA, GAMMA)
    paused = False
    iteration = 0

    while True:
        key = cv2.waitKey(1) & 0xFF
        should_quit, paused, SIGMA, rebuild_forces = handle_key(
            key, pit, snake, paused, SIGMA,
        )
        if should_quit:
            break
        if rebuild_forces:
            force_field = compute_image_force_field(
                image, W_LINE, W_EDGE, W_TERM, SIGMA,
            )

        pit.sync()

        if not paused:
            snake = step_snake(
                snake,
                inv_operator,
                GAMMA,
                KAPPA,
                force_field,
                MAX_PX_MOVE,
                image.shape,
                pit.springs,
                pit.all_volcanoes(),
                SPRING_K,
                VOLCANO_K,
                VOLCANO_RADIUS,
            )
            holder["snake"] = snake
            iteration += 1

        status = format_status(
            iteration,
            SIGMA,
            len(pit.springs),
            len(pit.all_volcanoes()),
            paused,
        )
        display = render_overlay(
            image,
            snake,
            springs=pit.springs,
            volcanoes=pit.all_volcanoes(),
            volcano_radius=VOLCANO_RADIUS,
            buttons=buttons,
            active_button_ids=pit.active_button_ids(),
            help_lines=[
                "Place volcano, then click to pin  |  Mouse volcano follows cursor  |  R-drag also follows",
                status,
            ],
        )
        redraw_image(display)

    close_window()


if __name__ == "__main__":
    main()
