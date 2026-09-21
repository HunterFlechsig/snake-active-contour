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
    evolve_snake,
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
    AUTO_SIGMAS = (5.0, 3.0, 1.5)
    AUTO_MAX_ITERATIONS = 800
    AUTO_CONVERGENCE = 0.08
    AUTO_RESAMPLE_EVERY = 25

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
    finished = False
    iteration = 0

    def draw_frame(current_snake, current_sigma, current_paused, current_finished):
        status = format_status(
            iteration,
            current_sigma,
            len(pit.springs),
            len(pit.all_volcanoes()),
            current_paused,
            current_finished,
        )
        display = render_overlay(
            image,
            current_snake,
            springs=pit.springs,
            volcanoes=pit.all_volcanoes(),
            volcano_radius=VOLCANO_RADIUS,
            buttons=buttons,
            active_button_ids=pit.active_button_ids(),
            help_lines=[
                "Auto finish settles the snake then stops  |  Place volcano pins  |  Mouse volcano follows",
                status,
            ],
        )
        redraw_image(display)

    def run_auto_finish(current_snake):
        nonlocal iteration
        pit.auto_requested = False
        pit.auto_running = True
        pit.mouse_volcano_on = False
        pit.volcano_live = False
        pit.clear_springs()
        print("Auto finish: running until the snake settles (press q to abort).")

        def on_iteration(total_iterations, evolving_snake, sigma, displacement):
            nonlocal iteration
            iteration = start_iteration + total_iterations
            holder["snake"] = evolving_snake
            draw_frame(evolving_snake, sigma, False, False)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if total_iterations == 1 or total_iterations % 25 == 0:
                print(
                    f"auto iter {total_iterations:4d}  sigma={sigma:.1f}  "
                    f"max move={displacement:.4f} px"
                )
            return True

        start_iteration = iteration
        current_snake, extra = evolve_snake(
            current_snake,
            image,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            kappa=KAPPA,
            w_line=W_LINE,
            w_edge=W_EDGE,
            w_term=W_TERM,
            sigmas=AUTO_SIGMAS,
            max_iterations=AUTO_MAX_ITERATIONS,
            convergence=AUTO_CONVERGENCE,
            max_px_move=MAX_PX_MOVE,
            resample_every=AUTO_RESAMPLE_EVERY,
            on_iteration=on_iteration,
        )
        iteration = start_iteration + extra
        pit.auto_running = False
        print(f"Auto finish done after {extra} iterations.")
        return current_snake

    while True:
        key = cv2.waitKey(1) & 0xFF
        should_quit, paused, SIGMA, rebuild_forces, start_auto = handle_key(
            key, pit, snake, paused, SIGMA,
        )
        if should_quit:
            break
        if rebuild_forces:
            force_field = compute_image_force_field(
                image, W_LINE, W_EDGE, W_TERM, SIGMA,
            )

        if pit.auto_requested or start_auto:
            snake = run_auto_finish(snake)
            holder["snake"] = snake
            paused = True
            finished = True
            inv_operator = build_implicit_operator_inverse(
                len(snake), ALPHA, BETA, GAMMA,
            )
            force_field = compute_image_force_field(
                image, W_LINE, W_EDGE, W_TERM, SIGMA,
            )

        pit.sync()

        if not paused:
            finished = False
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

        draw_frame(snake, SIGMA, paused, finished)

    close_window()


if __name__ == "__main__":
    main()
