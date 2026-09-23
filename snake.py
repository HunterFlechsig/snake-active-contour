"""Load the image, wire the modules, run the live loop, quit."""

import argparse
from pathlib import Path

import cv2

import defaults
from display import (
    close_window,
    display_image,
    format_status,
    redraw_image,
    render_overlay,
    save_snapshot,
)
from interaction import (
    SnakePit,
    collect_snakes,
    handle_key,
    install_pit_mouse,
    print_runtime_help,
)
from physics import (
    build_implicit_operator_inverse,
    compute_image_force_field,
    evolve_snakes,
    initialize_snake,
    step_all_snakes,
)

SNAPSHOT_EVERY = 25
IMAGES_DIR = Path("images")
DEFAULT_IMAGE = "coins.jpg"


def parse_circle(value):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "circle must be CX,CY,RX,RY (center and a rim point)."
        )
    cx, cy, rx, ry = (float(part) for part in parts)
    return (cx, cy), (rx, ry)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Kass snakes: click to place contours, then evolve. "
            "Hyperparameter defaults match the report live program."
        ),
    )
    parser.add_argument(
        "--test",
        type=int,
        default=1,
        metavar="N",
        help="Test number. Snapshots go to results/test_N/ (default: 1).",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        help="After placing snakes, run until they settle before the live loop.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=SNAPSHOT_EVERY,
        metavar="K",
        help=(
            f"Save an overlay every K iterations into results/test_N/ "
            f"(default: {SNAPSHOT_EVERY}; 0 disables)."
        ),
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        metavar="FILE",
        help=(
            f"Image filename or path. Bare names are loaded from {IMAGES_DIR}/ "
            f"(default: {DEFAULT_IMAGE})."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=defaults.ALPHA,
        help=f"Tension weight (default: {defaults.ALPHA}).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=defaults.BETA,
        help=f"Bending weight (default: {defaults.BETA}).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=defaults.GAMMA,
        help=f"Implicit stay-put weight (default: {defaults.GAMMA}).",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=defaults.KAPPA,
        help=f"Image-force scale (default: {defaults.KAPPA}).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=defaults.SIGMA,
        help=f"Gaussian blur width (default: {defaults.SIGMA}).",
    )
    parser.add_argument(
        "--w-line",
        type=float,
        default=defaults.W_LINE,
        help=f"Line-energy weight (default: {defaults.W_LINE}).",
    )
    parser.add_argument(
        "--w-edge",
        type=float,
        default=defaults.W_EDGE,
        help=f"Edge-energy weight (default: {defaults.W_EDGE}).",
    )
    parser.add_argument(
        "--w-term",
        type=float,
        default=defaults.W_TERM,
        help=f"Termination-energy weight (default: {defaults.W_TERM}).",
    )
    parser.add_argument(
        "--skimage",
        action="store_true",
        help=(
            "Use scikit-image astronaut-example α, β, γ "
            f"({defaults.SKIMAGE_ALPHA}, {defaults.SKIMAGE_BETA}, "
            f"{defaults.SKIMAGE_GAMMA}); other weights stay at defaults."
        ),
    )
    parser.add_argument(
        "--circle",
        type=parse_circle,
        metavar="CX,CY,RX,RY",
        help="Skip clicking: build one closed snake from a center and a rim point.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No window. Requires --circle. Runs --steps (or --converge) and exits.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Headless iteration cap at the current σ (default: auto-finish max).",
    )
    args = parser.parse_args()
    if args.skimage:
        args.alpha = defaults.SKIMAGE_ALPHA
        args.beta = defaults.SKIMAGE_BETA
        args.gamma = defaults.SKIMAGE_GAMMA
    if args.headless and args.circle is None:
        parser.error("--headless requires --circle.")
    if args.steps is None:
        args.steps = defaults.AUTO_MAX_ITERATIONS
    return args


def resolve_image_path(name):
    candidate = Path(name)
    if candidate.exists():
        return candidate

    in_images = IMAGES_DIR / name
    if in_images.exists():
        return in_images

    raise SystemExit(
        f"Image not found: {name}\n"
        f"Looked at {candidate.resolve()} and {in_images.resolve()}."
    )


def results_dir(test_number):
    path = Path("results") / f"test_{test_number}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def build_operators(snakes, alpha, beta, gamma):
    return [
        build_implicit_operator_inverse(len(snake), alpha, beta, gamma)
        for snake in snakes
    ]


def hyperparam_lines(args):
    return [
        f"alpha={args.alpha}",
        f"beta={args.beta}",
        f"gamma={args.gamma}",
        f"kappa={args.kappa}",
        f"sigma={args.sigma}",
        f"w_line={args.w_line}",
        f"w_edge={args.w_edge}",
        f"w_term={args.w_term}",
        f"skimage={args.skimage}",
    ]


def write_run_notes(folder, args, extra_lines):
    notes = folder / "run.txt"
    notes.write_text("\n".join([
        f"test={args.test}",
        f"image={args.image}",
        f"converge={args.converge}",
        f"snapshot_every={args.snapshot_every}",
        f"headless={args.headless}",
        *hyperparam_lines(args),
        *extra_lines,
        "",
    ]))


def main():

    args = parse_args()
    out_dir = results_dir(args.test)
    image_path = resolve_image_path(args.image)

    image = load_image(image_path)

    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    kappa = args.kappa
    w_line = args.w_line
    w_edge = args.w_edge
    w_term = args.w_term
    sigma = args.sigma
    max_px_move = defaults.MAX_PX_MOVE
    spring_k = defaults.SPRING_K
    volcano_k = defaults.VOLCANO_K
    volcano_radius = defaults.VOLCANO_RADIUS
    auto_sigmas = defaults.AUTO_SIGMAS
    auto_max_iterations = defaults.AUTO_MAX_ITERATIONS
    auto_convergence = defaults.AUTO_CONVERGENCE

    if args.circle is not None:
        snakes = [initialize_snake(list(args.circle), density=defaults.DENSITY)]
    else:
        if args.headless:
            raise SystemExit("--headless requires --circle.")
        display_image(image)
        snakes = collect_snakes(image)
        if not snakes:
            close_window()
            return

    if not args.headless and args.circle is not None:
        display_image(image)

    pit = SnakePit()
    holder = {"snakes": snakes}
    if not args.headless:
        install_pit_mouse(pit, holder)

    print(f"Starting with {len(snakes)} snake(s).")
    print(f"Image: {image_path}")
    print(f"Results folder: {out_dir}")
    print(
        f"α={alpha}  β={beta}  γ={gamma}  κ={kappa}  "
        f"σ={sigma}  w_line={w_line}  w_edge={w_edge}  w_term={w_term}"
    )
    if args.converge:
        print("Converge-first: snakes will settle before the live loop.")
    if args.snapshot_every > 0:
        print(f"Saving a snapshot every {args.snapshot_every} iterations.")
    if not args.headless:
        print_runtime_help()

    force_field = compute_image_force_field(
        image, w_line, w_edge, w_term, sigma,
    )
    operators = build_operators(snakes, alpha, beta, gamma)
    paused = False
    finished = False
    iteration = 0

    def draw_frame(current_snakes):
        if args.headless:
            return
        display = render_overlay(
            image,
            snakes=current_snakes,
            springs=pit.springs,
            volcanoes=pit.all_volcanoes(),
            volcano_radius=volcano_radius,
        )
        redraw_image(display)

    def maybe_snapshot(current_snakes, step, tag=None):
        if args.snapshot_every <= 0:
            return
        if tag is None and step != 0 and step % args.snapshot_every != 0:
            return
        name = tag if tag else f"iter_{step:04d}"
        path = out_dir / f"{name}.png"
        save_snapshot(
            path,
            image,
            current_snakes,
            springs=pit.springs,
            volcanoes=pit.all_volcanoes(),
            volcano_radius=volcano_radius,
        )
        print(f"Saved {path}")

    def print_status(current_sigma, current_paused, current_finished):
        print(
            format_status(
                iteration,
                current_sigma,
                len(holder["snakes"]),
                len(pit.springs),
                len(pit.all_volcanoes()),
                current_paused,
                current_finished,
            )
        )

    def run_auto_finish(current_snakes):
        nonlocal iteration
        pit.auto_requested = False
        pit.auto_running = True
        pit.mouse_volcano_on = False
        pit.volcano_live = False
        pit.clear_springs()
        print("Auto finish: running until every snake settles (q aborts).")

        def on_iteration(total_iterations, evolving_snakes, current_sigma, displacement):
            nonlocal iteration
            iteration = start_iteration + total_iterations
            holder["snakes"] = evolving_snakes
            draw_frame(evolving_snakes)
            maybe_snapshot(evolving_snakes, iteration)
            if args.headless:
                if total_iterations == 1 or total_iterations % 25 == 0:
                    print(
                        f"auto iter {total_iterations:4d}  sigma={current_sigma:.1f}  "
                        f"max move={displacement:.4f} px"
                    )
                return True
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if total_iterations == 1 or total_iterations % 25 == 0:
                print(
                    f"auto iter {total_iterations:4d}  sigma={current_sigma:.1f}  "
                    f"max move={displacement:.4f} px"
                )
            return True

        start_iteration = iteration
        current_snakes, extra = evolve_snakes(
            current_snakes,
            image,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            kappa=kappa,
            w_line=w_line,
            w_edge=w_edge,
            w_term=w_term,
            sigmas=auto_sigmas,
            max_iterations=auto_max_iterations,
            convergence=auto_convergence,
            max_px_move=max_px_move,
            on_iteration=on_iteration,
        )
        iteration = start_iteration + extra
        pit.auto_running = False
        maybe_snapshot(current_snakes, iteration, tag="iter_final")
        print(f"Auto finish done after {extra} iterations.")
        return current_snakes

    maybe_snapshot(snakes, 0, tag="iter_0000")

    if args.converge:
        snakes = run_auto_finish(snakes)
        holder["snakes"] = snakes
        operators = build_operators(snakes, alpha, beta, gamma)
        force_field = compute_image_force_field(
            image, w_line, w_edge, w_term, sigma,
        )
        paused = True
        finished = True
        print_status(sigma, paused, finished)
        if not args.headless:
            print("Converge-first finished. Live loop is paused — space to resume, q to quit.")

    if args.headless and not args.converge:
        print(f"Headless: {args.steps} steps at σ={sigma}.")
        for _ in range(args.steps):
            snakes = step_all_snakes(
                snakes,
                operators,
                gamma,
                kappa,
                force_field,
                max_px_move,
                image.shape,
            )
            holder["snakes"] = snakes
            iteration += 1
            maybe_snapshot(snakes, iteration)
            if iteration == 1 or iteration % 50 == 0:
                print_status(sigma, False, False)
        finished = True
        maybe_snapshot(snakes, iteration, tag="iter_final")
        write_run_notes(
            out_dir,
            args,
            [
                f"n_snakes={len(snakes)}",
                f"final_iteration={iteration}",
                f"finished={finished}",
            ],
        )
        print(f"Headless done after {iteration} iterations.")
        return

    if args.headless:
        write_run_notes(
            out_dir,
            args,
            [
                f"n_snakes={len(snakes)}",
                f"final_iteration={iteration}",
                f"finished={finished}",
            ],
        )
        return

    write_run_notes(
        out_dir,
        args,
        [
            f"n_snakes={len(snakes)}",
            *hyperparam_lines(args),
        ],
    )

    while True:
        key = cv2.waitKey(1) & 0xFF
        should_quit, paused, sigma, rebuild_forces, start_auto = handle_key(
            key, pit, snakes, paused, sigma,
        )
        if should_quit:
            break
        if rebuild_forces:
            force_field = compute_image_force_field(
                image, w_line, w_edge, w_term, sigma,
            )

        if pit.auto_requested or start_auto:
            snakes = run_auto_finish(snakes)
            holder["snakes"] = snakes
            operators = build_operators(snakes, alpha, beta, gamma)
            force_field = compute_image_force_field(
                image, w_line, w_edge, w_term, sigma,
            )
            paused = True
            finished = True
            print_status(sigma, paused, finished)

        pit.sync()

        if not paused:
            finished = False
            snakes = step_all_snakes(
                snakes,
                operators,
                gamma,
                kappa,
                force_field,
                max_px_move,
                image.shape,
                pit.springs,
                pit.all_volcanoes(),
                spring_k,
                volcano_k,
                volcano_radius,
            )
            holder["snakes"] = snakes
            iteration += 1
            maybe_snapshot(snakes, iteration)
            if iteration == 1 or iteration % 50 == 0:
                print_status(sigma, paused, finished)

        draw_frame(snakes)

    maybe_snapshot(snakes, iteration, tag="iter_final")
    write_run_notes(
        out_dir,
        args,
        [
            f"n_snakes={len(snakes)}",
            *hyperparam_lines(args),
            f"final_iteration={iteration}",
            f"finished={finished}",
        ],
    )
    close_window()


if __name__ == "__main__":
    main()
