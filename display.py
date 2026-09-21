"""Window, scale, overlay drawing, and colors."""

import cv2
import numpy as np

WINDOW_NAME = "Image"
DISPLAY_SCALE = 2.5

# BGR: one color per snake, then orange springs / magenta volcano
SNAKE_COLORS = [
    (0, 0, 255),
    (0, 180, 0),
    (255, 128, 0),
    (0, 220, 255),
    (255, 0, 180),
    (180, 80, 0),
]
SPRING_COLOR = (0, 140, 255)
VOLCANO_COLOR = (255, 0, 220)
SNAKE_COLOR = SNAKE_COLORS[0]
PREVIEW_CLOSE_COLOR = (0, 255, 255)


def get_int_point(point):
    return (int(round(point[0])), int(round(point[1])))


def window_to_image_xy(x, y):
    return (
        int(round(x / DISPLAY_SCALE)),
        int(round(y / DISPLAY_SCALE)),
    )


def bind_mouse(on_event):
    def _callback(event, x, y, flags, param):
        ix, iy = window_to_image_xy(x, y)
        on_event(event, ix, iy, flags, param)

    cv2.setMouseCallback(WINDOW_NAME, _callback)


def display_image(image):
    height, width = image.shape[:2]
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        int(width * DISPLAY_SCALE),
        int(height * DISPLAY_SCALE),
    )
    redraw_image(_gray_to_bgr(image) if len(image.shape) == 2 else image)
    cv2.waitKey(1)


def redraw_image(image):
    height, width = image.shape[:2]
    scaled = cv2.resize(
        image,
        (int(width * DISPLAY_SCALE), int(height * DISPLAY_SCALE)),
        interpolation=cv2.INTER_LINEAR,
    )
    cv2.imshow(WINDOW_NAME, scaled)


def close_window():
    cv2.destroyWindow(WINDOW_NAME)


def _gray_to_bgr(image):
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def display_snake(snake, image):
    display = render_overlay(image, snake=snake)
    redraw_image(display)


def save_snapshot(path, image, snakes, springs=None, volcanoes=None, volcano_radius=20):
    overlay = render_overlay(
        image,
        snakes=snakes,
        springs=springs,
        volcanoes=volcanoes,
        volcano_radius=volcano_radius,
    )
    cv2.imwrite(str(path), overlay)


def format_status(iteration, sigma, n_snakes, n_springs, n_volcanoes, paused, finished=False):
    state = "done" if finished else ("paused" if paused else "running")
    return (
        f"{n_snakes} snake(s)  iter {iteration}  sigma={sigma:.1f}  "
        f"springs={n_springs}  volcanoes={n_volcanoes}  {state}"
    )


def snake_color(index):
    return SNAKE_COLORS[index % len(SNAKE_COLORS)]


def _draw_snake(display, snake, color):
    if snake is None or len(snake) == 0:
        return
    for i, point in enumerate(snake):
        next_point = snake[(i + 1) % len(snake)]
        cv2.circle(display, get_int_point(point), 1, color, -1)
        cv2.line(
            display,
            get_int_point(point),
            get_int_point(next_point),
            color,
            1,
        )


def render_overlay(
    image,
    snakes=None,
    snake=None,
    springs=None,
    volcano=None,
    volcanoes=None,
    volcano_radius=20,
):
    display = _gray_to_bgr(image)

    if snakes is None:
        snakes = [] if snake is None else [snake]

    for index, contour in enumerate(snakes):
        _draw_snake(display, contour, snake_color(index))

    if springs:
        for spring in springs:
            start_snake = snakes[spring.get("snake", 0)]
            start = get_int_point(start_snake[spring["i"]])
            if spring.get("other_snake") is not None:
                end_snake = snakes[spring["other_snake"]]
                end = get_int_point(end_snake[spring["j"]])
            elif spring.get("j") is not None:
                end = get_int_point(start_snake[spring["j"]])
            else:
                end = get_int_point(spring["anchor"])
            cv2.line(display, start, end, SPRING_COLOR, 2)
            cv2.circle(display, start, 5, SPRING_COLOR, 2)
            cv2.circle(display, end, 5, SPRING_COLOR, -1)

    volcanoes = volcanoes or []
    if volcano is not None:
        volcanoes = [*volcanoes, volcano]
    for center in volcanoes:
        _draw_volcano(display, get_int_point(center), volcano_radius)

    return display


def render_contour_preview(image, points, mode, existing_snakes=None):
    display = render_overlay(image, snakes=existing_snakes or [])
    preview = np.array(points, dtype=np.float64) if points else None
    if preview is None or len(preview) == 0:
        return display

    color = snake_color(len(existing_snakes or []))
    if len(preview) >= 2:
        for i in range(len(preview) - 1):
            cv2.line(
                display,
                get_int_point(preview[i]),
                get_int_point(preview[i + 1]),
                color,
                1,
            )
        if mode == "polyline" and len(preview) >= 3:
            cv2.line(
                display,
                get_int_point(preview[-1]),
                get_int_point(preview[0]),
                PREVIEW_CLOSE_COLOR,
                1,
            )
    for point in preview:
        cv2.circle(display, get_int_point(point), 4, color, -1)
    return display


def _draw_volcano(display, center, volcano_radius):
    crater = max(4, int(round(volcano_radius)))
    skirt = max(crater + 1, int(round(volcano_radius * 2.5)))
    cv2.circle(display, center, skirt, VOLCANO_COLOR, 2)
    cv2.circle(display, center, crater, VOLCANO_COLOR, 2)
    cv2.circle(display, center, 4, VOLCANO_COLOR, -1)
