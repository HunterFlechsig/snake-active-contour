"""Window, scale, overlay drawing, and colors."""

import cv2
import numpy as np

WINDOW_NAME = "Image"
DISPLAY_SCALE = 2.5

# BGR: orange springs, magenta volcano (white vanished on the coin background)
SPRING_COLOR = (0, 140, 255)
VOLCANO_COLOR = (255, 0, 220)
SNAKE_COLOR = (0, 0, 255)
HELP_COLOR = (0, 180, 0)
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
    display = render_overlay(image, snake)
    redraw_image(display)


def format_status(iteration, sigma, n_springs, n_volcanoes, paused, finished=False):
    state = "done" if finished else ("paused" if paused else "running")
    return (
        f"iter {iteration}  sigma={sigma:.1f}  "
        f"springs={n_springs}  "
        f"volcanoes={n_volcanoes}  "
        f"{state}"
    )


def render_overlay(
    image,
    snake,
    springs=None,
    volcano=None,
    volcanoes=None,
    volcano_radius=20,
    help_lines=None,
    buttons=None,
    active_button_ids=None,
):
    display = _gray_to_bgr(image)

    if snake is not None and len(snake) > 0:
        for i, point in enumerate(snake):
            next_point = snake[(i + 1) % len(snake)]
            point_int = get_int_point(point)
            next_point_int = get_int_point(next_point)
            cv2.circle(display, point_int, 1, SNAKE_COLOR, -1)
            cv2.line(display, point_int, next_point_int, SNAKE_COLOR, 1)

    if springs:
        for spring in springs:
            start = get_int_point(snake[spring["i"]])
            if spring.get("j") is not None:
                end = get_int_point(snake[spring["j"]])
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

    if buttons:
        _draw_buttons(display, buttons, active_button_ids or [])

    if help_lines:
        y = 48 if buttons else 14
        for line in help_lines:
            cv2.putText(
                display,
                line,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                HELP_COLOR,
                1,
                cv2.LINE_AA,
            )
            y += 16

    return display


def render_contour_preview(image, points, mode):
    help_lines = (
        ["Click outline, Enter to start  |  c = circle  |  Backspace = undo"]
        if mode == "polyline"
        else ["Circle: click center, then a point on/outside the rim"]
    )
    preview = np.array(points, dtype=np.float64) if points else None
    display = render_overlay(image, preview, help_lines=help_lines)
    if preview is not None and len(preview) >= 2:
        if mode == "polyline" and len(preview) >= 3:
            cv2.line(
                display,
                get_int_point(preview[-1]),
                get_int_point(preview[0]),
                PREVIEW_CLOSE_COLOR,
                1,
            )
        for point in preview:
            cv2.circle(display, get_int_point(point), 4, SNAKE_COLOR, -1)
    return display


def _draw_volcano(display, center, volcano_radius):
    crater = max(4, int(round(volcano_radius)))
    skirt = max(crater + 1, int(round(volcano_radius * 2.5)))
    cv2.circle(display, center, skirt, VOLCANO_COLOR, 2)
    cv2.circle(display, center, crater, VOLCANO_COLOR, 2)
    cv2.circle(display, center, 4, VOLCANO_COLOR, -1)


def _draw_buttons(display, buttons, active_ids):
    for button in buttons:
        x, y, w, h = button["rect"]
        fill = VOLCANO_COLOR if button["id"] in active_ids else (40, 40, 40)
        cv2.rectangle(display, (x, y), (x + w, y + h), fill, -1)
        cv2.rectangle(display, (x, y), (x + w, y + h), VOLCANO_COLOR, 2)
        text_size = cv2.getTextSize(
            button["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1,
        )[0]
        tx = x + (w - text_size[0]) // 2
        ty = y + (h + text_size[1]) // 2
        cv2.putText(
            display,
            button["label"],
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
