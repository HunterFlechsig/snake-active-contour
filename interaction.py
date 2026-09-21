"""Contour picking, Snake Pit mouse, buttons, and key bindings."""

import cv2
import numpy as np

from display import bind_mouse, redraw_image, render_contour_preview


def nearest_snake_index(snake, point, max_dist):
    distances = np.linalg.norm(snake - np.asarray(point, dtype=np.float64), axis=1)
    index = int(np.argmin(distances))
    if distances[index] <= max_dist:
        return index
    return None


def make_pit_buttons(image_shape):
    _height, width = image_shape[:2]
    button_w, button_h, gap, top = 128, 26, 8, 8
    mouse_x = width - gap - button_w
    place_x = mouse_x - gap - button_w
    return [
        {
            "id": "place",
            "label": "Place volcano",
            "rect": (place_x, top, button_w, button_h),
        },
        {
            "id": "mouse",
            "label": "Mouse volcano",
            "rect": (mouse_x, top, button_w, button_h),
        },
    ]


def hit_button(x, y, buttons):
    for button in buttons:
        bx, by, bw, bh = button["rect"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return button["id"]
    return None


class SnakePit:
    """Mouse-driven springs and volcano from Kass section 2.2."""

    def __init__(self, pick_radius=18.0):
        self.pick_radius = pick_radius
        self.mouse = np.array([0.0, 0.0])
        self.springs = []
        self.dragging = None
        self.placed_volcanoes = []
        self.place_mode = False
        self.mouse_volcano_on = False
        self.volcano_live = False

    def handle_mouse(self, event, x, y, snake, buttons=None):
        self.mouse = np.array([x, y], dtype=np.float64)
        buttons = buttons or []

        if event == cv2.EVENT_MOUSEMOVE:
            self._follow_mouse()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = hit_button(x, y, buttons)
            if hit == "place":
                self.place_mode = not self.place_mode
                return
            if hit == "mouse":
                self.mouse_volcano_on = not self.mouse_volcano_on
                return
            if self.place_mode:
                self.placed_volcanoes.append(self.mouse.copy())
                self.place_mode = False
                return
            self._grab_or_make_spring(snake)
        elif event == cv2.EVENT_LBUTTONUP:
            self._release_spring(snake)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.volcano_live = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.volcano_live = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.delete_nearest_constraint(snake)

    def sync(self):
        self._follow_mouse()

    def _follow_mouse(self):
        if self.dragging is not None:
            self.springs[self.dragging]["anchor"] = self.mouse.copy()
            self.springs[self.dragging]["j"] = None

    def live_volcano(self):
        if self.volcano_live or self.mouse_volcano_on:
            return self.mouse.copy()
        return None

    def all_volcanoes(self):
        volcanoes = [volcano.copy() for volcano in self.placed_volcanoes]
        live = self.live_volcano()
        if live is not None:
            volcanoes.append(live)
        return volcanoes

    def active_button_ids(self):
        active = []
        if self.place_mode:
            active.append("place")
        if self.mouse_volcano_on or self.volcano_live:
            active.append("mouse")
        return active

    def _grab_or_make_spring(self, snake):
        index = nearest_snake_index(snake, self.mouse, self.pick_radius)
        if index is None:
            return

        for i, spring in enumerate(self.springs):
            if spring["i"] == index and spring.get("j") is None:
                self.dragging = i
                spring["anchor"] = self.mouse.copy()
                return

        self.springs.append(
            {
                "i": index,
                "j": None,
                "anchor": self.mouse.copy(),
            }
        )
        self.dragging = len(self.springs) - 1

    def _release_spring(self, snake):
        if self.dragging is None:
            return

        spring = self.springs[self.dragging]
        other = nearest_snake_index(snake, self.mouse, self.pick_radius)
        if other is not None and other != spring["i"]:
            spring["j"] = other
            spring["anchor"] = None
        else:
            spring["j"] = None
            spring["anchor"] = self.mouse.copy()

        self.dragging = None

    def delete_nearest_constraint(self, snake):
        best_kind = None
        best_i = None
        best_dist = self.pick_radius

        for i, spring in enumerate(self.springs):
            start = snake[spring["i"]]
            if spring.get("j") is not None:
                end = snake[spring["j"]]
            else:
                end = spring["anchor"]
            dist = min(
                np.linalg.norm(start - self.mouse),
                np.linalg.norm(end - self.mouse),
            )
            if dist < best_dist:
                best_dist = dist
                best_kind = "spring"
                best_i = i

        for i, volcano in enumerate(self.placed_volcanoes):
            dist = np.linalg.norm(volcano - self.mouse)
            if dist < best_dist:
                best_dist = dist
                best_kind = "volcano"
                best_i = i

        if best_kind == "spring":
            self.springs.pop(best_i)
            self.dragging = None
        elif best_kind == "volcano":
            self.placed_volcanoes.pop(best_i)

    def delete_nearest_spring(self, snake):
        self.delete_nearest_constraint(snake)

    def clear_volcano(self):
        self.placed_volcanoes = []
        self.place_mode = False
        self.mouse_volcano_on = False
        self.volcano_live = False

    def clear_springs(self):
        self.springs = []
        self.dragging = None


def install_pit_mouse(pit, holder, buttons):
    def on_event(event, x, y, flags, param):
        pit.handle_mouse(event, x, y, holder["snake"], buttons)

    bind_mouse(on_event)


def handle_key(key, pit, snake, paused, sigma):
    """Apply one live-loop key. Returns (quit, paused, sigma, rebuild_forces)."""

    if key == ord("q"):
        return True, paused, sigma, False

    rebuild_forces = False

    if key == ord(" "):
        paused = not paused
    if key == ord("v"):
        pit.clear_volcano()
    if key == ord("s"):
        pit.clear_springs()
    if key == ord("x"):
        pit.delete_nearest_spring(snake)
    if key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
        sigma = float(key - ord("0"))
        rebuild_forces = True
        print(f"sigma = {sigma:.1f}")

    return False, paused, sigma, rebuild_forces


def print_runtime_help():
    print("Snake is minimizing. Drag it with springs or push it with the volcano.")
    print("Place volcano: click the button, then click the image to pin one.")
    print("Mouse volcano: toggle the button, or hold right-click to follow the cursor.")
    print("Left-drag a bead: spring. x/M-click deletes the nearest spring or volcano.")
    print("1-5 change blur, space pauses, v clears volcanoes, s clears springs, q quits.")


def get_initial_contour(image):
    """Snake Pit style start: click points around the object, or 'c' for a circle."""

    points = []
    mode = "polyline"
    finished = False

    print("Click points around the object, then press Enter to close the snake.")
    print("Press 'c' for the old circle shortcut (center, then a rim point).")
    print("Backspace undoes a point. q quits.")

    def redraw():
        redraw_image(render_contour_preview(image, points, mode))

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        print(f"Point registered at: X={x}, Y={y}")
        redraw()

    bind_mouse(on_click)
    redraw()

    while not finished:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):
            if mode == "circle" and len(points) >= 2:
                finished = True
            elif mode == "polyline" and len(points) >= 3:
                finished = True
            else:
                need = 2 if mode == "circle" else 3
                print(f"Need at least {need} points before starting.")

        elif key == ord("c"):
            mode = "circle"
            points.clear()
            print("Circle mode: click the center, then a rim point.")
            redraw()

        elif key == ord("p"):
            mode = "polyline"
            points.clear()
            print("Polyline mode: click around the object, then Enter.")
            redraw()

        elif key in (8, 127):
            if points:
                points.pop()
                redraw()

        elif key == ord("q"):
            return None, None

        if mode == "circle" and len(points) >= 2:
            finished = True

    return points, mode
