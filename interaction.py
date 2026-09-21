"""Contour picking, Snake Pit mouse, and keyboard bindings."""

import cv2
import numpy as np

from display import bind_mouse, redraw_image, render_contour_preview
from physics import initialize_snake, initialize_snake_from_polyline


def nearest_point(snakes, point, max_dist):
    """Closest bead across all snakes, or None if none are within max_dist."""

    best = None
    best_dist = max_dist
    for snake_index, snake in enumerate(snakes):
        distances = np.linalg.norm(
            snake - np.asarray(point, dtype=np.float64), axis=1,
        )
        index = int(np.argmin(distances))
        if distances[index] <= best_dist:
            best_dist = distances[index]
            best = (snake_index, index)
    return best


class SnakePit:
    """Keyboard-armed, mouse-aimed springs and volcanoes."""

    def __init__(self, pick_radius=18.0):
        self.pick_radius = pick_radius
        self.mouse = np.array([0.0, 0.0])
        self.springs = []
        self.dragging = None
        self.placed_volcanoes = []
        self.place_mode = False
        self.mouse_volcano_on = False
        self.volcano_live = False
        self.auto_requested = False
        self.auto_running = False

    def handle_mouse(self, event, x, y, snakes):
        self.mouse = np.array([x, y], dtype=np.float64)

        if event == cv2.EVENT_MOUSEMOVE:
            self._follow_mouse()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.place_mode:
                self.placed_volcanoes.append(self.mouse.copy())
                self.place_mode = False
                print(f"Volcano placed at ({x}, {y}).")
                return
            self._grab_or_make_spring(snakes)
        elif event == cv2.EVENT_LBUTTONUP:
            self._release_spring(snakes)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.volcano_live = True
            print("Mouse volcano ON (right mouse).")
        elif event == cv2.EVENT_RBUTTONUP:
            self.volcano_live = False
            print("Mouse volcano OFF.")
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.delete_nearest_constraint(snakes)

    def sync(self):
        self._follow_mouse()

    def _follow_mouse(self):
        if self.dragging is not None:
            spring = self.springs[self.dragging]
            spring["anchor"] = self.mouse.copy()
            spring["j"] = None
            spring["other_snake"] = None

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

    def _grab_or_make_spring(self, snakes):
        hit = nearest_point(snakes, self.mouse, self.pick_radius)
        if hit is None:
            return

        snake_index, index = hit
        for i, spring in enumerate(self.springs):
            if (
                spring.get("snake", 0) == snake_index
                and spring["i"] == index
                and spring.get("other_snake") is None
            ):
                self.dragging = i
                spring["anchor"] = self.mouse.copy()
                return

        self.springs.append(
            {
                "snake": snake_index,
                "i": index,
                "other_snake": None,
                "j": None,
                "anchor": self.mouse.copy(),
            }
        )
        self.dragging = len(self.springs) - 1

    def _release_spring(self, snakes):
        if self.dragging is None:
            return

        spring = self.springs[self.dragging]
        hit = nearest_point(snakes, self.mouse, self.pick_radius)
        owner = spring.get("snake", 0)
        if hit is not None and hit != (owner, spring["i"]):
            spring["other_snake"] = hit[0]
            spring["j"] = hit[1]
            spring["anchor"] = None
        else:
            spring["other_snake"] = None
            spring["j"] = None
            spring["anchor"] = self.mouse.copy()

        self.dragging = None

    def delete_nearest_constraint(self, snakes):
        best_kind = None
        best_i = None
        best_dist = self.pick_radius

        for i, spring in enumerate(self.springs):
            start = snakes[spring.get("snake", 0)][spring["i"]]
            if spring.get("other_snake") is not None:
                end = snakes[spring["other_snake"]][spring["j"]]
            elif spring.get("j") is not None:
                end = snakes[spring.get("snake", 0)][spring["j"]]
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
            print("Deleted a spring.")
        elif best_kind == "volcano":
            self.placed_volcanoes.pop(best_i)
            print("Deleted a volcano.")
        else:
            print("Nothing nearby to delete.")

    def delete_nearest_spring(self, snakes):
        self.delete_nearest_constraint(snakes)

    def clear_volcano(self):
        self.placed_volcanoes = []
        self.place_mode = False
        self.mouse_volcano_on = False
        self.volcano_live = False
        print("Cleared all volcanoes.")

    def clear_springs(self):
        self.springs = []
        self.dragging = None
        print("Cleared all springs.")

    def toggle_place_mode(self):
        self.place_mode = not self.place_mode
        if self.place_mode:
            print("Place volcano: click the image to pin one.")
        else:
            print("Place volcano cancelled.")

    def toggle_mouse_volcano(self):
        self.mouse_volcano_on = not self.mouse_volcano_on
        if self.mouse_volcano_on:
            print("Mouse volcano ON — follows the cursor.")
        else:
            print("Mouse volcano OFF.")


def install_pit_mouse(pit, holder):
    def on_event(event, x, y, flags, param):
        pit.handle_mouse(event, x, y, holder["snakes"])

    bind_mouse(on_event)


def handle_key(key, pit, snakes, paused, sigma):
    """Apply one live-loop key.

    Returns (quit, paused, sigma, rebuild_forces, start_auto).
    """

    if key == 255 or key == -1:
        return False, paused, sigma, False, False

    if key == ord("q"):
        return True, paused, sigma, False, False

    rebuild_forces = False
    start_auto = key == ord("a")

    if key == ord(" "):
        paused = not paused
        print("Paused." if paused else "Running.")
    elif key == ord("p"):
        pit.toggle_place_mode()
    elif key == ord("m"):
        pit.toggle_mouse_volcano()
    elif key == ord("v"):
        pit.clear_volcano()
    elif key == ord("s"):
        pit.clear_springs()
    elif key == ord("x"):
        pit.delete_nearest_spring(snakes)
    elif key == ord("h"):
        print_runtime_help()
    elif key == ord("a"):
        print("Auto finish requested.")
    elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
        sigma = float(key - ord("0"))
        rebuild_forces = True
        print(f"Blur sigma = {sigma:.1f}")

    return False, paused, sigma, rebuild_forces, start_auto


def print_setup_help():
    print()
    print("=" * 62)
    print("  SNAKES — setup")
    print("  Click the image window so keys go there, not this terminal.")
    print("=" * 62)
    print("  Draw one snake")
    print("    click        add a point")
    print("    Backspace    undo last point")
    print("    c            circle mode (center, then a rim point)")
    print("    p            polyline mode (click around the object)")
    print("    Enter        close this snake")
    print()
    print("  After a snake is closed")
    print("    n            draw another snake")
    print("    Enter        start the run")
    print("    q            quit")
    print("=" * 62)
    print()


def print_runtime_help():
    print()
    print("=" * 62)
    print("  SNAKES — running")
    print("  Click the image window so keys go there, not this terminal.")
    print("=" * 62)
    print("  Springs")
    print("    left-drag a bead    pull with a spring")
    print("    release in empty    pin that end")
    print("    release on a bead   spring between beads (any snake)")
    print("    x                   delete nearest spring or volcano")
    print("    s                   clear all springs")
    print()
    print("  Volcano")
    print("    p then click        place a volcano (stays put)")
    print("    m                   toggle mouse volcano (follows cursor)")
    print("    right-drag          mouse volcano while held")
    print("    v                   clear all volcanoes")
    print()
    print("  Solve")
    print("    space               pause / resume")
    print("    a                   auto finish (settle, then stop)")
    print("    1-5                 blur amount")
    print("    h                   print this help again")
    print("    q                   quit")
    print("=" * 62)
    print()


def get_initial_contour(image, existing_snakes=None, snake_number=1):
    """Click points for one snake. Returns (points, mode) or (None, None) on quit."""

    existing_snakes = existing_snakes or []
    points = []
    mode = "polyline"
    finished = False

    print(f"Snake {snake_number}: click around the object, then Enter to close.")
    print("  c = circle, p = polyline, Backspace = undo, q = quit")

    def redraw():
        redraw_image(
            render_contour_preview(image, points, mode, existing_snakes)
        )

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        print(f"  point {len(points)} at ({x}, {y})")
        redraw()

    bind_mouse(on_click)
    redraw()

    while not finished:
        key = cv2.waitKey(20) & 0xFF

        if key in (13,):
            if mode == "circle" and len(points) >= 2:
                finished = True
            elif mode == "polyline" and len(points) >= 3:
                finished = True
            else:
                need = 2 if mode == "circle" else 3
                print(f"  Need at least {need} points before closing.")

        elif key == ord("c"):
            mode = "circle"
            points.clear()
            print("  Circle mode: click the center, then a rim point.")
            redraw()

        elif key == ord("p"):
            mode = "polyline"
            points.clear()
            print("  Polyline mode: click around the object, then Enter.")
            redraw()

        elif key in (8, 127):
            if points:
                points.pop()
                print(f"  undid last point ({len(points)} left)")
                redraw()

        elif key == ord("q"):
            return None, None

        if mode == "circle" and len(points) >= 2:
            finished = True

    return points, mode


def collect_snakes(image):
    """Draw one or more snakes, then Enter to start the run."""

    print_setup_help()
    snakes = []

    while True:
        points, mode = get_initial_contour(
            image,
            existing_snakes=snakes,
            snake_number=len(snakes) + 1,
        )
        if points is None:
            return None if not snakes else snakes

        if mode == "circle":
            snakes.append(initialize_snake(points))
        else:
            snakes.append(initialize_snake_from_polyline(points))

        redraw_image(render_contour_preview(image, [], "polyline", snakes))
        print(f"Added snake {len(snakes)}.")
        print("  n = another snake   Enter = start run   q = quit")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("n"), ord("N")):
                break
            if key in (13, 32):
                return snakes
            if key == ord("q"):
                return snakes
