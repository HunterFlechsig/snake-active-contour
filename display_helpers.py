import cv2
WINDOW_NAME = "Image"

def get_int_point(point):
    return (int(point[0]), int(point[1]))

def display_snake(snake, image):
    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, point in enumerate(snake):
        next_point = snake[(i + 1) % len(snake)]
        point_int = get_int_point(point)
        next_point_int = get_int_point(next_point)

        cv2.circle(display, point_int, 1, (0, 0, 255), -1)  
        cv2.line(display, point_int, next_point_int, (0, 0, 255), 1)
    redraw_image(display)


def load_image():
    return cv2.imread('coins.jpg', cv2.IMREAD_GRAYSCALE)

def display_image(image):
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, image)
    cv2.waitKey(1)

def redraw_image(image):
    cv2.imshow(WINDOW_NAME, image)


def get_points(image):
    clicked_points = []
    prompts = [
        "Click roughly the center of the object",
        "Now click a point outside the object",
    ]

    def show_prompt():
        if len(clicked_points) < len(prompts):
            print(prompts[len(clicked_points)])

    def mouse_click_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            print(f"Point registered at: X={x}, Y={y}")
            display = image.copy()
            for point in clicked_points:
                cv2.circle(display, point, 5, (0, 0, 255), -1)
            redraw_image(display)
            show_prompt()

    cv2.setMouseCallback(WINDOW_NAME, mouse_click_callback)
    show_prompt()

    while len(clicked_points) < 2:
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    return clicked_points

def close_window():
    cv2.destroyWindow(WINDOW_NAME)