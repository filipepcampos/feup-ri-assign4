import cv2
import numpy as np

# 0: yellow (duckie), 1: red (duckiebot)
LOWER_COLORS = [
    np.array([20, 245, 125]),
    np.array([0, 230, 60]),
]
UPPER_COLORS = [
    np.array([30, 255, 255]),
    np.array([10, 255, 150]),
]


def filter_color(img, class_id):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLORS[class_id], UPPER_COLORS[class_id])
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def apply_threshold(gray_img):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def apply_morphology(thresh, class_id):
    kernel = np.ones((3, 3), np.uint8)

    if class_id == 0:
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif class_id == 1:
        closing = cv2.dilate(thresh, kernel, iterations=2)
    return closing


def find_contour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    biggest_contour = contours[0]
    biggest_area = cv2.contourArea(biggest_contour)
    for contour in contours[1:]:
        area = cv2.contourArea(contour)
        if area > biggest_area:
            biggest_contour = contour
            biggest_area = area

    return biggest_contour


def find_bounding_box(contour):
    if contour is None:
        return None

    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, w, h]


def eval_img_object(img, class_id):
    filtered_img = filter_color(img, class_id)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    thresh = apply_threshold(gray_img)
    closing = apply_morphology(thresh, class_id)
    contours = find_contour(closing)
    bounding_box = find_bounding_box(contours)
    return bounding_box
