import cv2
import numpy as np

LOWER_COLOR = np.array([20, 230, 60])
UPPER_COLOR = np.array([30, 255, 150])


def filter_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
    res = cv2.bitwise_and(img, img, mask=mask)
    return res


def apply_threshold(gray_img):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def apply_morphology(thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing


def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_bounding_boxes(contours):
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes


def eval_img_duckies(img):
    filtered_img = filter_color(img)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    thresh = apply_threshold(gray_img)
    closing = apply_morphology(thresh)
    contours = find_contours(closing)
    bounding_boxes = find_bounding_boxes(contours)
    return bounding_boxes
