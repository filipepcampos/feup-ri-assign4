import cv2 as cv
import numpy as np


class EdgeDetector:
    def __init__(self):
        self.erode_kernel = np.ones((5, 5), np.uint8)
        self.dilate_kernel = np.ones((9, 9), np.uint8)
        
    def define_masks(self, frame):
        lower_white, upper_white = np.array([0, 0, 200]), np.array([175, 175, 255])
        lower_yellow, upper_yellow = np.array([20, 100, 100]), np.array([30, 255, 255])
        # 356, 78, 92 as to be expected (170, 70, 50), Scalar(180, 255, 255)
        lower_red, upper_red = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask_white = cv.inRange(frame, lower_white, upper_white)
        mask_yellow = cv.inRange(frame, lower_yellow, upper_yellow)
        mask_red = cv.inRange(frame, lower_red, upper_red)
        
        return mask_white, mask_yellow, mask_red
        
    def erode_and_dilate(self, masks): 
        mask_white, mask_yellow, mask_red = masks
        mask_white = cv.erode(mask_white, self.erode_kernel, iterations=2)
        mask_yellow = cv.erode(mask_yellow, self.erode_kernel, iterations=1)
        mask_yellow = cv.dilate(mask_yellow, self.dilate_kernel, iterations=1)

        mask_red = cv.erode(mask_red, self.erode_kernel, iterations=2)
        mask_red = cv.dilate(mask_red, self.dilate_kernel, iterations=1)


        return mask_white, mask_yellow, mask_red

    def detect_edges(self, masks):
         mask_white, mask_yellow, mask_red = masks
         edges_white = cv.Canny(mask_white, 100, 200)    
         edges_yellow = cv.Canny(mask_yellow, 100, 200)
         edges_red = cv.Canny(mask_red, 100, 200)
         return edges_white, edges_yellow, edges_red

    def detect_lines(self, colored_edges, maxLineGaps): 
        # Get lines from edges
        white_edges, yellow_edges, red_edges = colored_edges

        lines = lambda edges, maxLineGap: cv.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=maxLineGap
        )

        return lines(white_edges, maxLineGaps[0]), \
               lines(yellow_edges, maxLineGaps[1]), \
               lines(red_edges, maxLineGaps[2])

    def remove_horizontal_lines(self, lines): 
        return lines[abs(lines[:, :, 1] - lines[:, :, 3]) > 50]

    def get_horizontal_lines(self, lines):
        return lines[abs(lines[:, :, 1] - lines[:, :, 3]) < 50]
    
    def get_average_line(self, lines): 
        return np.mean(lines, axis=0, dtype=np.int32)

    
    def get_angle(self, line): 
        x1, y1, x2, y2 = line
        return np.arctan2(y2 - y1, x2 - x1) - np.pi / 2

    def draw_line(self, frame, line, color):
        x1, y1, x2, y2 = line
        cv.line(frame, (x1, y1), (x2, y2), color, 5)
        

