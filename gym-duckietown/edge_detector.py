import cv2 as cv
import numpy as np


class EdgeDetector:
    def __init__(self):
        self.erode_kernel = np.ones((5, 5), np.uint8)
        self.dilate_kernel = np.ones((9, 9), np.uint8)
        
    def define_masks(self, frame):
        sensitivity = 40
        lower_white, upper_white = np.array([0, 0, 150]), np.array([175, 100, 255])
        lower_yellow, upper_yellow = np.array([20, 50, 0]), np.array([50, 255, 255])
        # 356, 78, 92 as to be expected (170, 70, 50), Scalar(180, 255, 255)
        lower_red, upper_red = np.array([170, 70, 50]), np.array([180, 255, 255])
        mask_white = cv.inRange(frame, lower_white, upper_white)
        mask_yellow = cv.inRange(frame, lower_yellow, upper_yellow)
        mask_red = cv.inRange(frame, lower_red, upper_red)
        
        return mask_white, mask_yellow, mask_red
        
    def erode_and_dilate(self, masks): 
        mask_white, mask_yellow, mask_red = masks
        mask_white = cv.erode(mask_white, self.erode_kernel, iterations=2)
        mask_white = cv.dilate(mask_white, self.dilate_kernel, iterations=1)

        cv.imwrite("mask_white.png", mask_white)


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

    def get_lines(self,converted, frame): 
         # Define masks
          mask_white, mask_yellow, mask_red = self.define_masks(converted)
          mask_white, mask_yellow, mask_red = self.erode_and_dilate((mask_white, mask_yellow, mask_red))
                                                                                                                     
          cv.imshow("mask_white", mask_white)
          cv.imshow("mask_yellow", mask_yellow)
          cv.imshow("mask_red", mask_red)
                                                                                                                     
          # Get lanes by detecting edges 
          edges_white, edges_yellow, edges_red = self.detect_edges((mask_white, mask_yellow, mask_red))   
                                                                                                                     
          cv.imshow("edges_white", edges_white)
          cv.imshow("edges_yellow", edges_yellow)
          cv.imshow("edges_red", edges_red)
                                                                                                                     
          # Get lines from edges
          white_lines, yellow_lines, red_lines = self.detect_lines((edges_white, edges_yellow, edges_red), 
                                                                            [10,50, 100])
          
          if white_lines is not None:
              white_lines = self.remove_horizontal_lines(white_lines)
          if red_lines is not None:
              red_lines = self.get_horizontal_lines(red_lines)
          
                                                                                                                     
          white_line = None
          yellow_line = None
          red_line = None
            
          # Get the average line
          if white_lines is not None:
              white_line = self.get_average_line(white_lines)
              self.draw_line(frame, white_line, (0, 0, 255))
          if yellow_lines is not None:
              yellow_line = self.get_average_line(yellow_lines)
              self.draw_line(frame, yellow_line[0], (255, 0, 0))
          if red_lines is not None and len(red_lines) > 0:
              #red_line = self.get_average_line(red_lines)
              red_line = red_lines[0]
              self.draw_line(frame, red_line, (0, 255, 0))

          return white_line, yellow_line, red_line

