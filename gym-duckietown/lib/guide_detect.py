from collections import deque
from enum import Enum
import numpy as np
from .aruco_detector import ArucoDetector
import cv2 as cv

class Direction(Enum):
    VERY_LEFT = 0
    LEFT = 1
    CENTER = 2
    RIGHT = 3
    VERY_RIGHT = 4
    NONE = 5

class Distance(Enum):
    VERY_CLOSE = 0
    CLOSE = 1
    MEDIUM = 2
    FAR = 3
    VERY_FAR = 4
    NONE = 5
    

class GuideBotDetector():
    def __init__(self):
        self.direction = Direction.CENTER
        self.distance = Distance.MEDIUM

        # DEBUG
        self.max = 1
        self.min = 1000

    def update(self):
        raise NotImplementedError("Inherit from this class and implement update()")

    def get_direction(self):
        return self.direction
    
    def get_distance(self):
        return self.distance

class ArUcoBotDetector(GuideBotDetector):
    def __init__(self):
        super().__init__()
        self.aruco_detector = ArucoDetector(
            np.array(
                [
                    [305.5718893575089, 0, 303.0797142544728],
                    [0, 308.8338858195428, 231.8845403702499],
                    [0, 0, 1],
                ]
            ),
            np.array([-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]),
        )
        self.last_aruco_angle = 0
        self.aruco_angle_history = deque(maxlen=4)

    
    def update(self, frame):
        angle, distance = self.aruco_detection(frame)
        if angle is not None:
            self.aruco_angle_history.append(angle)
        self.update_angle(angle)
        self.update_distance(distance)

    def update_distance(self, distance):
        print("DISTANCE: ", distance)
        if distance is None:
            self.distance = Distance.NONE
        elif distance < 0.1:
            self.distance = Distance.VERY_CLOSE
        elif distance < 0.2:
            self.distance = Distance.CLOSE
        elif distance < 0.9:
            self.distance = Distance.MEDIUM
        elif distance < 1.5:
            self.distance = Distance.FAR
        else:
            self.distance = Distance.VERY_FAR

    def update_angle(self, angle):
        if angle is None:
            self.direction = Direction.NONE
            return
        
        delta_angle = angle - self.last_aruco_angle
        self.last_aruco_angle = angle

        print(f"ANGLE: {angle}, DELTA ANGLE: {delta_angle}")

        angle = np.mean(self.aruco_angle_history)

        #DEBUG 
        self.max = angle if abs(angle) > abs(self.max) else self.max
        self.min = angle if abs(angle) < abs(self.min) else self.min
        print("MAX, MIN: ", self.max, self.min)

        if angle > 1.85:
            self.direction = Direction.RIGHT
        elif angle < 1.0:
            self.direction = Direction.LEFT
        else:
            self.direction = Direction.CENTER
    
    def aruco_detection(self, frame):
        corners, ids, _ = self.aruco_detector .detectMarkers(frame)
        pose = self.aruco_detector .estimatePose(corners) if ids is not None else None
        print("\n\n\n Aruco pose: ", pose)

        if ids is None or pose is None:
            return None, None

        for i, id_val in enumerate(ids):
            if id_val != 0:
                break
        
            rvecs, tvecs = pose
            aruco_angle = rvecs[i][2]
            distance = tvecs[i][2]
            aruco_angle += np.pi / 2
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            marker_coordinates = corners[0]
            center_point = np.mean(marker_coordinates, axis=1, dtype=np.int32)[0]

            if aruco_angle is not None:
                x1, y1 = center_point[0], center_point[1]
                #aruco_angle += np.pi / 2
                x2, y2 = int(np.cos(aruco_angle) * 100)+x1, int(np.sin(aruco_angle) * 100)+y1

                self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
                    # cv.arrowedLine(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)
            return aruco_angle, distance
        return None, None
    
    def draw(self, frame):
        angle = np.mean(self.aruco_angle_history)
        if angle is not None and len(self.aruco_angle_history) > 0:
            # x1, y1 = 320, 240
            # x2, y2 = int(np.cos(angle) * 100)+x1, int(np.sin(angle) * 100)+y1
            cv.arrowedLine(frame, (self.x2, self.y2), (self.x1, self.y1), (0, 255, 0), 2)
            cv.putText(frame, f"Angle: {angle}", (self.x2, self.y2), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

import sys
sys.path.insert(0, './feup-ri-assign4-model')
from models.yolo import Model
from models.common import DetectMultiBackend
import torch

class YOLOBotDetector():
    def __init__(self, weights_path="weights/best.pt"):
        self.direction = Direction.CENTER
        self.distance = Distance.MEDIUM
        
        self.model = DetectMultiBackend(weights_path)
        self.model.warmup(imgsz=(1, 3, 256, 256))

        self.dist_queue = deque(maxlen=2)
        self.rot_queue = deque(maxlen=2)
        self.detections = []
        self.best_duckiebot_detection = None

    def update(self, frame):
        resized_frame = cv.resize(frame, (256, 256))
        img = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        img = img[None] # expand dimension to batch size 1    

        results = self.model(img)[0][0]
        self.process_results(results)
        self.update_direction_and_distance()

    
    def update_direction_and_distance(self):
        if self.best_duckiebot_detection is None:
            self.direction = Direction.NONE
            self.distance = Distance.NONE
            return
        center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = self.best_duckiebot_detection

        self.dist_queue.append([dist1, dist2, dist3, dist4, dist5])

        self.rot_queue.append([rot1, rot2, rot3, rot4, rot5])


        
    def process_results(self, results):
        detections = []
        conf_thresh = 0.35
        best_duckiebot_detection = None
        best_duckiebot_conf = 0

        for result in results:
            conf = result[4]
            class_label = 0 if result[5] > result[6] else 1

            if conf > conf_thresh:
                if class_label == 0:
                    detections.append((class_label, result))

            if conf > best_duckiebot_conf and class_label == 1 and conf > 0.2:
                best_duckiebot_detection = result
                best_duckiebot_conf = conf

        if best_duckiebot_detection is not None:
            self.best_duckiebot_detection = best_duckiebot_detection
            detections.append((1, best_duckiebot_detection))
        
        self.detections = detections

    def get_direction(self):
        avg_rot = [sum(x) / len(x) for x in zip(*self.rot_queue)]
        y = np.argmax(avg_rot)
        return [Direction.VERY_LEFT, Direction.LEFT, Direction.CENTER, Direction.RIGHT, Direction.VERY_RIGHT][y]
    
    def get_distance(self):
        avg_dist = [sum(x) / len(x) for x in zip(*self.dist_queue)]
        x = np.argmax(avg_dist)
        return [Distance.VERY_CLOSE, Distance.CLOSE, Distance.MEDIUM, Distance.FAR, Distance.VERY_FAR][x]
    
    def draw(self, frame):
        resized_frame = cv.resize(frame, (256, 256))

        # Draw bounding boxes
        for detection in self.detections:
            label, dets = detection
            center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = dets
            x1, y1 = center_x - width / 2, center_y - height / 2
            x2, y2 = center_x + width / 2, center_y + height / 2
            cv.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw best duckiebot information
        def draw_info():
            if self.best_duckiebot_detection is None:
                return
            center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = self.best_duckiebot_detection

            
            x = int(256 / 2 + 20)

            cv.putText(resized_frame, "distance", (x, 230), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 127, 0), 1)

            values = [sum(x) / len(x) for x in zip(*self.dist_queue)]
            # values = torch.softmax(torch.tensor(values), dim=0)
            for dist, label in zip(values, ["VC", "C", "M", "F", "VF"]):
                cv.rectangle(resized_frame, (x, 200), (x + 10, 200-int(50 * dist)), (0, 127, 0), 2)
                cv.putText(resized_frame, label, (x, 210), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 127, 0), 1)
                x += 10
            
            x = int(256 / 2 - 20 - 10*5)
            cv.putText(resized_frame, "rot", (x, 230), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            values = [sum(x) / len(x) for x in zip(*self.rot_queue)]
            # values = torch.softmax(torch.tensor(values), dim=0)
            
            for rot, label in zip(values, ["VL", "L", "M", "R", "VR"]):
                cv.rectangle(resized_frame, (x, 200), (x + 10, 200-int(50 * rot)), (0, 0, 255), 2)
                cv.putText(resized_frame, label, (x, 210), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
                x += 10
        
        draw_info()
        return resized_frame