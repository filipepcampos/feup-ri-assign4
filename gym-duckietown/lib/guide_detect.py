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

    
    def update(self, frame):
        angle, distance = self.aruco_detection(frame)
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

        if delta_angle > 1.0:
            self.direction = Direction.RIGHT
        elif delta_angle < -1.0:
            self.direction = Direction.LEFT
    
    def aruco_detection(self, frame):
        corners, ids, _ = self.aruco_detector .detectMarkers(frame)
        pose = self.aruco_detector .estimatePose(corners) if ids is not None else None
        print("\n\n\n Aruco pose: ", pose)

        if pose is not None:
            rvecs, tvecs = pose
            aruco_angle = rvecs[0][2]
            distance = tvecs[0][2]
            
            cv.aruco.drawDetectedMarkers(frame, corners, ids)
            marker_coordinates = corners[0]
            center_point = np.mean(marker_coordinates, axis=1, dtype=np.int32)[0]

            if aruco_angle is not None:
                x1, y1 = center_point[0], center_point[1]
                aruco_angle += np.pi / 2
                x2, y2 = int(np.cos(aruco_angle) * 100)+x1, int(np.sin(aruco_angle) * 100)+y1
                # cv.arrowedLine(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)

            return aruco_angle, distance
        return None, None
