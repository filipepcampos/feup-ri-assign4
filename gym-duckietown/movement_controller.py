from enum import Enum
import queue
import cv2 as cv
import numpy as np
from simple_pid import PID


class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    GO_FORWARD = 2
    GO_BACKWARD = 3
    STOP = 4

class State(Enum):
    MOVING_IN_LANE = 0
    CURVE = 1

class ArucoMovementController:
    def __init__(self): 
        self.direction = 0
        self.speed = 0
        self.action_queue = queue.Queue()
        self.wheel_distance = 0.102
        self.min_rad = 0.08
        self.forward_speed = 0.44
        self.state = State.MOVING_IN_LANE
        self.curve_pos = (0,0)
        self.distance_to_aruco = 0

        self.pid = PID(0.5, 0.1, 0.05, setpoint=0)
        self.last_measured_angle = 0


    def add_action(self, action: Action):
        self.action_queue.put(action)

    def move(self, action: Action):
        self.direction = 0
        self.speed = 0
        if action == Action.TURN_LEFT:
            self.direction += 1
        elif action == Action.TURN_RIGHT:
            self.direction -= 1
        elif action == Action.GO_FORWARD:
            self.speed += 0.44
        elif action == Action.GO_BACKWARD:
            self.speed -= 0.44
        elif action == Action.STOP:
            self.speed = 0
            self.direction = 0
        
        return self.adjust_speed((self.speed, self.direction))

    def adjust_speed(self, movement):
        v1, v2 = movement[0], movement[1]

        # Limit radius of curvature
        q = (self.min_rad + self.wheel_distance / 2.0) / (self.min_rad - self.wheel_distance / 2.0)
        if v1 == 0 or abs(v2 / v1) > q:
            # adjust velocities evenly such that condition is fulfilled
            delta_v = (v2 - v1) / 2 - self.wheel_distance / (4 * self.min_rad) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v
    
        return v1, v2
    
    def at_intersection(self, red_line): 
        x1, _, x2, _ = red_line
        return abs(x1 - x2) > 400


    def detect_curve(self, angle): 
        if abs(angle - self.last_measured_angle) > 0.1:
            self.last_measured_angle = angle
            self.action_queue.put(Action.TURN_LEFT)

    
    def make_curve(self):
        pass 

    def lane_following(self, aruco_angle, aruco_distance, white_angle, yellow_angle, white_lines, yellow_lines):
        # use PID to keep the car in the lane and follow the aruco marker at a constant distance
        
        # if we're not describing a curve, follow the aruco marker
        if not self.state == State.CURVE:
            if aruco_angle is not None and aruco_distance is not None:
                self.distanc_to_aruco = aruco_distance
                self.pid.setpoint = aruco_angle
                self.direction = self.pid(white_angle)
                self.speed = self.forward_speed
            else:
                self.direction = 0
                self.speed = 0
                self.state = State.CURVE

    
