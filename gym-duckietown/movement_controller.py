from enum import Enum
import queue
import cv2 as cv
import numpy as np
from simple_pid import PID
import math

RAD_TO_DEG = 180 / np.pi
WHEEL_DISTANCE = 0.102
MIN_RAD = 0.08
FORWARD_SPEED= 0.44
FORWARD_WITH_CAUTION_SPEED = 0.1
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

class Action(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    GO_FORWARD = 2
    STOP = 3

class State(Enum):
    MOVING_IN_LANE = 0

    CURVING_LEFT = 1
    CURVING_RIGHT = 2
    GOING_FORWARD = 3
    STOPPED = 4


# map action to State 

action_to_state = {
    Action.TURN_LEFT: State.CURVING_LEFT,
    Action.TURN_RIGHT: State.CURVING_RIGHT,
    Action.GO_FORWARD: State.GOING_FORWARD,
    Action.STOP: State.STOPPED
}

ROAD_TILE_SIZE =  0.61
CURVE_LEFT_BEZIER = np.array([[[-0.20, 0, -0.50], [-0.20, 0, 0.00], [0.00, 0, 0.20], [0.50, 0, 0.20]], 
                        [[0.50, 0, -0.20],[0.30, 0, -0.20], [0.20, 0, -0.30],[0.20, 0, -0.50]]])

CURVE_RIGHT_BEZIER = np.array([[[-0.20, 0, -0.50], [-0.20, 0, -0.20], [-0.30, 0, -0.20], [-0.50, 0, -0.20]],
                        [[-0.50, 0, 0.20],[-0.30, 0, 0.20], [0.30, 0, 0.00],[0.20, 0, -0.50]]])


CURVE_LEFT_BEZIER = ROAD_TILE_SIZE * CURVE_LEFT_BEZIER
CURVE_RIGHT_BEZIER = ROAD_TILE_SIZE * CURVE_RIGHT_BEZIER


def bezier_curve(curve, t):
        middle_points = [np.array(np.mean([curve[0][i], curve[1][i]], axis=0)) for i in range(len(curve[0]) - 1)]
        n = len(middle_points) - 1
        result = np.zeros_like(middle_points)
        for i, point in enumerate(middle_points):
            result += np.array(point) * math.comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)
        return result
    
num_steps = 200
left_curve_points = np.array([bezier_curve(CURVE_LEFT_BEZIER[0], t) for t in np.linspace(0, 1, num_steps)])
right_curve_points = np.array([bezier_curve(CURVE_RIGHT_BEZIER[0], t) for t in np.linspace(0, 1, num_steps)])

avg_points = lambda curve: np.array([np.mean([curve[0][i], curve[1][i]], axis=0) for i in range(len(curve[0]))])

class ArucoMovementController:
    def __init__(self): 
        self.direction = 0

        print(f"Left curve points: {left_curve_points.shape}, Right curve points: {right_curve_points.shape}")
        self.last_aruco_angle = 0
        self.current_speed = FORWARD_SPEED, 0.0

        self.action_queue = queue.Queue()
        self.action_queue.put(Action.TURN_RIGHT)
        self.state = State.MOVING_IN_LANE

        self.action_step = 0
        self.safety_distance = 0.5


        self.pid = PID(0.5, 0.1, 0.05, setpoint=0)


    def adjust_speed(self, movement):
        v1, v2 = movement[0], movement[1]

        # Limit radius of curvature
        q = (MIN_RAD + WHEEL_DISTANCE / 2.0) / (MIN_RAD - WHEEL_DISTANCE / 2.0)
        if v1 == 0 or abs(v2 / v1) > q:
            # adjust velocities evenly such that condition is fulfilled
            delta_v = (v2 - v1) / 2 - WHEEL_DISTANCE / (4 * MIN_RAD) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v
    
        return np.array([v1, v2])

    def deliberate_action(self): 
        
        if self.action_step != 0: 
            return self.state

        return State.GOING_FORWARD if self.action_queue.empty() else action_to_state[self.action_queue.get()]

    def at_intersection(self, red_line): 
        if red_line is None:
            return False

        x1, _, x2, _ = red_line
        is_intersection = abs(x1 - x2) > 300

        if self.action_step == 0 and is_intersection:
            self.state = self.deliberate_action()
            self.action_step = 1

        return is_intersection
   
    def is_taking_action(self):
        if (n := self.state) != State.MOVING_IN_LANE: 
            print(f"Taking action: {n}")
            return True
        return False
        

    def in_lane(self, white_line, yellow_line):

        if yellow_line is None and white_line is None:
            return False
        
        if white_line is None and yellow_line is not None:
            x1, _, x2, _ = yellow_line[0]
            return x1 > FRAME_WIDTH / 3 or x2 > FRAME_WIDTH / 3

        if white_line is not None and yellow_line is None:
            x1, _, x2, _ = white_line
            return x1 > 2 * FRAME_WIDTH / 3 or x2 > 2 * FRAME_WIDTH / 3
        
        x1, _, x2, _ = white_line
        x3, _, x4, _ = yellow_line[0]

        return  abs(x1 - x3) < 450 and abs(x2 - x4) < 450

    def take_curve(self, curve_type):
        # take the bezier curve and use it to move across 100 frames
        curve = left_curve_points if curve_type == Action.TURN_LEFT else right_curve_points

        next_move = curve[self.action_step]
        prev_move = curve[self.action_step - 1] if self.action_step > 0 else next_move
        
        print(f"Next move: {next_move}, Prev move: {prev_move}")
            
        angle = np.arctan2(next_move[1] - prev_move[1], next_move[0] - prev_move[0])

        angle = - angle if curve_type == Action.TURN_RIGHT else angle 

        v1, v2 = FORWARD_WITH_CAUTION_SPEED , FORWARD_WITH_CAUTION_SPEED * angle / (np.pi / 2.0)
        # * np.sin(angle)
        # , FORWARD_SPEED * angle / (np.pi / 2.0)
        # (next_move[0] - prev_move[0]) * 10 #
        print(f"Angle: {RAD_TO_DEG * angle}, V1: {v1}, V2: {v2}")

        self.action_step += 1
        print(f"Action step: {self.action_step}")
        if self.action_step == num_steps: 
            self.state = State.MOVING_IN_LANE
            self.action_step = 0

        return v1, v2

    # LEFT is positive, RIGHT is negative
    def take_action(self):
        # TODO: 3way, 4way intersection
        print(f"CURRENT STATE: {self.state}")
        if self.state == State.GOING_FORWARD: 
            return FORWARD_SPEED, 0
        elif self.state == State.CURVING_LEFT:
             return self.take_curve(Action.TURN_LEFT)
        elif self.state == State.CURVING_RIGHT:
            print("\n\n\n\nCURVING RIGHT")
            return self.take_curve(Action.TURN_RIGHT)
        
        return 0, 0
    

    def detect_curve(self, angle): 
        delta_angle = angle - self.last_aruco_angle
        if delta_angle > 1.0:
            self.last_aruco_angle = angle
            self.action_queue.put(Action.TURN_RIGHT)
            print("RIGHT CURVE")
        elif delta_angle < -1.0:
            self.last_aruco_angle = angle
            self.action_queue.put(Action.TURN_LEFT)
            print("LEFT CURVE")

        # TODO: Do not include cases which are not intersection curves
        return angle < 0


    def get_angle_correction_from_aruco(self, aruco_angle):
        # TODO: compute angle correction from aruco
        return -1.0 * aruco_angle
    
    def compute_aruco_move(self, aruco_pose):
        if aruco_pose is not None:
            rvecs, tvecs = aruco_pose
            aruco_angle = rvecs[0][2]
            aruco_distance = tvecs[0][2]

            aruco_angle_correction = self.get_angle_correction_from_aruco(aruco_angle)
            print(f"Aruco angle: {aruco_angle}, Aruco angle correction: {aruco_angle_correction}")
            distance_speedup = 1.0 - aruco_distance / self.safety_distance
            
            print(f"Distance to aruco: {aruco_distance}")
            if aruco_distance < self.safety_distance:
                return 0.0, aruco_angle_correction
            
            return FORWARD_SPEED * distance_speedup, aruco_angle_correction

        return FORWARD_WITH_CAUTION_SPEED, 0.0


    def compute_lane_following_move(self, lines):
        white_line_info, yellow_line_info = lines
        white_line, white_angle = white_line_info
        yellow_line, yellow_angle = yellow_line_info
        
        white_line_reference =  0.873 # 0.873 # 0.785 # 50 degrees
        yellow_line_reference = 2.275 # 130 degrees

#        print(f"White angle: {RAD_TO_DEG * white_angle if white_angle is not None else None}\
#                Yellow angle: {RAD_TO_DEG * yellow_angle if yellow_angle is not None else None}")
        
        dire = lambda x: "LEFT" if x > 0 else "RIGHT"
#        line_length = lambda p1, p2: np.linalg.norm(np.array(p1) - np.array(p2))
#        dist_to_pov = lambda line: FRAME_WIDTH - line[2]
        mean_correction = lambda x, y: np.mean([x, y])  # np.mean([abs(x), abs(y)]) * np.sign(x + y)
        
        correction = 0.0
        if white_angle is None and yellow_angle is None:
            return FORWARD_WITH_CAUTION_SPEED, 0.0   


        if yellow_angle is not None and abs(abs(yellow_angle) - yellow_line_reference) > 0.1: 
            yellow_line_correction =  1.0 * (yellow_line_reference - abs(yellow_angle))  
            correction = yellow_line_correction             
            #print(f"Yellow line correction: {yellow_line_correction}, DIR: {dire(yellow_line_correction)}")
        if white_angle is not None and abs(abs(white_angle) - white_line_reference) > 0.1\
                and (yellow_angle is None or abs(yellow_angle) > abs(white_angle)):
            white_line_correction =  -1.0 * (white_line_reference - abs(white_angle))  
            correction = white_line_correction if correction == 0.0 else mean_correction(white_line_correction, correction)
            #print(f"White line correction: {white_line_correction}, DIR: {dire(white_line_correction)}")
       
        speed = FORWARD_SPEED/(3.0 + abs(correction))
        
        print(f"V1: {speed}, V2: {correction}, DIR: {dire(correction)}")
        self.current_speed = speed, correction

        return self.current_speed

    def move_in_lane(self, aruco_pose, lines): 
        if lines is None:
            return FORWARD_WITH_CAUTION_SPEED, 0
         
        aruco_move = self.compute_aruco_move(aruco_pose)
        lane_move = self.compute_lane_following_move(lines)
#         move = np.mean([aruco_move, lane_move], axis=0)
        
        return self.adjust_speed(lane_move)


    def move(self, aruco_pose, lines): 

        v1, v2 = FORWARD_WITH_CAUTION_SPEED, 0.0
        
        if self.state == State.MOVING_IN_LANE: 
            v1, v2 = self.move_in_lane(aruco_pose, lines)
        else: 
            v1, v2 = self.take_action()

        return self.adjust_speed((v1, v2))      
