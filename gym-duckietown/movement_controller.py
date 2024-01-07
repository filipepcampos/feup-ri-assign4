from enum import Enum
import queue
import numpy as np
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

# np.array([[[0.5, 0, 0.2], [0.0, 0, 0.2], [-0.2, 0, 0.0], [-0.2, 0, -0.5]], 


#CURVE_RIGHT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, 0.00], [0.00, 0, 0.20], [0.50, 0, 0.20],],
#                        [[0.50, 0, -0.20], [0.30, 0, -0.20],[0.20, 0, -0.30],[0.20, 0, -0.50]]])



CURVE_RIGHT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, 0.00], [0.00, 0, 0.20], [0.50, 0, 0.20],],
#                        [[0.50, 0, -0.25], [0.30, 0, -0.15],[0.20, 0, -0.30],[0.20, 0, -0.50]]])
                        [[0.2, 0, -0.50], [0.2, 0, -0.30], [0.3, 0, -0.15], [0.5, 0, -0.25]]])

# original
CURVE_LEFT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, -0.20], [-0.30, 0, -0.20], [-0.50, 0, -0.10],],
                        [[0.2, 0, -0.50], [0.3, 0, 0.0], [-0.3, 0, 0.20], [-0.5, 0, 0.2]]])

#CURVE_LEFT_BEZIER = np.array([[[-0.20, 0, -0.50],[-0.20, 0, -0.30], [-0.30, 0, -0.20], [-0.50, 0, -0.20],],
#                        [[0.2, 0, -0.50], [0.3, 0, 0.10], [0.1, 0, 0.30], [-0.5, 0, 0.4]]])

avg_curve = lambda x, y: np.mean([x, y], axis=0)
avg_weighted_curve = lambda x, y, w: np.average([x, y], axis=0, weights=[w, 1 - w])


def bezier_curve_with_straight_ending(curve, timesteps, direction="left"):
    p0, p1, p2, p3 = curve[0], curve[1], curve[2], curve[3]
    p = lambda t: (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    
    # describe the curve until 0.8 of the total time
    curve_timesteps = int(0.8 * timesteps)
    curve_points = np.array([p(t) for t in np.linspace(0, 1, curve_timesteps)])

    # describe a straight line in 0.2 of the total time 
    first_point = curve_points[-1]
    direction = -1 if direction == "left" else 1
    straight_points = np.array([(first_point[0] + direction * i, 0, first_point[2]) for i in range(timesteps - curve_timesteps)])

    return np.concatenate((curve_points, straight_points))
    
def bezier_curve(curve, timesteps):
    p0, p1, p2, p3 = curve[0], curve[1], curve[2], curve[3]
    p = lambda t: (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
    
    return np.array([p(t) for t in np.linspace(0, 1, timesteps)])


def bezier_curve_derivative(curve, timesteps):
    p0, p1, p2, p3 = curve[0], curve[1], curve[2], curve[3]
    p_prime = lambda t: 3 * (1 - t)**2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t**2 * (p3 - p2)
    
    return np.array([p_prime(t) for t in np.linspace(0, 1, timesteps)])




TURN_LEFT_STEPS = 275
TURN_RIGHT_STEPS = 200


#left_curve_points = bezier_curve(avg_weighted_curve(CURVE_LEFT_BEZIER[0], CURVE_LEFT_BEZIER[1], 0.4), TURN_LEFT_STEPS, "left")
left_curve_points = bezier_curve_with_straight_ending(CURVE_LEFT_BEZIER[0], TURN_LEFT_STEPS, "left")
right_curve_points = bezier_curve(CURVE_RIGHT_BEZIER[0], TURN_RIGHT_STEPS)

left_curve_derivative = bezier_curve_derivative(CURVE_LEFT_BEZIER[0], TURN_LEFT_STEPS)
right_curve_derivative = bezier_curve_derivative(CURVE_RIGHT_BEZIER[1], TURN_RIGHT_STEPS)


class ArucoMovementController:
    def __init__(self): 
        self.direction = 0

        print(f"Left curve points: {left_curve_points.shape}, Right curve points: {right_curve_points.shape}")
        self.last_aruco_angle = 0
        self.current_speed = FORWARD_SPEED, 0.0

        self.action_queue = queue.Queue()
        self.state = State.MOVING_IN_LANE

        # DEBUG
        self.action_queue.put(Action.TURN_LEFT)
        self.action_queue.put(Action.TURN_RIGHT)

        self.action_step = 0
        self.safety_distance = 0.5


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

        if self.action_queue.empty():
            return State.MOVING_IN_LANE

        return action_to_state[self.action_queue.get()]


    def at_intersection(self, red_line): 
        if red_line is None:
            return False

        x1, _, x2, _ = red_line
        is_intersection = abs(x1 - x2) > 300 # TODO: use area of the rectangle

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

#        if white_line is not None and yellow_line is None:
#            x1, _, x2, _ = white_line
#            return x1 > 2 * FRAME_WIDTH / 3 or x2 > 2 * FRAME_WIDTH / 3
        
#        x1, _, x2, _ = white_line
#        x3, _, x4, _ = yellow_line[0]

#        return  abs(x1 - x3) < 450 and abs(x2 - x4) < 450
        return True

    def take_curve(self, curve_type):
        # take the bezier curve and use it to move across 100 frames
       curve = left_curve_points if curve_type == Action.TURN_LEFT else right_curve_points

#        curve_derivative = left_curve_derivative if curve_type == Action.TURN_LEFT else right_curve_derivative


       next_move = curve[self.action_step]
       prev_move = curve[self.action_step - 1] if self.action_step > 0 else next_move
       
       print(f"Next move: {next_move}, Prev move: {prev_move}")
       angle = math.atan2(next_move[2] - prev_move[2], next_move[0] - prev_move[0])
       sg = 1 if curve_type == Action.TURN_LEFT else -1
       
       v1 = 0.1 # curve_derivative[self.action_step][0] * FORWARD_SPEED
       v2 = (FORWARD_SPEED) * abs(np.cos(angle)) * sg * (1.0 if curve_type == Action.TURN_LEFT else 1.9)
#       v2 = (FORWARD_SPEED * 1.2) * sg * abs(curve_derivative[self.action_step][2]) 
       print(f"Angle: {RAD_TO_DEG * angle}, V1: {v1}, V2: {v2}")

#       print(f"V1: {v1}, V2: {v2}, curve_derivative: {curve_derivative[self.action_step]}")

       self.action_step += 1
       print(f"Action step: {self.action_step}")
       if self.action_step == len(curve):
           self.state = State.MOVING_IN_LANE
           self.action_step = 0

       return v1, v2

    def take_action(self):
        # TODO: 3way, 4way intersection
        print(f"CURRENT STATE: {self.state}")
        if self.state == State.GOING_FORWARD: 
            return FORWARD_SPEED, 0
        elif self.state == State.CURVING_LEFT:
             return self.take_curve(Action.TURN_LEFT)
        elif self.state == State.CURVING_RIGHT:
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


    def compute_line_side(self, line, color="white"):
        x1, _, x2, _ = line
        
        # get percentage of line on the left side of the frame

        p_left = (x1 - FRAME_WIDTH / 2) / (FRAME_WIDTH / 2)
        p_right = (x2 - FRAME_WIDTH / 2) / (FRAME_WIDTH / 2)
        # return 0 if p_left > p_right else 1
    
        if color == "white":
            return 0 if (x1 < FRAME_WIDTH / 3 and x2 < FRAME_WIDTH / 3) else 1    
        elif color == "yellow":
           return 0 if (x1 < FRAME_WIDTH / 2 and x2 < FRAME_WIDTH / 2) else 1
    

    def compute_lane_following_move(self, lines):
            white_line_info, yellow_line_info = lines
            white_line, white_angle = white_line_info
            yellow_line, yellow_angle = yellow_line_info

            print(f"White angle: {RAD_TO_DEG * white_angle if white_angle is not None else None}\
                    Yellow angle: {RAD_TO_DEG * yellow_angle if yellow_angle is not None else None}")

            dire = lambda x: "LEFT" if x > 0 else "RIGHT"           
            to_deg = lambda x: x * RAD_TO_DEG
            
            if white_line is None and yellow_line is None:
                return FORWARD_WITH_CAUTION_SPEED, 0.0
            
            white_angle_correction = 0.0
            if white_angle is not None: 
                white_angle = white_angle if white_angle > 0 else np.pi/2 + abs(white_angle)
                if self.compute_line_side(white_line, "white") == 0: # left
                    print(f"line side: {self.compute_line_side(white_line, 'white')}, whitee line: {white_line}")
                    white_angle_correction = -3.0
                else:
                    if 90 < (a := to_deg(white_angle)) <= 100:
                        white_angle_correction = -1.5 if yellow_line is not None else 1.5
                    elif 100 < a <= 125:
                        white_angle_correction = -0.5 if yellow_line is not None else 1.5
                   # elif 130 < a <= 135:
                   #     white_angle_correction = 0.0
                    elif 125 < a <= 170:
                        white_angle_correction = 1.0
                    elif 170 < a <= 180:
                        white_angle_correction = 2.0
                    elif a > 180:
                        white_angle_correction = 3.0
    
            yellow_angle_correction = 0.0       
            if yellow_angle is not None: 
                yellow_angle = yellow_angle if yellow_angle > 0 else np.pi + yellow_angle
                if self.compute_line_side(yellow_line, "yellow") == 1: # right 
                    yellow_angle_correction = 3.0
                else:
                    if 0 < (a := to_deg(yellow_angle)) <= 30:
                        yellow_angle_correction = -0.5 if white_line is not None else -1.0
                    elif 30 < a <= 50: 
                        yellow_angle_correction = -1.0
                   # elif 40 < a <= 45:
                   #     yellow_angle_correction = 0.0
                    elif 50 < a <= 90:
                        yellow_angle_correction = 1.0 if white_line is not None else -0.5
                    elif a >= 90:
                        yellow_angle_correction = 2.0 if white_line is not None else -0.5
    
            print(f"White angle: {RAD_TO_DEG * white_angle if white_angle is not None else None}\
                    Yellow angle: {RAD_TO_DEG * yellow_angle if yellow_angle is not None else None}")

            new_dir = white_angle_correction + yellow_angle_correction
            print(f"V1: {FORWARD_SPEED}, V2: {new_dir}, DIR: {dire(new_dir)}")

            return 0.6 * FORWARD_SPEED, 0.6 * new_dir
    
    

    def move_in_lane(self, aruco_pose, lines): 
        if lines is None:
            return FORWARD_WITH_CAUTION_SPEED, 0
         
        aruco_move = self.compute_aruco_move(aruco_pose)
        lane_move = self.compute_lane_following_move(lines)
#         move = np.mean([aruco_move, lane_move], axis=0)
        
        return lane_move


    def move(self, aruco_pose, lines): 

        v1, v2 = FORWARD_WITH_CAUTION_SPEED, 0.0
        
        if self.state == State.MOVING_IN_LANE: 
            v1, v2 = self.move_in_lane(aruco_pose, lines)
            v1, v2 = self.adjust_speed((v1, v2))
        else: 
            v1, v2 = self.take_action()

#        return self.adjust_speed((v1, v2))      
        return v1, v2
