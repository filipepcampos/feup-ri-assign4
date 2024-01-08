from .movement_state import MovementState, State, Action
from .constants import *
from .bezier import left_curve, right_curve


import numpy as np
import math




class MovementActor:
    def __init__(self, state: MovementState):
        self.state = state

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


    def go_straight(self):
        self.state.increment_action_step()
        self.state.check_going_forward_end()

        print("STEP: ", self.state.action_step)

        return FORWARD_SPEED, 0.0


    def take_curve(self, curve_type):
        # take the bezier curve and use it to move across 100 frames
       curve = left_curve if curve_type == Action.TURN_LEFT else right_curve

       next_move, prev_move = self.state.get_curve_moves(curve)
       
       print(f"Next move: {next_move}, Prev move: {prev_move}")
       angle = math.atan2(next_move[2] - prev_move[2], next_move[0] - prev_move[0])
       sign = 1 if curve_type == Action.TURN_LEFT else -1
       
       v1 = 0.1 
       v2 = (FORWARD_SPEED) * abs(np.cos(angle)) * sign * (1.0 if curve_type == Action.TURN_LEFT else 1.9)

       self.state.increment_action_step()
       self.state.check_curve_end(curve)

       return v1, v2

    def take_action(self):
        # TODO: 3way, 4way intersection
        print(f"CURRENT STATE: {self.state}")

        if self.state == State.CURVING_LEFT:
             return self.take_curve(Action.TURN_LEFT)
        elif self.state == State.CURVING_RIGHT:
            return self.take_curve(Action.TURN_RIGHT)
        elif self.state == State.GOING_FORWARD:
            return self.go_straight()
        
        return 0, 0
    
    def compute_line_side(self, line, color="white"):
        x1, _, x2, _ = line
        
        if color == "white":
            return 0 if (x1 < FRAME_WIDTH / 3 and x2 < FRAME_WIDTH / 3) else 1    
        elif color == "yellow":
           return 0 if (x1 < FRAME_WIDTH / 2 and x2 < FRAME_WIDTH / 2) else 1
    

    def compute_lane_following_move(self, lines):
        white_line_info, yellow_line_info = lines
        white_line, white_angle = white_line_info
        yellow_line, yellow_angle = yellow_line_info

        direction = lambda x: "LEFT" if x > 0 else "RIGHT" if x < 0 else "FORWARD"         
        to_deg = lambda x: x * RAD_TO_DEG
        
        if white_line is None and yellow_line is None:
            return FORWARD_WITH_CAUTION_SPEED, 0.0
        


        # ESQ + || DIR - 
        def get_white_angle_correction(white_line, white_angle, yellow_line):
            white_angle_correction = 0.0
            if white_angle is not None: 
                white_angle = white_angle if white_angle > 0 else np.pi/2 + abs(white_angle)
                print(f"White angle: {RAD_TO_DEG * white_angle if white_angle is not None else None}")
                if self.compute_line_side(white_line, "white") == 0: # left
                    white_angle_correction = -3.0
                else:
                    if 90 < (a := to_deg(white_angle)) <= 100:
                        white_angle_correction = -1.5 if yellow_line is not None else 3.0
                    elif 100 < a <= 125:
                        white_angle_correction = -1.0 if yellow_line is not None else 3.0
                    elif 125 < a <= 130:
                        white_angle_correction = -0.75
                    elif 130 < a <= 140:
                        white_angle_correction = 0.0
                    elif 145 < a <= 170:
                        white_angle_correction = 1.3
                    elif 170 < a <= 180:
                        white_angle_correction = 1.75
                    elif a > 180:
                        white_angle_correction = 3.0
            return white_angle_correction


        def get_yellow_angle_correction(yellow_line, yellow_angle, white_line):
            yellow_angle_correction = 0.0       
            if yellow_angle is not None: 
                yellow_angle = yellow_angle if yellow_angle > 0 else np.pi + yellow_angle
                print(f"Yellow angle: {RAD_TO_DEG * yellow_angle if yellow_angle is not None else None}")
                # if self.compute_line_side(yellow_line, "yellow") == 1: # right 
                #     yellow_angle_correction = 3.0
                # else:
                if 0 < (a := to_deg(yellow_angle)) <= 30:
                    yellow_angle_correction = -1.0 if white_line is not None else -1.0
                elif 30 < a <= 40: 
                    yellow_angle_correction = -0.5
                elif 40 < a <= 50:
                    yellow_angle_correction = 0.0
                elif 55 < a <= 90:
                    yellow_angle_correction = 1.0 if white_line is not None else -2.5
                elif a >= 90:
                    yellow_angle_correction = 2.0 if white_line is not None else -2.5
            return yellow_angle_correction
        
        white_angle_correction = get_white_angle_correction(white_line, white_angle, yellow_line)
        yellow_angle_correction = get_yellow_angle_correction(yellow_line, yellow_angle, white_line)


        print(f"White angle correction: {white_angle_correction}, Yellow angle correction: {yellow_angle_correction}")
        new_dir = white_angle_correction + yellow_angle_correction
        print(f"V1: {FORWARD_SPEED}, V2: {new_dir}, DIR: {direction(new_dir)}")

        return 0.6 * FORWARD_SPEED, 0.6 * new_dir

    def move_in_lane(self, lines): 
        if lines is None:
            return FORWARD_WITH_CAUTION_SPEED, 0
         
        # aruco_move = self.compute_aruco_move(aruco_pose)
        lane_move = self.compute_lane_following_move(lines)
#         move = np.mean([aruco_move, lane_move], axis=0)
        
        return lane_move