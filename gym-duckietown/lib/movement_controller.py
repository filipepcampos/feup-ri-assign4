from .movement_state import MovementState, State, Action
from .movement_actor import MovementActor
from .constants import *
from .guide_detect import GuideBotDetector, Distance, Direction
import numpy as np


guidebotdetection_to_action = {
    Direction.LEFT : Action.TURN_LEFT,
    Direction.RIGHT : Action.TURN_RIGHT,
    Direction.CENTER : Action.GO_FORWARD,
}





class ArucoMovementController:
    def __init__(self, guide_bot_detector: GuideBotDetector): 
        self.last_aruco_angle = 0
        self.safety_distance = 0.5

        self.state = MovementState()
        self.movement_actor = MovementActor(self.state)
        self.guide_bot_detector = guide_bot_detector


    def at_intersection(self, red_line): 
        if red_line is None:
            return False

        x1, _, x2, _ = red_line
        is_intersection = abs(x1 - x2) > 300 

        if self.state.action_step == 0 and is_intersection:
            self.state.deliberate_action()
            self.state.increment_action_step()

        return is_intersection     

    def in_lane(self, white_line, yellow_line):
        if yellow_line is None and white_line is None:
            return False
        
        if white_line is None and yellow_line is not None:
            x1, _, x2, _ = yellow_line
            return x1 > FRAME_WIDTH / 3 or x2 > FRAME_WIDTH / 3

        return True
    
    def is_taking_action(self):
        return self.state.action_step != 0



    def move(self, lines): 
        v1, v2 = FORWARD_WITH_CAUTION_SPEED, 0.0
        
        print(f"CURRENT DISTANCE: {self.guide_bot_detector.distance}")
        print(f"CURRENT DIRECTION: {self.guide_bot_detector.direction}")
        print(f"CURRENT ACTIONS IN QUEUE: {self.state.action_queue.queue}")

        # update queue 
        if self.guide_bot_detector.direction != Direction.NONE:
            self.state.action_queue.queue.clear()
            self.state.action_queue.put(guidebotdetection_to_action[self.guide_bot_detector.direction])

        if self.guide_bot_detector.distance == Distance.CLOSE or self.guide_bot_detector.distance == Distance.VERY_CLOSE:
            print("GUIDEBOT IS CLOSE: EMERGENCY BRAKE")
            return 0.0, 0.0
        
        if self.state == State.MOVING_IN_LANE: 
            v1, v2 = self.movement_actor.move_in_lane(lines)
            # v1, v2 = self.movement_actor.adjust_speed((v1, v2))
        else: 
            v1, v2 = self.movement_actor.take_action()
   
        

        return v1, v2
