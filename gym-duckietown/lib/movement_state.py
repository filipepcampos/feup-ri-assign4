
from enum import Enum
from .bezier import get_left_curve, get_right_curve
import queue


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
    Action.STOP: State.STOPPED
}




class MovementState: 
    action_to_state = {
        Action.TURN_LEFT: State.CURVING_LEFT,
        Action.TURN_RIGHT: State.CURVING_RIGHT,
        Action.STOP: State.STOPPED
    }
    
    def __init__(self):
        self.state = State.MOVING_IN_LANE
        self.action_step = 0
        self.action_queue = queue.Queue()
        self.last_aruco_angle = 0

        # DEBUG
        self.action_queue.put(Action.TURN_RIGHT)
        self.action_queue.put(Action.TURN_RIGHT)

    def get_curve_moves(self, curve):
       next_move = curve[self.action_step]
       prev_move = curve[self.action_step - 1] if self.action_step > 0 else next_move
       return (next_move, prev_move)

    def check_curve_end(self, curve):
        if self.action_step == len(curve):
           self.state = State.MOVING_IN_LANE
           self.action_step = 0

    def deliberate_action(self): 
        if self.action_step != 0: 
            return self.state
        elif self.action_queue.empty():
            self.state = State.MOVING_IN_LANE
        else:
            self.state = action_to_state[self.action_queue.get()]
            
        return self.state
        
    def increment_action_step(self):
        self.action_step += 1

    def __eq__(self, other):
        if isinstance(other, State):
            return self.state == other
        return self.state == other.state
    
    def __str__(self):
        return str(self.state)

