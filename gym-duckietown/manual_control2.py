#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys
from collections import deque

import gym
import numpy as np
import cv2 as cv
import pyglet
from pyglet.window import key
import queue
from enum import Enum
from gym_duckietown.envs import DuckietownEnv
import math
# from experiments.utils import save_img

from edge_detector import EdgeDetector
from lib.movement_controller import ArucoMovementController
from lib.guide_detect import ArUcoBotDetector, YOLOBotDetector

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument(
    "--draw-curve", action="store_true", help="draw the lane following curve"
)
parser.add_argument(
    "--draw-bbox", action="store_true", help="draw collision detection bounding boxes"
)
parser.add_argument(
    "--domain-rand", action="store_true", help="enable domain randomization"
)
parser.add_argument(
    "--dynamics_rand", action="store_true", help="enable dynamics randomization"
)
parser.add_argument(
    "--frame-skip", default=1, type=int, help="number of frames to skip"
)
parser.add_argument("--seed", default=1, type=int, help="seed")
args = parser.parse_args()

if args.env_name and args.env_name.find("Duckietown") != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
        camera_rand=args.camera_rand,
        dynamics_rand=args.dynamics_rand,
        max_steps=9999999,
    )
else:
    env = gym.make(args.env_name)



env.reset()
env.render()


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print("RESET")
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)


edge_detector = EdgeDetector()

guide_bot_detector_aruco = ArUcoBotDetector()
guide_bot_detector_obj_detection = YOLOBotDetector()
movement_controller = ArucoMovementController(guide_bot_detector=guide_bot_detector_aruco)


white_line_history = deque(maxlen=4)
yellow_line_history = deque(maxlen=4)


info_record = []


def normalize_angle(angle):
    # Normalize angle between -pi and pi
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle

def get_relative_angle_and_pos(): 

     # Get main duckiebot position and angle
    main_duckiebot = [env.cur_pos, env.cur_angle]

    # Get world objects
    world_objects = env.objects

    # Get other duckiebot position and angle
    other_duckiebot = [world_objects[0].pos, world_objects[0].angle]

    # Get relative position and angle
    relative_pos = np.array(main_duckiebot[0]) - np.array(other_duckiebot[0])
    relative_distance = np.linalg.norm(relative_pos)

    relative_angle = main_duckiebot[1] - other_duckiebot[1]

    # Normalize relative angle between -pi and pi
    relative_angle = normalize_angle(relative_angle)

    return relative_angle, relative_distance
    

def get_distance_class(distance: float):
    if distance < 0.25:
        return 0
    elif distance < 0.35:
        return 1
    elif distance < 0.45:
        return 2
    elif distance < 0.55:
        return 3
    return 4

def distance_from_class(distance):
    if distance == 0: 
        return 0.25 / 2
    elif distance == 1:
        return (0.25 + 0.35) / 2
    elif distance == 2:
        return (0.35 + 0.45) / 2
    elif distance == 3:
        return (0.45 + 0.55) / 2
    return 0.55


def deg_to_rad(deg: float):
    return deg * math.pi / 180.0


def get_angle_class(angle: float): # TODO: This is not correct
    if angle < -deg_to_rad(70):
        return 0
    if angle < -deg_to_rad(25):
        return 1
    if angle < deg_to_rad(25):
        return 2
    if angle < deg_to_rad(70):
        return 3
    return 4


def angle_from_class(angle):
    if angle == 0:
        return -deg_to_rad(70) 
    elif angle == 1:
        return (-deg_to_rad(70) + deg_to_rad(25)) / 2
    elif angle == 2:
        return 0
    elif angle == 3:
        return (deg_to_rad(70) - deg_to_rad(25)) / 2
    return deg_to_rad(70)
    



def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action += np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action -= np.array([0.44, 0])
    if key_handler[key.LEFT]:
        action += np.array([0, 1])
    if key_handler[key.RIGHT]:
        action -= np.array([0, 1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    v1, v2 = movement_controller.movement_actor.adjust_speed(action)

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5



    obs = env.render_obs()

    # dentro da função
    frame = cv.cvtColor(
        obs, cv.COLOR_RGB2BGR
    )  # conversão PIL para OPENCV https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format, don't ask me

    # TODO:
    # Remove colors which are not yellow or white
    # Convert to HSV
    converted = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   
    white_line, yellow_line, red_line = edge_detector.get_lines(converted, frame)

    
    # white_line_history.append(white_line)
    # white_line = np.mean(white_line_history, axis=0)
    # yellow_line_history.append(yellow_line)

    # if len([i for i in yellow_line_history if i is not None]) < len(yellow_line_history):
    #     yellow_line = None
    # else:
    #     yellow_line = np.mean(yellow_line_history, axis=0)

    # ARUCO

    guide_bot_detector_aruco.update(frame)
    guide_bot_detector_obj_detection.update(frame)

    
    aruco_angle = guide_bot_detector_aruco.angle - np.pi/2
    aruco_distance = guide_bot_detector_aruco.distance_val
    
    yolo_angle = angle_from_class(guide_bot_detector_obj_detection.direction) - np.pi/2
    yolo_distance = distance_from_class(guide_bot_detector_obj_detection.distance)
    
    relative_angle, relative_pos = get_relative_angle_and_pos()


    # extract from 'aruco_angle': array([    0.10453]), 'aruco_distance': array([    0.41858])
    info_record.append({
        "relative_angle": relative_angle,
        "relative_pos": relative_pos,
        "aruco_angle": aruco_angle if aruco_angle is not None else None, 
        "aruco_distance": aruco_distance if aruco_distance is not None else None,
        "yolo_angle": yolo_angle,
        "yolo_distance": yolo_distance,
    })


    #guide_bot_detector_aruco.draw(frame)
    
    
    # frame = guide_bot_detector_obj_detection.draw(frame)

    cv.imshow("frame", frame)
    cv.waitKey(1)
    
    

    if movement_controller.at_intersection(red_line):
        print("\n==============================AT INTERSECTION deliberate action")
    elif movement_controller.in_lane(white_line, yellow_line): 
        print("In lane - following lane")
    elif movement_controller.is_taking_action():
        print("Taking action - following curve or going straight")

    line_info = None
    white_angle = None
    yellow_angle = None
    if white_line is not None: 
        white_angle = edge_detector.get_angle(white_line)

    if yellow_line is not None:
        # yellow_line = yellow_line[0]
        yellow_angle = edge_detector.get_angle(yellow_line)

    line_info = (white_line, white_angle), (yellow_line, yellow_angle)
     

    bot_action = movement_controller.move(line_info)
    print(f'Bot move: {bot_action[0]}, {bot_action[1]}')

    user_control = np.array([v1, v2])
    _, _, done, _ = env.step(bot_action)

    # print(info_record)

    if key_handler[key.RETURN]:
        # Save frame as png
        # frame = cv.cvtColor(obs, cv.COLOR_RGB2BGR)
        # cv.imwrite("screen.png", frame)
        with open("info_record.json", "w") as f:
            import json
            json.dump(info_record, f)

    cv.imshow("frame", frame)
    cv.waitKey(1)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
