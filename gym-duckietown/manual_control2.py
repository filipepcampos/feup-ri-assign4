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
# from experiments.utils import save_img

from edge_detector import EdgeDetector
from lib.movement_controller import ArucoMovementController
from lib.guide_detect import ArUcoBotDetector

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
guide_bot_detector = ArUcoBotDetector()
movement_controller = ArucoMovementController(guide_bot_detector=guide_bot_detector)

white_line_history = deque(maxlen=4)
yellow_line_history = deque(maxlen=4)


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
    guide_bot_detector.update(frame)
    guide_bot_detector.draw(frame)

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
#    print(f'Bot move: {bot_action[0]}, {bot_action[1]}')

    user_control = np.array([v1, v2])
    _, _, done, _ = env.step(bot_action)


    if key_handler[key.RETURN]:
        # Save frame as png
        frame = cv.cvtColor(obs, cv.COLOR_RGB2BGR)
        cv.imwrite("screen.png", frame)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
