#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
from PIL import Image
import argparse
import sys

import gym
import numpy as np
import cv2 as cv
import pyglet
from pyglet.window import key
import queue
from enum import Enum
from gym_duckietown.envs import DuckietownEnv
# from experiments.utils import save_img

from aruco_detector import ArucoDetector
from edge_detector import EdgeDetector
from movement_controller import ArucoMovementController

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


aruco_detector = ArucoDetector(
    np.array(
        [
            [305.5718893575089, 0, 303.0797142544728],
            [0, 308.8338858195428, 231.8845403702499],
            [0, 0, 1],
        ]
    ),
    np.array([-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]),
)

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
movement_controller = ArucoMovementController()

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

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

    av1, v2 = movement_controller.adjust_speed(action)

    # Speed boost
    if key_handler[key.LSHIFT]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    # print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    # dentro da função
    frame = cv.cvtColor(
        obs, cv.COLOR_RGB2BGR
    )  # conversão PIL para OPENCV https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format, don't ask me

    # TODO:
    # Remove colors which are not yellow or white
    # Convert to HSV
    converted = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   
    # Define masks
    mask_white, mask_yellow, mask_red = edge_detector.define_masks(converted)
    mask_white, mask_yellow, mask_red = edge_detector.erode_and_dilate((mask_white, mask_yellow, mask_red))

    cv.imshow("mask_white", mask_white)
    cv.imshow("mask_yellow", mask_yellow)
    cv.imshow("mask_red", mask_red)

    # Get lanes by detecting edges 
    edges_white, edges_yellow, edges_red = edge_detector.detect_edges((mask_white, mask_yellow, mask_red))   

    cv.imshow("edges_white", edges_white)
    cv.imshow("edges_yellow", edges_yellow)
    cv.imshow("edges_red", edges_red)

    # Get lines from edges
    white_lines, yellow_lines, red_lines = edge_detector.detect_lines((edges_white, edges_yellow, edges_red), 
                                                                      [10,50, 100])
    
    if white_lines is not None:
        white_lines = edge_detector.remove_horizontal_lines(white_lines)
    if red_lines is not None:
        print("red lines")
        red_lines = edge_detector.get_horizontal_lines(red_lines)

    # Get the average line
    if white_lines is not None:
        white_line = edge_detector.get_average_line(white_lines)
        edge_detector.draw_line(frame, white_line, (0, 0, 255))
        white_angle = edge_detector.get_angle(white_line)
    if yellow_lines is not None:
        yellow_line = edge_detector.get_average_line(yellow_lines)
        edge_detector.draw_line(frame, yellow_line[0], (255, 0, 0))
        yellow_angle = edge_detector.get_angle(yellow_line[0])
    if red_lines is not None:
        red_line = edge_detector.get_average_line(red_lines)
        edge_detector.draw_line(frame, red_line, (0, 255, 0))
        movement_controller.at_intersection(red_line)

    cv.imshow("frame", frame)
    cv.waitKey(1)
    
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
   
   

   # Draw markers
    if ids is not None:
        rvecs, tvecs = aruco_detector.estimatePose(corners)
        angle = rvecs[0][2]
        distance = tvecs[0][2]
        
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw line with angle
        marker_coordinates = corners[0]
        # Get center_point
        center_point = np.mean(marker_coordinates, axis=1, dtype=np.int32)[0]

        if angle:
            print(f'Angle: {angle},\n\
                    angle + 90: {angle + np.pi / 2},\n\
                    detected curve: {movement_controller.detect_curve(angle)}')

            x1, y1 = center_point[0], center_point[1]
            angle += np.pi / 2
            x2, y2 = int(np.cos(angle) * 100)+x1, int(np.sin(angle) * 100)+y1
            
            # Draw an arrow
#            cv.arrowedLine(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)

    # print(f'Detected curve {movement_controller.detect_curve(angle)}')

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
