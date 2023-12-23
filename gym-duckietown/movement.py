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

from gym_duckietown.envs import DuckietownEnv

# from experiments.utils import save_img

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


class ArucoDetector:
    def __init__(
        self,
        camera_matrix: np.array,
        distortion_coefs: np.array,
    ):
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        self.parameters = cv.aruco.DetectorParameters()
        #self.parameters.maxErroneousBitsInBorderRate = 0.9


        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.camera_matrix = camera_matrix
        self.distortion_coefs = distortion_coefs

    def detectMarkers(self, image):
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(image)
        return corners, ids, rejectedImgPoints

    def estimateAngle(self, corners, ids):
        if ids is not None:
            rvecs, tvecs, trash = self.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.distortion_coefs
            )
            return rvecs[0][2]
        else:
            return None

    def estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())

        Shamelessly sourcedfrom:
        https://stackoverflow.com/questions/76802576/how-to-estimate-pose-of-single-marker-in-opencv-python-4-8-0
        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        trash, rvecs, tvecs = [], [], []

        for c in corners:
            n, R, t = cv.solvePnP(
                marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE
            )
            rvecs.append(R)
            tvecs.append(t)
            trash.append(n)
        return rvecs, tvecs, trash


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

from collections import deque
previous_angles = deque(maxlen=10)
previous_white_lines = deque(maxlen=5)
previous_yellow_lines = deque(maxlen=5)

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

    v1 = action[0]
    v2 = action[1]
    # Limit radius of curvature
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (
        min_rad - wheel_distance / 2.0
    ):
        # adjust velocities evenly such that condition is fulfilled
        delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
        v1 += delta_v
        v2 -= delta_v

    action[0] = v1
    action[1] = v2

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
    def detect_lanes(frame):
        # Remove colors which are not yellow or white
        # Convert to HSV
        converted = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # Define color ranges
        lower_white, upper_white = np.array([0, 0, 200]), np.array([175, 175, 255])
        lower_yellow, upper_yellow = np.array([15, 75, 75]), np.array([35, 255, 255])
        # Create masks
        mask_white = cv.inRange(converted, lower_white, upper_white)
        mask_yellow = cv.inRange(converted, lower_yellow, upper_yellow)

        # Erode and dilate masks
        erode_kernel = np.ones((5, 5), np.uint8)
        dilate_kernel = np.ones((9, 9), np.uint8)
        mask_white = cv.erode(mask_white, erode_kernel, iterations=2)
        mask_yellow = cv.dilate(mask_yellow, dilate_kernel, iterations=1)

        # Get lanes by detecting edges
        edges_white = cv.Canny(mask_white, 100, 200)
        edges_yellow = cv.Canny(mask_yellow, 100, 200)

        # Get lines from edges
        white_lines = cv.HoughLinesP(
            edges_white, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=5
        )
        yellow_lines = cv.HoughLinesP(
            edges_yellow, 1, np.pi / 180, 100, minLineLength=30, maxLineGap=50
        )

        # Remove horizontal lines
        if white_lines is not None:
            white_lines = white_lines[abs(white_lines[:, :, 1] - white_lines[:, :, 3]) > 50]

        # Remove yellow lines that are not near the bottom of the image
        if yellow_lines is not None:
            yellow_lines = yellow_lines[
                yellow_lines[:, :, 1] > frame.shape[0] * 0.6
            ]

        # Get the average line
        white_line, yellow_line = None, None
        if white_lines is not None and len(white_lines) > 0:
            white_line = np.mean(white_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = white_line
        if yellow_lines is not None and len(yellow_lines) > 0:
            yellow_line = np.mean(yellow_lines, axis=0, dtype=np.int32)
            x1, y1, x2, y2 = yellow_line
            # cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        return white_line, yellow_line
    
    white_line, yellow_line = detect_lanes(frame)
    if white_line is not None:
        previous_white_lines.append(white_line)
    if yellow_line is not None:
        previous_yellow_lines.append(yellow_line)
    average_white_line = np.mean(previous_white_lines, axis=0, dtype=np.int32)
    average_yellow_line = np.mean(previous_yellow_lines, axis=0, dtype=np.int32)

    if white_line is not None:
        x1, y1, x2, y2 = average_white_line
        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
    if yellow_line is not None:
        x1, y1, x2, y2 = average_yellow_line
        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)

    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(frame)

    angle = aruco_detector.estimateAngle(corners, ids)
    
    if angle is not None:
        previous_angles.append(angle)
    average_angle = np.mean(previous_angles)

    # Draw markers
    if ids is not None:
        cv.aruco.drawDetectedMarkers(frame, corners, ids)

        # Draw line with angle
        marker_coordinates = corners[0]
        # Get center_point
        center_point = np.mean(marker_coordinates, axis=1, dtype=np.int32)[0]

        if angle:
            #x1, y1 = frame.shape[1] // 2, frame.shape[0] // 2
            x1, y1 = center_point[0], center_point[1]
            angle += np.pi / 2
            x2, y2 = int(np.cos(average_angle) * 100)+x1, int(np.sin(average_angle) * 100)+y1
            
            # Draw an arrow
            cv.arrowedLine(frame, (x2, y2), (x1, y1), (0, 255, 0), 2)

    if angle:
        print(f"Angle: {angle}, Average: {np.mean(previous_angles)}")
    else:
        print(f"Not detected")

    if key_handler[key.RETURN]:
        # Save frame as png
        frame = cv.cvtColor(obs, cv.COLOR_RGB2BGR)
        cv.imwrite("screen.png", frame)


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
