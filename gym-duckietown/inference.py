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
import pyglet
from pyglet.window import key
import torch
import cv2 as cv
import yaml

from gym_duckietown.envs import DuckietownEnv

import sys
sys.path.insert(0, './feup-ri-assign4-model')
from models.yolo import Model
from models.common import DetectMultiBackend

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


# hyp = None
# with open("weights/hyp.yaml", errors='ignore') as f:
#     hyp = yaml.safe_load(f)  # load hyps dict

# ckpt = torch.load("weights/best.pt", map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
# model = Model(ckpt['model'].yaml, ch=3, nc=2, anchors=hyp.get('anchors'))  # create
# model.eval()
model = DetectMultiBackend("weights/best.pt")
model.warmup(imgsz=(1, 3, 256, 256))

dist_queue = deque(maxlen=3)
rot_queue = deque(maxlen=3)

previous_distance = 2

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    wheel_distance = 0.102
    min_rad = 0.08

    global previous_distance # TODO: this is a hack

    action = np.array([0.0, 0.0])

    if previous_distance == 0:
        action[0] = 0
    elif previous_distance == 1:
        action[0] = 0.2
    elif previous_distance == 2:
        action[0] = 0.35
    elif previous_distance == 3:
        action[0] = 0.6
    elif previous_distance == 4:
        action[0] = 1.5

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

    if True:
        resized_frame = cv.resize(obs, (256, 256))

        img = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
        img = img[None] # expand dimension to batch size 1    
        results = model(img)[0][0]

        detections = []
        conf_thresh = 0.35
        best_duckiebot_detection = None
        best_duckiebot_conf = 0

        for result in results:
            conf = result[4]
            class_label = 0 if result[5] > result[6] else 1

            if conf > conf_thresh:
                if class_label == 0:
                    detections.append((class_label, result))

            if conf > best_duckiebot_conf and class_label == 1 and conf > 0.1:
                best_duckiebot_detection = result
                best_duckiebot_conf = conf
        if best_duckiebot_detection is not None:
            detections.append((1, best_duckiebot_detection))

        resized_frame = cv.cvtColor(resized_frame, cv.COLOR_RGB2BGR)

        # Draw bounding boxes
        for detection in detections:
            label, dets = detection
            center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = dets
            x1, y1 = center_x - width / 2, center_y - height / 2
            x2, y2 = center_x + width / 2, center_y + height / 2
            cv.rectangle(resized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Draw best duckiebot information
        def draw_info():
            if best_duckiebot_detection is None:
                return
            center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = best_duckiebot_detection

            dist_queue.append([dist1, dist2, dist3, dist4, dist5])
            rot_queue.append([rot1, rot2, rot3, rot4, rot5])

            x = int(256 / 2 + 20)

            cv.putText(resized_frame, "distance", (x, 230), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 127, 0), 1)

            values = [sum(x) / len(x) for x in zip(*dist_queue)]
            # values = torch.softmax(torch.tensor(values), dim=0)
            for dist, label in zip(values, ["VC", "C", "M", "F", "VF"]):
                cv.rectangle(resized_frame, (x, 200), (x + 10, 200-int(50 * dist)), (0, 127, 0), 2)
                cv.putText(resized_frame, label, (x, 210), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 127, 0), 1)
                x += 10
            
            x = int(256 / 2 - 20 - 10*5)
            cv.putText(resized_frame, "rot", (x, 230), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            values = [sum(x) / len(x) for x in zip(*rot_queue)]
            # values = torch.softmax(torch.tensor(values), dim=0)
            
            for rot, label in zip(values, ["VL", "L", "M", "R", "VR"]):
                cv.rectangle(resized_frame, (x, 200), (x + 10, 200-int(50 * rot)), (0, 0, 255), 2)
                cv.putText(resized_frame, label, (x, 210), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1)
                x += 10

        draw_info()
        

        if best_duckiebot_detection is not None:
            center_x, center_y, width, height, conf, class1, class2, dist1, dist2, dist3, dist4, dist5, rot1, rot2, rot3, rot4, rot5 = best_duckiebot_detection
            previous_distance = np.argmax([dist1, dist2, dist3, dist4, dist5])

        cv.imshow("frame", resized_frame)
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
