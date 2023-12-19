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
import pyglet
from pyglet.window import key
import cv2 as cv
import os

from gym_duckietown.envs import DuckietownEnv
from detect_bb import eval_img_duckies

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", default=None)
parser.add_argument("--map-name", default="udem1")
parser.add_argument("--distortion", default=False, action="store_true")
parser.add_argument("--camera_rand", default=False, action="store_true")
parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
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

DATASET_PATH = "../dataset/"
IMAGES_PATH = os.path.join(DATASET_PATH, "images")
LABELS_PATH = os.path.join(DATASET_PATH, "labels")


def get_latest_image_id():
    # Get all images names
    images_names = os.listdir(IMAGES_PATH)

    # Get latest image id
    latest_image_id = -1
    for image_name in images_names:
        image_id = int(image_name.split(".")[0])
        if image_id > latest_image_id:
            latest_image_id = image_id

    return latest_image_id


current_image_id = get_latest_image_id() + 1


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
    if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
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
    #print("step_count = %s, reward=%.3f" % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        save_screenshot(obs)

    if done:
        print("done!")
        env.reset()
        env.render()

    env.render()


def normalize_angle(angle):
    # Normalize angle between -pi and pi
    angle = angle % (2 * np.pi)
    if angle > np.pi:
        angle -= 2 * np.pi
    return angle


def write_label(class_id, relative_pos, relative_angle, bounding_boxes):
    if len(bounding_boxes) == 0:
        label = ""
    else:
        label = f"{class_id} {' '.join(bounding_boxes)} {' '.join(relative_pos)} {relative_angle}"

    with open(os.path.join(LABELS_PATH, str(current_image_id) + ".txt"), "w") as f:
        f.write(label)


def save_screenshot(obs):
    global current_image_id
    img = Image.fromarray(obs)
    img.save(os.path.join(IMAGES_PATH, str(current_image_id) + ".png"))

    frame = cv.cvtColor(obs, cv.COLOR_RGB2BGR)

    # Get main duckiebot position and angle
    main_duckiebot = [env.cur_pos, env.cur_angle]

    # Get world objects
    world_objects = env.objects

    # Get other duckiebot position and angle
    other_duckiebot = [world_objects[0].pos, world_objects[0].angle]

    # Get relative position and angle
    relative_pos = np.array(main_duckiebot[0]) - np.array(other_duckiebot[0])
    relative_angle = main_duckiebot[1] - other_duckiebot[1]

    # Normalize relative angle between -pi and pi
    relative_angle = normalize_angle(relative_angle)

    # Print relative position and angle
    print("Relative position: ", relative_pos)
    print("Relative angle: ", relative_angle)

    # Get bounding boxes
    bounding_boxes = eval_img_duckies(frame)

    # Print bounding boxes
    print("Bounding boxes: ", bounding_boxes)

    # Write label
    write_label(relative_pos, relative_angle, bounding_boxes)

    # Increment image id
    current_image_id += 1


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
