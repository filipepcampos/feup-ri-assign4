#!/usr/bin/env python3

import cv2


ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(0, 128, 161), (10, 225, 225)]
DUCKS_WALKING_MASK = [(0, 33, 124), (24, 255, 255)]
DEBUG = True
ENGLISH = False
DEFAULT_VELOCITY = 0.28

"""
  Template for lane follow code was taken from eclass "Lane Follow Package".
  Author: Justin Francis
  Link: https://eclass.srv.ualberta.ca/mod/resource/view.php?id=6952069
"""
class LaneFollow():
  def __init__(self):
    self.stop = False  # true if it detected a stop line

    self.veh = 10 # TODO

    # PID Variables
    self.proportional = None

    ENGLISH = False
    if ENGLISH:
        self.offset = -240
    else:
        self.offset = 240

    DEFAULT_VELOCITY = 0.28
    self.velocity = DEFAULT_VELOCITY

    self.P = 0.04
    self.D = -0.004
    self.I = 0.008
    
    self.last_error = 0
    self.last_time = 0 # TODO
    


    # Initialize static parameters from camera info message
    # camera_info_msg = rospy.wait_for_message(f'/{self.veh}/camera_node/camera_info', CameraInfo)
    # self.camera_model = PinholeCameraModel()
    # self.camera_model.fromCameraInfo(camera_info_msg)
    # H, W = camera_info_msg.height, camera_info_msg.width

    # # find optimal rectified pinhole camera
    # rect_K, _ = cv2.getOptimalNewCameraMatrix(
    #   self.camera_model.K, self.camera_model.D, (W, H), self.rectify_alpha
    # )

    # # store new camera parameters
    # self._camera_parameters = (rect_K[0, 0], rect_K[1, 1], rect_K[0, 2], rect_K[1, 2])

    # self._mapx, self._mapy = cv2.initUndistortRectifyMap(
    #   self.camera_model.K, self.camera_model.D, None, rect_K, (W, H), cv2.CV_32FC1
    # )

    # Turn & action variables
    self.next_action = None
    self.left_turn_duration = 2.3
    self.right_turn_duration = 1
    self.straight_duration = 3
    self.started_action = None


  def distance_callback(self, msg):
    self.vehicle_distance = msg.data

  def detection_callback(self, msg):
    self.detecting_bot = msg.data


  def set_time(self, time):
    self.current_time = time

  def get_time(self):
    return self.current_time

  def callback(self, msg):
    img = msg
    crop = img[100:-1, :, :] # TODO: Be careful with this
    crop_width = crop.shape[1]
    self.frame_center = crop_width/2
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
    crop = cv2.bitwise_and(crop, crop, mask=mask)
    contours, hierarchy = cv2.findContours(mask,
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    
    # Search for lane in front
    max_area = 20
    max_idx = -1
    for i in range(len(contours)):
      area = cv2.contourArea(contours[i])
      if area > max_area:
        max_idx = i
        max_area = area

    if max_idx != -1:
      M = cv2.moments(contours[max_idx])
      try:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        self.proportional = cx - int(crop_width / 2) + self.offset
        
        if DEBUG:
          cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
          cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
          cv2.imshow("crop", crop)
      except:
        pass
    else:
      self.proportional = None

    # STOP LINE HANDLING
    # Mask for stop lines
    stopMask = cv2.inRange(hsv, STOP_LINE_MASK[0], STOP_LINE_MASK[1])
    # crop = cv2.bitwise_and(crop, crop, mask=stopMask)
    stopContours, _ = cv2.findContours(
      stopMask,
      cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_NONE
    )

    # Search for lane in front
    # max_area = self.stop_threshold_area
    # max_idx = -1
    # for i in range(len(stopContours)):
    #   area = cv2.contourArea(stopContours[i])
    #   if area > max_area:
    #     max_idx = i
    #     max_area = area

    # if max_idx != -1:
    #   M = cv2.moments(stopContours[max_idx])
    #   try:
    #     cx = int(M['m10'] / M['m00'])
    #     cy = int(M['m01'] / M['m00'])
    #     self.stop = False # TODO: I changed this to false
    #     if DEBUG:
    #       cv2.drawContours(crop, stopContours, max_idx, (0, 255, 0), 3)
    #       cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
    #   except:
    #     pass
    # else:
    self.stop = False


    # ==================================================================

    # if (self.vehicle_distance is not None and self.detecting_bot == True
    #     and self.vehicle_distance < 0.55
    #     and not self.check_duckie_down):
    #   self.check_duckie_down = True
    #   self.stop_starttime = self.get_time()

  def drive(self):
    # Determine next action, if we haven't already
    # Get available action from last detected april tag

    self.ducks_crossing = False
    self.check_duckie_down = False
    self.drive_around_bot = False

    return self._compute_lane_follow_PID()

    if self.ducks_crossing:
      return (0, 0)

    elif self.check_duckie_down:
      if self.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        return (0, 0)
      else:
        self.check_duckie_down = False
        self.last_stop_time = self.get_time()
        self.offset = -200
        self.drive_around_bot = True
        self.start_backup = None
        return None # TODO?
      
  
    elif self.stop:
      if self.get_time() - self.stop_starttime < self.stop_duration:
        # Stop
        self.velocity = DEFAULT_VELOCITY
        return (0, 0)
      else:
        # Do next action
        if self.next_action == "left":
          # Go left
          if self.started_action == None:
            self.started_action = self.get_time()
            return None # TODO?
          elif self.get_time() - self.started_action < self.left_turn_duration:
            return (self.velocity, 2.5)
          else:
            self.started_action = None
            self.next_action = None
            return None # TODO?
        elif self.next_action == "right":
          # lane following defaults to right turn
          if self.started_action == None:
            self.started_action = self.get_time()
            return None # TODO?
          elif self.get_time() - self.started_action < self.right_turn_duration:
            self._compute_lane_follow_PID()  # lane follow defaults to right turn
            return None # TODO?
          else:
            self.started_action = None
            self.next_action = None
            self.velocity = 0.25  # lower velocity so that we can see the next apriltag
            return (self.velocity, 0) # TODO? I'm uncertain about this one
        elif self.next_action == "straight":
          # Go straight
          if self.started_action == None:
            self.started_action = self.get_time()
            return None # TODO?
          elif self.get_time() - self.started_action < self.straight_duration:
            return (self.velocity, 0)
          else:
            self.started_action = None
            self.next_action = None
            return None # TODO?
        else:
          self.stop = False
          self.last_stop_time = self.get_time()
        return None # TODO?
    else: # do lane following
      return self._compute_lane_follow_PID()


  def _compute_lane_follow_PID(self):
    if self.drive_around_bot and self.get_time() - self.last_stop_time > self.drive_around_duration:
      self.drive_around_bot = False
      self.offset = 240

    omega = 0
    vel = 0

    if self.proportional is None:
        print("A")
        omega = 0
        self.last_error = 0
    else:
        print("B")
        # P Term
        P = -self.proportional * self.P

        # D Term
        d_time = (self.get_time() - self.last_time)
        d_error = (self.proportional - self.last_error) / d_time
        self.last_error = self.proportional
        self.last_time = self.get_time()
        D = d_error * self.D

        # I Term
        I = -self.proportional * self.I * d_time

        vel = self.velocity
        omega = P + I + D
        if DEBUG:
            print(self.proportional, P, D, omega, vel)

    return (vel, omega)


