import cv2 as cv
import numpy as np
import math


class ArucoDetector:
    def __init__(
        self,
        camera_matrix: np.array,
        distortion_coefs: np.array,
    ):
        self.dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
        self.parameters = cv.aruco.DetectorParameters()
        self.parameters.minCornerDistanceRate = 0.01
        self.parameters.minDistanceToBorder = 1
        self.parameters.minMarkerPerimeterRate = 0.01

        self.detector = cv.aruco.ArucoDetector(self.dictionary, self.parameters)

        self.camera_matrix = camera_matrix
        self.distortion_coefs = distortion_coefs

    def detectMarkers(self, image):
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(image)
        return corners, ids, rejectedImgPoints

    def estimatePose(self, corners): 
        rvecs, tvecs, _ = self.estimatePoseSingleMarkers(
            corners, 0.05, self.camera_matrix, self.distortion_coefs
        )
        return rvecs, tvecs

    def estimateAngle(self, image):
        corners, ids, rejectedImgPoints = self.detectMarkers(image)
        if ids is not None:
            rvecs, tvecs, trash = self.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.distortion_coefs
            )
            return rvecs[0][2]
        else:
            return None


    def estimateDistance(self, image):
        corners, ids, rejectedImgPoints = self.detectMarkers(image)
        if ids is not None:
            _, tvecs, _ = self.estimatePoseSingleMarkers(
                corners, 0.05, self.camera_matrix, self.distortion_coefs
            )
            return tvecs[0][2]
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
