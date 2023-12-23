import cv2 as cv

dictionary = cv.aruco.getPredefinedDictionary(
    cv.aruco.DICT_4X4_250
)  # isto passaria para outro sítio
parameters = cv.aruco.DetectorParameters()  # isto também
parameters.polygonalApproxAccuracyRate = 0.03
detector = cv.aruco.ArucoDetector(dictionary, parameters)  # isto também

img = cv.imread("screen.png")

corners, ids, rejectedImgPoints = detector.detectMarkers(img)
print(corners)

for rejected in rejectedImgPoints:
    coordinates = rejected[0]
    p1 = coordinates[0]
    p2 = coordinates[2]
    img = cv.rectangle(
        img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2
    )

im_out = cv.aruco.drawDetectedMarkers(img, corners, ids)

cv.imwrite("screen2.png", img)
