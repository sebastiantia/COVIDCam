import threading
from datetime import datetime

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
recordControl = 0

CHECKERBOARD = (5, 7)


class recordVideo(threading.Thread):  # thread class to record video
    def run(self):
        now = datetime.now()
        time = datetime.time(now)
        name = "capture_V_" + now.strftime("%y%m%d") + time.strftime("%H%M%S") + ".avi"

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, fourcc, 30.0, (640, 480))

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if recordControl == 0:
                out.write(frame)
            elif recordControl == 2:
                break
        out.release()


def capture_image(img):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgpoints = []

    # Pre-processing to find corners
    lower = np.array([0, 0, 175])
    upper = np.array([179, 61, 252])
    hsv_value = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv_value, lower, upper)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilated = cv2.dilate(msk, krn, iterations=5)
    processed_img = 255 - cv2.bitwise_and(dilated, msk)

    cv2.imshow('capture', processed_img)

    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(processed_img, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(processed_img, corners, (11, 11), (-1, -1), criteria)

        imgpoints = corners2

    return imgpoints, ret


def show_video():
    global recordControl
    print('Press "c" to capture calibration image')
    while True:
        now = datetime.now()
        time = datetime.time(now)
        name = "capture_" + now.strftime("%y%m%d") + time.strftime("%H%M%S") + ".jpg"

        ret, frame = cap.read()
        if ret is True:
            frame = cv2.flip(frame, 1)  # 1 = vertical , 0 = horizontal

            cv2.imshow('frame', frame)

            k = cv2.waitKey(1) & 0Xff

            if k == ord('q'):  # Quit program and recording
                if recordControl != 2:  # make sure that out child process is complet
                    recordControl = 2
                    threadObj.join()
                break
            elif k == ord('c'):  # capture Image

                impt, ret = capture_image(frame)

                if ret:

                    # Draw and display the corners
                    frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, impt[0], ret)

                    print("Calibration successful! Close this window to continue")

                    cv2.destroyWindow('capture')
                    cv2.destroyWindow('frame')

                    cv2.imshow('img', frame)
                    cv2.waitKey(0)

                    return impt
                else:
                    print("Calibration failed. Try again")

        else:
            break


def calibrate(square_len, depth):

    impts = []

    if cap.isOpened():
        impts = show_video()
    else:
        cap.open()
        impts = show_video()

    cap.release()
    cv2.destroyAllWindows()

    X = CHECKERBOARD[0]
    Y = CHECKERBOARD[1]

    x_total = 0
    for j in range(Y):
        x_total += np.abs(impts[j*X][0][0] - impts[(j + 1)*X - 1][0][0])

    y_total = 0
    for i in range(X):
        y_total += np.abs(impts[i][0][1] - impts[i + Y - 1])[0][1]

    n = X * Y
    pix_x_ave = x_total / n
    pix_y_ave = y_total / n

    ave_img_sq_len = (pix_x_ave + pix_y_ave) / 2

    return depth * ave_img_sq_len / square_len
