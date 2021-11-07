import itertools
import os

import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

from calibration import calibrate
from detector_funcs import detect_people, detect_and_predict_mask
from params import Params
from logger import Logger


def dist(person_a: tuple, person_b: tuple, focal: float, calibration: float) -> float:
    wid_a, flat_coords_a = person_a
    wid_b, flat_coords_b = person_b

    y1a, y2a = flat_coords_a
    y1b, y2b = flat_coords_b

    distance = np.sqrt((y1a - y1b)**2 + (y2a - y2b)**2) / calibration

    return distance


def main(cal_const):
    p = Params()
    log = Logger()

    root_dir = os.path.abspath(" ")[:-2]

    ABS_FACE = os.path.sep.join([root_dir, p.FACE])
    ABS_MODEL = os.path.sep.join([root_dir, p.MODEL])
    ABS_MODEL_PATH = os.path.sep.join([root_dir, p.MODEL_PATH])

    prototxt_path = os.path.sep.join([ABS_FACE, "deploy.prototxt"])
    weights_path = os.path.sep.join([ABS_FACE,
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    mask_net = load_model(ABS_MODEL)

    labels_path = os.path.sep.join([ABS_MODEL_PATH, r"data\coco.names"])
    LABELS = open(labels_path).read().strip().split("\n")

    weights_path = os.path.sep.join([ABS_MODEL_PATH, "yolov3.weights"])
    config_path = os.path.sep.join([ABS_MODEL_PATH, r"cfg\yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    vs = cv2.VideoCapture(0)

    prev_no_mask_count = 0
    prev_dist_viol_count = 0
    while True:
        retval, frame = vs.read()
        if not retval:
            break

        frame = imutils.resize(frame, width=700)

        # Person detection
        results = detect_people(frame, net, ln,
                                person_idx=LABELS.index("person"))

        people = []
        boxes = []
        for person in results:
            con, box, center = person
            if con > p.MIN_CONF:
                boxes.append(box)
                startX, startY, endX, endY = box

                height = np.abs(startY - endY)
                people.append((height, center))

        # Mask
        (locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

        no_mask_count = 0
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            certainty = max(mask, withoutMask) * 100
            label = "{}: {:.2f}%".format(label, certainty)

            if mask <= withoutMask:
                no_mask_count += 1

            if prev_no_mask_count > no_mask_count:
                prev_no_mask_count = 0
                no_mask_count = 0

            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        if no_mask_count > prev_no_mask_count:
            for i in range(no_mask_count - prev_no_mask_count):
                log.log_violation("MASK")
            prev_no_mask_count = no_mask_count

        bound_color = (255, 0, 0)

        # Calculate pairwise distances
        dist_viol_count = 0
        for person_a, person_b in itertools.combinations_with_replacement(people, 2):
            if person_a != person_b:
                focal_len = 1
                _, center_a = person_a
                _, center_b = person_b

                xa, ya = center_a
                xb, yb = center_b

                mid_centers = (int((xa + xb) / 2), int((ya + yb) / 2))

                distance_ab = dist(person_a, person_b, focal_len, cal_const)
                cv2.line(frame, center_a, center_b, (255, 0, 0), 3)
                cv2.putText(frame, f"Dist.={distance_ab}", mid_centers, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                if distance_ab < p.DIST_THRESH:
                    bound_color = (140, 33, 255)
                    dist_viol_count += 1

        if prev_dist_viol_count > dist_viol_count:
            prev_dist_viol_count = 0
            dist_viol_count = 0

        print(dist_viol_count, prev_dist_viol_count)

        if dist_viol_count > prev_dist_viol_count:
            for i in range(dist_viol_count - prev_dist_viol_count):
                log.log_violation("DIST", distance_ab)
            prev_dist_viol_count = dist_viol_count

        for box in boxes:
            startX, startY, endX, endY = box

            cv2.rectangle(frame, (startX, startY), (endX, endY), bound_color, 2)
            cv2.circle(frame, center, 2, bound_color, 2)

        cv2.imshow('Output', frame)

        # Closes window on esc.
        c = cv2.waitKey(1)
        if c == 27:
            break

    vs.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    CALIBRATION_FILE = "saved_const.txt"

    with open(CALIBRATION_FILE, "r") as file:
        if not file.readline():
            flag = "y"
        else:
            flag = input("Do you want to re-calibrate [y] or use previous calibration constant? [n]")

    if flag == "y":
        checker_square_len = 0.032
        checker_depth = 4
        calibration_const = calibrate(checker_square_len, checker_depth)
        with open(CALIBRATION_FILE, "w") as file:
            file.write(str(calibration_const))
    else:
        with open(CALIBRATION_FILE, "r") as file:
            calibration_const = float(file.readline())

    main(calibration_const)
