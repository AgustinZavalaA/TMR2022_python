import cv2
import numpy as np

from utils.Motors import Motors
from utils.ArduinoSerialComm import ArduinoComm

import time


def main(hsv_min: tuple[int, int, int], hsv_max: tuple[int, int, int]):
    # motors = Motors()
    velocitiy = 50

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error al leer la camara")
                break

            water_roi = frame[300:360, :]
            water_roi = cv2.cvtColor(water_roi, cv2.COLOR_BGR2HSV)
            masked_water = cv2.inRange(water_roi, hsv_min, hsv_max)

            # print if water is detected, majority of the pixels are 1

            threshold = 0.8
            print(
                np.sum(masked_water),
                masked_water.shape[0] * masked_water.shape[1] * 255 * threshold,
                end="   ",
            )
            if (
                np.sum(masked_water)
                > masked_water.shape[0] * masked_water.shape[1] * 255 * threshold
            ):
                # motors.stop()
                print("Water detected")
            else:
                # motors.move(True, velocitiy, True)
                # motors.move(False, velocitiy, True)
                print("No water detected")

            cv2.imshow("frame", frame)
            cv2.imshow("masked_water", masked_water)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting out")
                break
    except KeyboardInterrupt:
        # motors.stop()
        # motors.disable()
        print("Ctrl+C pressed. Exiting...")


if __name__ == "__main__":
    main(
        hsv_min=(85, 0, 38),
        hsv_max=(152, 133, 125),
    )
