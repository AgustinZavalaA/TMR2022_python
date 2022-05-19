import cv2
import numpy as np

from modules.Motors import Motors
from modules.ArduinoSerialComm import ArduinoComm

import time


def check_if_there_is_water(
    img: np.array,
    hsv_min: tuple[int, int, int],
    hsv_max: tuple[int, int, int],
    threshold: int = 0.7,
) -> bool:
    water_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_water = cv2.inRange(water_roi, hsv_min, hsv_max)

    return np.count_nonzero(masked_water) > img.shape[0] * img.shape[1] * threshold


def main(
    hsv_min: tuple[int, int, int], hsv_max: tuple[int, int, int], visible: bool = False
):
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    time.sleep(2)
    motors = Motors()
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

            if check_if_there_is_water(frame[300:360, :], hsv_min, hsv_max):
                motors.stop()
                arduino.communicate(data="1500")
                print("Water detected")
            else:
                motors.move(True, velocitiy, True)
                motors.move(False, velocitiy, True)
                print("No water detected")

            if visible:
                cv2.imshow("frame", frame)
                cv2.imshow("masked_water", masked_water)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Exiting out")
                    break
    except KeyboardInterrupt:
        motors.stop()
        motors.disable()
        print("Ctrl+C pressed. Exiting...")


if __name__ == "__main__":
    main(
        hsv_min=(96, 40, 88),
        hsv_max=(112, 243, 255),
        visible=False,
    )
