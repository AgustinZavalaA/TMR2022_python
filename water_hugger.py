import cv2
import numpy as np

from modules.Motors import Motors

import time


def water_hugger_areas_relation(
    img: np.array,
    hsv_min: tuple[int, int, int],
    hsv_max: tuple[int, int, int],
    cut_zone: int = 60,
) -> tuple[float, float]:
    water_roi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    masked_water = cv2.inRange(water_roi, hsv_min, hsv_max)

    return (
        np.count_nonzero(masked_water[:, :cut_zone]) / img.shape[0] * cut_zone,
        np.count_nonzero(masked_water[:, cut_zone:])
        / img.shape[0]
        * (img.shape[1] - cut_zone),
    )


def main(
    hsv_min: tuple[int, int, int], hsv_max: tuple[int, int, int], visible: bool = False
):
    motors = Motors()
    velocitiy = 70

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error al leer la camara")
                break

            frame = cv2.flip(frame, 1)

            water_left_side, water_right_side = water_hugger_areas_relation(
                frame[300:360, :], hsv_min, hsv_max, cut_zone=60
            )

            print(f"{water_left_side=} {water_right_side=}")

            if water_left_side < 0.7:
                print("Poca agua en izquierda, moviendose a ella")
                motors.move(True, velocitiy, False)
                motors.move(False, velocitiy, True)
                continue

            if water_right_side > 0.4:
                print("Mucha agua en derecha, moviendose a derecha")
                motors.move(True, velocitiy, True)
                motors.move(False, velocitiy, False)
                continue

            print("Abrazando awa")
            motors.move(True, velocitiy, True)
            motors.move(False, velocitiy, True)

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
    # main(
    #     hsv_min=(110, 38, 0),
    #     hsv_max=(131, 255, 255),
    #     visible=False,
    # )
    main(
        hsv_min=(96, 40, 88),
        hsv_max=(112, 243, 255),
        visible=False,
    )
