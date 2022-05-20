import cv2
import numpy as np

import sys
import time
import argparse
from typing import NamedTuple
import time

from camera_utils.object_detector import ObjectDetector
from camera_utils.object_detector import ObjectDetectorOptions
from camera_utils import utils
from modules.Motors import Motors
from image_area import get_area_from_box
from modules.ArduinoSerialComm import ArduinoComm
from walk_to_water import check_if_there_is_water
from main import pick_up_can, move_tray
from goal_centroid import get_goal_centroid
from water_hugger import water_hugger_areas_relation


class my_detection(NamedTuple):
    """Class for storing all the metrics of the detection."""

    label: str
    score: float
    centroid: tuple[int, int]
    area: int


def map_range(x: int, in_min: int, in_max: int, out_min: int, out_max: int) -> int:
    """Convert a value from one range to another range."""
    return int((x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min)


def run(
    model: str,
    camera_id: int,
    width: int,
    height: int,
    num_threads: int,
    score_threshold: float,
) -> None:
    # water hsv
    water_hsv = ((100, 46, 171), (115, 248, 255))
    # variables for the program
    stopped_count = 0
    STOPPED_LIMIT = 6
    grab_can_count = 0
    GRAB_CAN_LIMIT = 6
    LOST_ROBOT_LIMIT = 10  # 70 #TODO change this
    lost_robot_count = 0
    MAX_AREA_LIMIT = 5_000
    number_of_cans_recolected = 99  # TODO: change to 0
    last_vel = 0
    found_something_of_interest = True
    STUCK_LIMIT = 25
    stuck_count = 0
    LOST_ROBOT_ADVANCE_LIMIT = 15
    lost_robot_advance_count = 0
    label_to_find = "black_can"
    # variables para water hugger
    water_hugger_get_to_water_action = False
    water_hugger_hugger_action = False

    # Start the motors and variables for motor control and arduino communication
    motors = Motors()
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    time.sleep(2)
    print("Ready to use")

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 2)

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=num_threads,
        score_threshold=score_threshold,
        max_results=3,
    )
    detector = ObjectDetector(model_path=model, options=options)

    # Execute the main code until the user presses ctrl-c
    try:
        # Continuously capture images from the camera and run inference
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit("ERROR: Unable to read from webcam.")

            # cv2.imshow("image1", image)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     raise KeyboardInterrupt

            arduino_data = arduino.communicate(data="1")
            if arduino_data is not None:
                (
                    btn_change,
                    btn_mode,
                    front_ultrasonic,
                    magnitud,
                    angle,
                    x_component,
                ) = arduino_data
            print(f"{front_ultrasonic=}")

            my_detections = process_detections(image, detector)
            # print(my_detections, end="\n\n")

            if water_hugger_get_to_water_action == True:
                motors.move(True, 60, True)
                motors.move(False, 60, True)
                goal_centroid = get_goal_centroid(
                    image,
                    hsv_low=(0, 148, 40),
                    hsv_high=(179, 255, 121),
                    area_threshold=300,
                )
                if goal_centroid is not None:
                    water_hugger_get_to_water_action = False
                    water_hugger_hugger_action = True
                    continue

                if check_if_there_is_water(
                    image[300:360, :],
                    hsv_min=water_hsv[0],
                    hsv_max=water_hsv[1],
                    threshold=0.5,
                ):
                    water_hugger_get_to_water_action = False
                    water_hugger_hugger_action = True
                    print("water_hugger_hugger_action")
                    motors.move(True, 70, False)
                    motors.move(False, 70, False)
                    time.sleep(0.5)

            if water_hugger_hugger_action == True:
                w_image = cv2.flip(image[300:360, :], 1)
                water_left_side, water_right_side = water_hugger_areas_relation(
                    w_image,
                    hsv_min=water_hsv[0],
                    hsv_max=water_hsv[1],
                    cut_zone=80,
                )
                print(f"{water_left_side=} {water_right_side=}")
                velocity = 70

                if water_left_side < 0.7:
                    print("Poca agua en izquierda, moviendose a ella")
                    motors.move(True, 0, False)
                    motors.move(False, velocity, True)
                    continue

                if water_right_side > 10:
                    print("Mucha agua en derecha, moviendose a derecha")
                    motors.move(True, velocity, True)
                    motors.move(False, 0, False)
                    continue

                print("Abrazando awa")
                motors.move(True, velocity, True)
                motors.move(False, velocity, True)

            if lost_robot_count > LOST_ROBOT_LIMIT:
                if label_to_find == "goal":
                    water_hugger_get_to_water_action = True
                    continue
                print("Lost robot")
                found_something_of_interest = True
                if lost_robot_advance_count > LOST_ROBOT_ADVANCE_LIMIT or (
                    label_to_find in [d.label for d in my_detections]
                ):
                    lost_robot_advance_count = 0
                    lost_robot_count = 0
                    found_something_of_interest = True
                    continue

                motors.move(True, 70, True)
                motors.move(False, 70, True)
                lost_robot_advance_count += 1

                if front_ultrasonic < 45 or check_if_there_is_water(
                    image[300:360, :],
                    hsv_min=water_hsv[0],
                    hsv_max=water_hsv[1],
                    threshold=0.5,
                ):
                    print("Casi choco, me voy pa atras")
                    motors.move(True, 50, False)
                    motors.move(False, 50, False)
                    time.sleep(1)

                    motors.move(True, 50, False)
                    motors.move(False, 50, True)
                    time.sleep(0.3)

            if not found_something_of_interest:
                print("Moviendose a la izquierda")
                print(f"{lost_robot_count=}")
                motors.move(True, 45, False)
                motors.move(False, 45, True)
                lost_robot_count += 1

            if check_if_there_is_water(
                image[300:360, :],
                hsv_min=water_hsv[0],
                hsv_max=water_hsv[1],
                threshold=0.5,
            ):
                print("There is water\n\n")
                arduino.communicate(data="7")
                motors.move(True, 70, False)
                motors.move(False, 70, False)
                found_something_of_interest = False
                continue

            # buscamos primero la zona de deposito si el numero de canes recolectados es mayor que 3

            if not my_detections and number_of_cans_recolected < 3:
                print("No object detected\n\n")
                found_something_of_interest = False
                continue
            found_something_of_interest = True

            if number_of_cans_recolected >= 3:
                label_to_find = "goal"
            else:
                label_to_find = "black_can"

            # If there are any detections, get the most important one (black can)
            # select the black can with the highest score

            # water hugger state action get to water
            water_hugger_get_to_water_action = False

            if number_of_cans_recolected >= 3:
                goal_centroid = get_goal_centroid(
                    image[150:300, :],
                    # hsv_low=(0, 100, 55),
                    # hsv_high=(6, 255, 210),
                    hsv_low=(0, 148, 40),
                    hsv_high=(179, 255, 121),
                    area_threshold=300,
                )
                if goal_centroid:
                    print("Found the goal\n\n")

                    distance_from_center = image.shape[1] // 2 - goal_centroid[0]
                    print(f"{distance_from_center=}")
                    water_hugger_hugger_action = False
                else:
                    found_something_of_interest = False
                    continue
            else:
                if my_detections:
                    selected_can = my_detections.pop(0)
                else:
                    found_something_of_interest = False
                    continue
                while my_detections and (
                    selected_can.label.find(label_to_find) == -1
                    or selected_can.area > 30_000
                ):
                    selected_can = my_detections.pop(0)

                # if the selected can is not the black can, then continue the loop
                if selected_can.label.find(label_to_find) == -1:
                    found_something_of_interest = False
                    continue

                # calculate the distance from the centroid to the center of the image
                distance_from_center = selected_can.centroid[0] - image.shape[1] // 2
                print(f"{distance_from_center=}", end=" ")

            lost_robot_count = 0

            # calculate the velocity using the scaled distance from 20 to 50 percent of the motors power
            # vel = map_range(abs(distance_from_center), 0, image.shape[1] // 2, 25, 40)
            vel = map_range(abs(distance_from_center), 0, image.shape[1] // 2, 35, 60)
            # apply some smoothing to the velocity
            vel = int(vel * 0.2 + last_vel * 0.8)

            if vel == last_vel:
                print(f"{stuck_count}")
                stuck_count += 1
                if stuck_count > STUCK_LIMIT:
                    print("Stuck")
                    v = 70
                    motors.move(True, v, False)
                    motors.move(False, v, False)
                    time.sleep(0.5)
                    motors.move(True, v, True)
                    motors.move(False, v, True)
                    time.sleep(0.5)
                    motors.move(True, v, False)
                    motors.move(False, v, False)
                    time.sleep(0.5)
                    stuck_count = 0
                    continue

            last_vel = vel
            print(f"velocity={vel}")

            # si el objeto esta en la mitad de la imagen (dentro del 20%), no hace nada
            if abs(distance_from_center) < image.shape[1] // 2 * 0.25:
                print(f"stopped {stopped_count}")
                # print(f"area={selected_can.area}")
                stopped_count += 1
                # si el robot se detiene por menos de 5 frames, entonces se detiene
                if stopped_count < STOPPED_LIMIT:
                    motors.stop()
                else:
                    if label_to_find == "goal":
                        print("buscando goal nose que hacer")

                        if front_ultrasonic > 18:
                            motors.move(True, 40, True)
                            motors.move(False, 40, True)

                        elif front_ultrasonic > 6:
                            motors.move(True, 20, True)
                            motors.move(False, 20, True)
                        else:
                            print("goal encontrado")
                            motors.stop()
                            move_tray(arduino)
                            time.sleep(0.5)
                            move_tray(arduino)
                            time.sleep(0.5)
                            move_tray(arduino)
                            time.sleep(0.5)
                            move_tray(arduino)

                            motors.move(True, 100, False)
                            motors.move(False, 100, False)

                            time.sleep(3)

                            motors.move(True, 100, True)
                            motors.move(False, 100, False)

                            time.sleep(1)

                            number_of_cans_recolected = 0

                        continue
                    # si el robot se detiene por mas de 5 frames, entonces se acerca al objeto
                    # calcula la velocidad para acercarse al objeto
                    vel = 48 - map_range(selected_can.area, 0, 15_000, 0, 35)
                    vel = int(vel * 0.2 + last_vel * 0.8)
                    vel = 0 if vel < 0 else vel
                    vel = 100 if vel > 100 else vel
                    last_vel = vel
                    # si el area del objeto es mayor que el limite, entonces se detiene
                    print(f"Probabilidad de que sea un can: {selected_can.score}")

                    if selected_can.area > MAX_AREA_LIMIT:
                        print("Can is too close")
                        # si esta muy cerca, entonces retrocede
                        if (
                            front_ultrasonic < 28 and front_ultrasonic < 50
                        ) or selected_can.area > 20_000:
                            vel = 35
                            motors.move(True, vel, False)
                            motors.move(False, vel, False)
                            grab_can_count = 0
                        else:
                            motors.stop()
                            arduino.communicate(data="6")
                            grab_can_count += 1
                            if grab_can_count > GRAB_CAN_LIMIT:
                                grab_can_count = 0
                                # image = cv2.circle(
                                #     image, selected_can.centroid, 5, (0, 0, 255), -1
                                # )
                                # cv2.imshow("image", image)
                                # print(image)
                                # if cv2.waitKey(1) & 0xFF == ord("q"):
                                #     raise KeyboardInterrupt
                                # if input("Do you want to grab the can? (y/n)") == "y":
                                time.sleep(1)
                                pick_up_can(arduino, motors)
                                number_of_cans_recolected += 1

                    else:
                        # si el area del objeto es menor que el limite, entonces se mueve con la velocidad calculada
                        print(f"vel forward {vel}")
                        motors.move(True, vel, True)
                        motors.move(False, vel, True)
                        grab_can_count = 0

            # si el objeto esta a la derecha, se mueve a la izquierda
            elif distance_from_center < 0:
                stopped_count = 0
                print("izquierda")
                # motors.move(True, vel, False)
                motors.move(True, 0, False)
                motors.move(False, vel, True)
            # si el objeto esta a la izquierda, se mueve a la derecha
            elif distance_from_center > 0:
                stopped_count = 0
                print("derecha")
                motors.move(True, vel, True)
                # motors.move(False, vel, False)
                motors.move(False, 0, False)

            print("\n\n")

    except KeyboardInterrupt:
        # Stop the motors when the user presses ctrl-c
        print("Program interrupted by user.")
        arduino.close()
        motors.stop()
        motors.disable()
        cap.release()
        cv2.destroyAllWindows()


def process_detections(image, detector):
    image = cv2.flip(image, 1)

    # Run object detection estimation using the model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect(rgb_image)
    # get the most important values (label, score, centroid, area) from the detections
    my_detections = []
    for i, det in enumerate(detections):
        l = det.bounding_box.left
        r = det.bounding_box.right
        t = det.bounding_box.top
        b = det.bounding_box.bottom
        w = r - l
        h = b - t
        my_detections.append(
            my_detection(
                det.categories[0].label,
                det.categories[0].score,
                (l + w // 2, t + h // 2),
                get_area_from_box(rgb_image[t:b, l:r]),
            )
        )
    print(my_detections)
    # sort the detections by score
    # return sorted(my_detections, key=lambda x: x.score)
    # return my_detections
    return sorted(my_detections, key=lambda x: x.area * x.score, reverse=True)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="tf_models/limpiaplayas2022v3.tflite",
    )
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=480,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=360,
    )
    parser.add_argument(
        "--numThreads",
        help="Number of CPU threads to run the model.",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "--scoreThreshold",
        help="threshold for object detection",
        required=False,
        type=float,
        default=0.50,
    )
    args = parser.parse_args()

    run(
        args.model,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
        int(args.numThreads),
        float(args.scoreThreshold),
    )


if __name__ == "__main__":
    main()
