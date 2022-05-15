import cv2

import sys
import time
import argparse
from typing import NamedTuple

from camera_utils.object_detector import ObjectDetector
from camera_utils.object_detector import ObjectDetectorOptions
from camera_utils import utils
from modules.Motors import Motors
from image_area import get_area_from_box


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
    # Start the motors and variables for motor control
    motors = Motors()
    stopped_count = 0
    STOPPED_LIMIT = 5
    MAX_AREA_LIMIT = 10_000
    last_vel = 0

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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

            image = cv2.flip(image, 1)

            # Run object detection estimation using the model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb_image)
            # get the most important values (label, score, centroid, area) from the detections
            my_detections = []
            for det in detections:
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
            # sort the detections by score
            my_detections = sorted(my_detections, key=lambda x: x.score)
            # print(my_detections, end="\n\n")

            if not my_detections:
                print("No object detected\n\n")
                motors.stop()
                continue
            # If there are any detections, get the most important one (black can)
            # select the black can with the highest score
            selected_can = my_detections.pop(0)
            while my_detections and selected_can.label != "black_can":
                selected_can = my_detections.pop(0)

            # if the selected can is not the black can, then continue the loop
            if selected_can.label != "black_can":
                continue

            # calculate the distance from the centroid to the center of the image
            distance_from_center = selected_can.centroid[0] - image.shape[1] // 2
            print(f"{distance_from_center=}", end=" ")

            # calculate the velocity using the scaled distance from 20 to 50 percent of the motors power
            vel = map_range(abs(distance_from_center), 0, image.shape[1] // 2, 30, 40)
            # apply some smoothing to the velocity
            vel = int(vel * 0.2 + last_vel * 0.8)
            last_vel = vel
            print(f"velocity={vel}")

            # si el objeto esta en la mitad de la imagen (dentro del 20%), no hace nada
            if abs(distance_from_center) < image.shape[1] // 2 * 0.2:
                print(f"stopped {stopped_count}")
                print(f"area={selected_can.area}")
                stopped_count += 1
                # si el robot se detiene por menos de 5 frames, entonces se detiene
                if stopped_count < STOPPED_LIMIT:
                    motors.stop()
                else:
                    # si el robot se detiene por mas de 5 frames, entonces se acerca al objeto
                    # calcula la velocidad para acercarse al objeto
                    vel = 60 - map_range(selected_can.area, 0, 15_000, 0, 30)
                    vel = int(vel * 0.2 + last_vel * 0.8)
                    last_vel = vel
                    # si el area del objeto es mayor que el limite, entonces se detiene
                    if selected_can.area > MAX_AREA_LIMIT:
                        print("Can is too close")
                        motors.stop()
                        # TODO aplicar el script de recoger lata
                    else:
                        # si el area del objeto es menor que el limite, entonces se mueve con la velocidad calculada
                        print(f"vel forward {vel}")
                        motors.move(True, vel, True)
                        motors.move(False, vel, True)

            # si el objeto esta a la derecha, se mueve a la izquierda
            elif distance_from_center < 0:
                stopped_count = 0
                print("izquierda")
                motors.move(True, vel, False)
                motors.move(False, vel, True)
            # si el objeto esta a la izquierda, se mueve a la derecha
            elif distance_from_center > 0:
                stopped_count = 0
                print("derecha")
                motors.move(True, vel, True)
                motors.move(False, vel, False)

            print("\n\n")

            # Draw keypoints and edges on input image
            # image = utils.visualize(image, detections)

            # Stop the program if the ESC key is pressed.
            # if cv2.waitKey(1) == 27:
            #     break
            # cv2.imshow("object_detector", image)

    except KeyboardInterrupt:
        # Stop the motors when the user presses ctrl-c
        print("Program interrupted by user.")
        motors.stop()
        motors.disable()
        cap.release()
        cv2.destroyAllWindows()


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
        default=0.6,
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
