import sys
import time
import argparse

import cv2
from camera_utils.object_detector import ObjectDetector
from camera_utils.object_detector import ObjectDetectorOptions
from camera_utils import utils

from typing import NamedTuple


class my_detection(NamedTuple):
    label: str
    score: float
    centroid: tuple[int, int]
    area: int


def run(
    model: str,
    camera_id: int,
    width: int,
    height: int,
    num_threads: int,
    score_threshold: float,
) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
      num_threads: The number of CPU threads to run the model.
    """
    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=num_threads,
        score_threshold=score_threshold,
        max_results=3,
    )
    detector = ObjectDetector(model_path=model, options=options)

    try:
        # Continuously capture images from the camera and run inference
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    "ERROR: Unable to read from webcam. Please verify your webcam settings."
                )

            image = cv2.flip(image, 1)

            # Run object detection estimation using the model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb_image)
            my_detections = []
            for det in detections:
                w = det.bounding_box.right - det.bounding_box.left
                h = det.bounding_box.bottom - det.bounding_box.top
                my_detections.append(
                    my_detection(
                        det.categories[0].label,
                        det.categories[0].score,
                        (det.bounding_box.left + w // 2, det.bounding_box.top + h // 2),
                        w * h,
                    )
                )
            my_detections = sorted(my_detections, key=lambda x: x.score)
            print(my_detections, end="\n\n")

            if len(my_detections) > 0:
                selected_can = my_detections.pop(0)
                while my_detections and selected_can.label != "black_can":
                    selected_can = my_detections.pop(0)

                if selected_can.label != "black_can":
                    continue

                distance_from_center = selected_can.centroid[0] - image.shape[1] // 2
                print(distance_from_center)

            # Draw keypoints and edges on input image
            # image = utils.visualize(image, detections)

            # Stop the program if the ESC key is pressed.
            # if cv2.waitKey(1) == 27:
            #     break
            # cv2.imshow("object_detector", image)

    except KeyboardInterrupt:
        print("Program interrupted by user.")
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
