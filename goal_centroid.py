import cv2
import numpy as np


def get_goal_centroid(
    img: np.array, hsv_low: tuple[int, int, int], hsv_high: tuple[int, int, int]
) -> tuple[int, int]:
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_img, hsv_low, hsv_high)

    # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    c_x = int(M["m10"] / M["m00"])
    c_y = int(M["m01"] / M["m00"])

    return c_x, c_y


def main(hsv_low: tuple[int, int, int], hsv_high: tuple[int, int, int]) -> None:
    img = cv2.imread("fotos/opencv_frame_3.png")

    c_x, c_y = get_goal_centroid(img, hsv_low, hsv_high)

    # put text and highlight the center
    cv2.circle(img, (c_x, c_y), 5, (255, 255, 255), -1)
    cv2.putText(
        img,
        "centroid",
        (c_x - 25, c_y - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    # display the image
    cv2.imshow("Image", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main(
        hsv_low=(143, 128, 36),
        hsv_high=(179, 255, 126),
    )
