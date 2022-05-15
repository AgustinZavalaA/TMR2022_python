import cv2
import numpy as np


def get_area_from_box(img: np.array, threshold: int = 80) -> int:
    if img.shape[0] == 0 or img.shape[1] == 0:
        return 0
    # set color to image
    grey_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # binarize image to the threshold
    _, bin_image = cv2.threshold(grey_img, threshold, 255, cv2.THRESH_BINARY_INV)

    # return the area of the binarized image
    return np.count_nonzero(bin_image)


def main() -> None:
    img = cv2.imread("src/black_can_img.jpeg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(get_area_from_box(img))


if __name__ == "__main__":
    main()
