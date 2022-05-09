# import opencv and numpy
import cv2
import numpy as np
import os

# trackbar callback fucntion to update HSV value
def callback(x):
    global H_low, H_high, S_low, S_high, V_low, V_high
    # assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos("low H", "controls")
    H_high = cv2.getTrackbarPos("high H", "controls")
    S_low = cv2.getTrackbarPos("low S", "controls")
    S_high = cv2.getTrackbarPos("high S", "controls")
    V_low = cv2.getTrackbarPos("low V", "controls")
    V_high = cv2.getTrackbarPos("high V", "controls")


# create a seperate window named 'controls' for trackbar
cv2.namedWindow("controls", 2)
cv2.resizeWindow("controls", 550, 10)


# global variables for trackbar
H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255

# create trackbars for high,low H,S,V
cv2.createTrackbar("low H", "controls", 0, 179, callback)
cv2.createTrackbar("high H", "controls", 179, 179, callback)

cv2.createTrackbar("low S", "controls", 0, 255, callback)
cv2.createTrackbar("high S", "controls", 255, 255, callback)

cv2.createTrackbar("low V", "controls", 0, 255, callback)
cv2.createTrackbar("high V", "controls", 255, 255, callback)


def main():
    images = []
    for filename in os.listdir("fotos/"):
        img = cv2.imread("fotos/" + filename)
        img = cv2.resize(img, (1920 // 5, 1080 // 5))
        images.append(img)

    # suponiendo que son 16 fotos
    vstack = []
    for i in range(4):
        vstack.append(
            np.vstack(
                [images[i * 4], images[i * 4 + 1], images[i * 4 + 2], images[i * 4 + 3]]
            )
        )

    collage = np.hstack([v for v in vstack])

    cv2.imshow("collage", collage)

    while 1:
        img = collage
        # convert sourece image to HSC color mode
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_low = np.array([H_low, S_low, V_low], np.uint8)
        hsv_high = np.array([H_high, S_high, V_high], np.uint8)

        # making mask for hsv range
        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        # masking HSV value selected color becomes black
        res = cv2.bitwise_and(img, img, mask=mask)

        # show image
        cv2.imshow("mask", mask)

        # waitfor the user to press escape and break the while loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # destroys all window
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
