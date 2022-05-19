import time
import cv2
import numpy as np

from ArduinoSerialComm import ArduinoComm
from Motors import Motors

from p1_desplazamiento import check_if_there_is_water


def main():
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    time.sleep(2)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    motors = Motors()
    print("Iniciando")
    try:
        while True:
            data = arduino.communicate(data="1")
            if data is not None:
                change, mode, u1, magnitud, angle, x_component = data
                print(change, mode)
            if change == 1:
                if mode == 0:
                    continue
                if mode == 1:
                    print("Prueba de desplazamiendo en lado menor")
                    # desplazamiento(arduino, hsv_min=(96, 40, 88), hsv_max=(112, 243, 255))
                    hsv_min = (110, 38, 0)
                    hsv_max = (131, 255, 255)
                    velocity = 50
                    while cap.isOpened():
                        data = arduino.communicate(data="1")
                        print(data)
                        if data is not None and data[0] == 0:
                            raise KeyboardInterrupt

                        ret, frame = cap.read()
                        if not ret:
                            print("Error al leer la camara")
                            break

                        if check_if_there_is_water(frame[300:360, :], hsv_min, hsv_max):
                            motors.stop()
                            print("Water detected")
                        else:
                            motors.move(True, velocity, True)
                            motors.move(False, velocity, True)
                            print("No water detected")
                        time.sleep(0.1)
                if mode == 2:
                    print("Prueba de evasion de mar")
                if mode == 3:
                    print("Prueba de evasion de objetos")
                if mode == 4:
                    print("Prueba de localizacion de residuo")
                if mode == 5:
                    print("Prueba de recoleccion de residuo")
                if mode == 6:
                    print("Prueba de deposito de residuo")
                if mode == 7:
                    print("k pendejo papi")
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Exiting...")
        arduino.close()
        motors.stop()
        motors.disable()


if __name__ == "__main__":
    main()
