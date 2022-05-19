import time

from ArduinoSerialComm import ArduinoComm
from p1_desplazamiento import main as desplazamiento


def main():
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    time.sleep(2)
    while True:
        change, mode, _ = arduino.communicate(data="1")
        if change == 1:
            if mode == 0:
                continue
            if mode == 1:
                print("Prueba de desplazamiendo en lado menor")
                desplazamiento(arduino, hsv_min=(96, 40, 88), hsv_max=(112, 243, 255))
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


if __name__ == "__main__":
    main()
