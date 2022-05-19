import time

from modules.ArduinoSerialComm import ArduinoComm
from walk_to_water import main as desplazamiento
from detect import run
from detect_lata import run as detect_lata


def main():
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    time.sleep(2)
    print("Iniciando")
    try:
        while True:
            data = arduino.communicate(data="1")
            if data is not None:
                change, mode, u1, magnitud, angle, x_component = data
                print(change, mode)
            if change == 1:
                if mode == 0 or mode == 7:
                    continue
                if mode == 1:
                    print("Prueba de desplazamiendo en lado menor")
                    desplazamiento(hsv_min=(96, 40, 88), hsv_max=(112, 243, 255))
                    # desplazamiento(arduino, hsv_min=(96, 40, 88), hsv_max=(112, 243, 255))
                if mode == 2:
                    print("Prueba de evasion de mar")
                    run(
                        model="/home/pi/TMR2022_python/tf_models/limpiaplayas2022v3.tflite",
                        camera_id=0,
                        width=480,
                        height=360,
                        num_threads=4,
                        score_threshold=0.5,
                    )
                if mode == 3:
                    print("Prueba de evasion de objetos")
                    continue
                if mode == 4:
                    print("Prueba de localizacion de residuo")
                    run(
                        model="/home/pi/TMR2022_python/tf_models/limpiaplayas2022v3.tflite",
                        camera_id=0,
                        width=480,
                        height=360,
                        num_threads=4,
                        score_threshold=0.5,
                    )
                if mode == 5:
                    print("Prueba de recoleccion de residuo")
                    run(
                        model="/home/pi/TMR2022_python/tf_models/limpiaplayas2022v3.tflite",
                        camera_id=0,
                        width=480,
                        height=360,
                        num_threads=4,
                        score_threshold=0.5,
                    )
                if mode == 6:
                    print("Prueba de deposito de residuo")
                    detect_lata(
                        model="/home/pi/TMR2022_python/tf_models/limpiaplayas2022v3.tflite",
                        camera_id=0,
                        width=480,
                        height=360,
                        num_threads=4,
                        score_threshold=0.5,
                    )
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Ctrl+C pressed. Exiting...")
        arduino


if __name__ == "__main__":
    main()
