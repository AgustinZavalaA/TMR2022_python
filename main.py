from modules.Motors import Motors
from modules.ArduinoSerialComm import ArduinoComm

import time


def main2() -> None:
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    motors = Motors()
    time.sleep(2)
    print("Ready to use")

    pick_up_can(arduino, motors)

    arduino.close()
    motors.stop()
    motors.disable()


def main():
    arduino = ArduinoComm(port="/dev/ttyACM0", baudrate=115200, timeout=0.1)
    motors = Motors()
    time.sleep(2)
    print("Ready to use")

    # modo obtenido del arduino, puede ser de 0 a 7
    mode = arduino.communicate(data="1")[1]
    # variable para mantener el estado del modo
    action_done = False
    last_mode = mode

    while True:
        change, mode, ultrasonic_data = arduino.communicate(data="1")

        if mode == 7:
            print("Exiting out")
            arduino.close()
            motors.stop()
            motors.disable()
            break

        if mode != last_mode:
            last_mode = mode
            action_done = False

        if mode == 0:
            continue

        if mode == 1:
            if not action_done:
                move_arm(arduino)
                action_done = True

        if mode == 2:
            if not action_done:
                move_tray(arduino)
                action_done = True

        if mode == 3:
            if not action_done:
                move_claw(arduino)
                action_done = True

        if mode == 4:
            if not action_done:
                pick_up_can(arduino, motors)
                action_done = True


def move_arm(arduino: ArduinoComm) -> None:
    arduino.communicate(data="2")
    time.sleep(1.5)
    pass


def move_tray(arduino: ArduinoComm) -> None:
    arduino.communicate(data="3")
    time.sleep(1.5)
    pass


def move_claw(arduino: ArduinoComm) -> None:
    arduino.communicate(data="4")
    time.sleep(1.5)
    pass


def pick_up_can(arduino: ArduinoComm, motors: Motors) -> None:
    # abre la garra
    # move_claw(arduino)
    # baja el brazo
    move_arm(arduino)
    # se acerca a la lata
    motors.move(True, 100, True)
    motors.move(False, 100, True)
    time.sleep(0.9)
    motors.stop()
    # cierra la garra
    move_claw(arduino)
    # sube el brazo
    move_arm(arduino)
    move_claw(arduino)


if __name__ == "__main__":
    # main()
    main2()
