from typing import List, Tuple
import serial
import time


class ArduinoComm:
    def __init__(
        self, port: str = "/dev/ttyACM0", baudrate: int = 115200, timeout: float = 0.1
    ) -> None:
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.ser.flush()

    def communicate(self, data: str = "1") -> Tuple[int, int, List[str]]:
        self.ser.write(data.encode("ascii"))
        if data != "1":
            return 0, 0, []

        line = self.ser.readline().decode("ascii").rstrip()
        line_list = line.split(",")
        r = None
        try:
            # r = int(line_list[0]), int(line_list[1]), line_list[2:]
            r = (
                int(line_list[0]),  # btn_change
                int(line_list[1]),  # btn_mode
                int(line_list[2]),  # u1
                int(line_list[3]),  # magnitud
                int(line_list[4]),  # angle
                int(line_list[5]),  # x_component
            )
        except ValueError:
            r = None
        return r

    def close(self) -> None:
        self.ser.close()


def main() -> None:
    arduino = ArduinoComm()
    # Cantidad minima de sleep seguido de hacer una instancia
    time.sleep(2)
    print(arduino.communicate())
    print(arduino.communicate("2"))
    time.sleep(3)
    print(arduino.communicate("2"))
    time.sleep(2)

    print(arduino.communicate("3"))
    time.sleep(3)
    print(arduino.communicate("3"))
    time.sleep(2)

    # si se necesita hacer un communicate("2") se necesita implementar un sleep seguido de este
    try:
        while True:
            print(arduino.communicate())
            time.sleep(0.1)
    except KeyboardInterrupt:
        arduino.close()


if __name__ == "__main__":
    main()

    # mlp
