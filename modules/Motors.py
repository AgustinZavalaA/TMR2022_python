import RPi.GPIO as GPIO
import time


class Motors:
    def __init__(self) -> None:
        GPIO.setmode(GPIO.BOARD)

        # Motor A
        self.PWMAIN = 12
        self.AIN1 = 13
        self.AIN2 = 15
        # Motor B
        self.PWMBIN = 35
        self.BIN1 = 16
        self.BIN2 = 18

        GPIO.setup(self.AIN1, GPIO.OUT)
        GPIO.setup(self.AIN2, GPIO.OUT)
        GPIO.setup(self.BIN1, GPIO.OUT)
        GPIO.setup(self.BIN2, GPIO.OUT)

        # Set GPIO pin 12 to output mode.
        GPIO.setup(self.PWMAIN, GPIO.OUT)
        # Initialize PWM on pwmPin 100Hz frequency
        self.pwm_a = GPIO.PWM(self.PWMAIN, 100)

        GPIO.setup(self.PWMBIN, GPIO.OUT)
        # Initialize PWM on pwmPin 100Hz frequency
        self.pwm_b = GPIO.PWM(self.PWMBIN, 100)

        dc = 0  # set dc variable to 0 for 0%
        self.pwm_a.start(dc)  # Start PWM with 0% duty cycle
        self.pwm_b.start(dc)

    def stop(self) -> None:
        # GPIO.output(STBY, False)
        dc = 1
        self.pwm_a.ChangeDutyCycle(dc)
        self.pwm_b.ChangeDutyCycle(dc)

        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    def move(self, motor: bool, speed: int, direction: bool) -> None:
        in_pin1 = GPIO.LOW
        in_pin2 = GPIO.HIGH

        if direction:
            in_pin1 = GPIO.HIGH
            in_pin2 = GPIO.LOW

        if motor:
            GPIO.output(self.AIN1, in_pin1)
            GPIO.output(self.AIN2, in_pin2)
            self.pwm_a.ChangeDutyCycle(speed)
        else:
            GPIO.output(self.BIN1, in_pin1)
            GPIO.output(self.BIN2, in_pin2)
            self.pwm_b.ChangeDutyCycle(speed)

    def disable(self) -> None:
        self.stop()
        self.pwm_a.stop()  # stop PWM
        self.pwm_b.stop()  # stop PWM
        GPIO.cleanup()  # resets GPIO ports used back to input mode


def main() -> None:
    motors = Motors()
    try:
        while True:
            motors.move(True, 100, True)
            motors.move(False, 100, True)
            time.sleep(1.2)

            motors.stop()
            time.sleep(1)

            motors.move(True, 100, False)
            motors.move(False, 100, False)
            time.sleep(1.2)

            motors.stop()
            time.sleep(1)
    except KeyboardInterrupt:
        motors.disable()


if __name__ == "__main__":
    main()
