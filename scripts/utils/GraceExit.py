import signal
import sys
import time


class GraceExit:
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.counter = 0
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        self.counter += 1
        print(f"\n[Ctrl+C] Pressed {self.counter}/{self.max_attempts} times")
        if self.counter >= self.max_attempts:
            print("Exiting now.")
            sys.exit(0)
