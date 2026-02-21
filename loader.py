# loader.py
import sys
import time
import threading

class Loader:
    def __init__(self, message="Loading"):
        self.message = message
        self.running = False
        self.thread = None

    def _animate(self):
        while self.running:
            for dots in range(4):  # 0-3 dots
                if not self.running:
                    break
                sys.stdout.write(f"\r{self.message}" + "." * dots + "   ")
                sys.stdout.flush()
                time.sleep(0.4)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write(f"\r{self.message} completed!✅        \n")
        # sys.stdout.flush()
