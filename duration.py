import time

class get_duration_seconds():
    def __init__(self):
        self.start_time = time.time()

    def get_duration(self, label=""):
        duration = time.time() - self.start_time
        if label:
            print(f"{label}: {duration} seconds")
