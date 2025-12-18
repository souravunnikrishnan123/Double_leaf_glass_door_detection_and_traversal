#!/usr/bin/env python3
import rospy
import time
from collections import defaultdict, deque
import os


class get_duration_seconds():
    def __init__(self, file_path="/app/catkin_ws/src/robodog_glass_door_detection/scripts/durations.txt"):
        self.file_path = file_path
        self.start_times = {}
        self.durations = defaultdict(lambda: deque(maxlen=5))
    
    def start(self, label=""):
        self.start_times[label] = time.time()

    def stop(self, label=""):
        if label not in self.start_times:
            raise ValueError(f"No start time recorded for label '{label}'")
        duration = time.time() - self.start_times[label]
        self.durations[label].append(duration)
        del self.start_times[label]
        self.write_text_file()

    def get_moving_average(self, label):
        values = self.durations[label]
        if len(values) == 0:
            return None
        return sum(values) / len(values)
    
    def get_all_stats(self):
        stats = {}
        for label, values in self.durations.items():
            stats[label] = {
                "last_values": list(values),
                "moving_average": sum(values) / len(values)
            }
        return stats
    
    def write_text_file(self):
        if os.path.exists(self.file_path):
            with open(self.file_path,"r") as f:
                existing_lines = f.readlines()
                existing_labels = [line.split(":")[0] for line in existing_lines]
        else:
            existing_lines = []
            existing_labels = []
        
        for label, values in self.durations.items():
                if len(values) > 0:
                    moving_avg = sum(values) / len(values)
                    if label in existing_labels:
                        index_of_existing_labels = existing_labels.index(label)
                        existing_lines[index_of_existing_labels] = f"{label}: {moving_avg:.6f}\n"
                    else:
                        existing_lines.append(f"{label}: {moving_avg:.6f}\n")
        with open(self.file_path, "w") as f:
            f.writelines(existing_lines)