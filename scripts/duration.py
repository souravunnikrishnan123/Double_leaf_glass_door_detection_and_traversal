#!/usr/bin/env python3
import rospy
import time
from collections import defaultdict, deque
import os


class get_duration_seconds():
    start_times = {}
    durations = defaultdict(lambda: deque(maxlen=5))
    file_path = None # IMPORTANT: defer initialization until after rospy is initialized
    
    def __init__(self):
        get_duration_seconds.file_path = rospy.get_param("~durations_file_path")
        
    
    def start(self, label=""):
        get_duration_seconds.start_times[label] = time.time()

    def stop(self, label=""):
        if label not in get_duration_seconds.start_times:
            raise ValueError(f"No start time recorded for label '{label}'")
        duration = time.time() - get_duration_seconds.start_times[label]
        get_duration_seconds.durations[label].append(duration)
        del get_duration_seconds.start_times[label]

    @classmethod
    def write_text_file(cls):
        if os.path.exists(get_duration_seconds.file_path):
            with open(get_duration_seconds.file_path,"r") as f:
                existing_lines = f.readlines()
                existing_labels = [line.split(":")[0] for line in existing_lines]
        else:
            existing_lines = []
            existing_labels = []
        
        for label, values in get_duration_seconds.durations.items():
                if len(values) > 0:
                    moving_avg = sum(values) / len(values)
                    if label in existing_labels:
                        index_of_existing_labels = existing_labels.index(label)
                        existing_lines[index_of_existing_labels] = f"{label}: {moving_avg:.6f}\n"
                    else:
                        existing_lines.append(f"{label}: {moving_avg:.6f}\n")
        with open(get_duration_seconds.file_path, "w") as f:
            f.writelines(existing_lines)