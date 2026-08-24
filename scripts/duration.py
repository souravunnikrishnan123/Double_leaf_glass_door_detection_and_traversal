#!/usr/bin/env python3
"""Lightweight moving-average timing for named pipeline stages."""

import rospy
import time
from collections import defaultdict, deque
import os


class get_duration_seconds():
    """
    Collect recent durations for labels shared across detector instances.

    Timing storage is class-level so calls made by separate pipeline objects
    contribute to the same report. Each label retains its five latest samples.

    Attributes:
        start_times:
            Mapping from active labels to their wall-clock start timestamps.

        durations:
            Mapping from labels to deques containing at most five durations.

        file_path:
            Output path loaded from the ``~durations_file_path`` ROS
            parameter.
    """

    start_times = {}
    durations = defaultdict(lambda: deque(maxlen=5))
    file_path = None # IMPORTANT: defer initialization until after rospy is initialized
    
    def __init__(self):
        """
        Initialize timing output configuration.

        Notes:
            Constructing any timer updates the class-wide ``file_path``. ROS
            must already be initialized so the private parameter can be read.
        """
        get_duration_seconds.file_path = rospy.get_param("~durations_file_path")
        
    
    def start(self, label=""):
        """
        Record the start time for a named operation.

        Args:
            label:
                Identifier used by the matching call to :meth:`stop`.

        Notes:
            Starting an already-active label overwrites its earlier timestamp.
        """
        # Labels let callers time pipeline stages without holding timer objects.
        get_duration_seconds.start_times[label] = time.time()

    def stop(self, label=""):
        """
        Finish a named operation and store its elapsed time.

        Args:
            label:
                Identifier previously passed to :meth:`start`.

        Returns:
            ``None``. The duration is appended to class-wide history.

        Raises:
            ValueError:
                If no start timestamp exists for ``label``.
        """
        if label not in get_duration_seconds.start_times:
            raise ValueError(f"No start time recorded for label '{label}'")
        # Remove completed starts so a forgotten stop cannot pollute later samples.
        duration = time.time() - get_duration_seconds.start_times[label]
        get_duration_seconds.durations[label].append(duration)
        del get_duration_seconds.start_times[label]

    @classmethod
    def write_text_file(cls):
        """
        Write the latest moving average for every measured label.

        Existing labels in the output file are replaced in place and new
        labels are appended.

        Raises:
            OSError:
                If the configured timing file cannot be read or written.

        Notes:
            Moving averages use the samples retained in each five-element
            deque. This method does not clear timing history.
        """
        # Preserve unknown lines in the file; developers often add their own notes.
        if os.path.exists(get_duration_seconds.file_path):
            with open(get_duration_seconds.file_path,"r") as f:
                existing_lines = f.readlines()
                existing_labels = [line.split(":")[0] for line in existing_lines]
        else:
            existing_lines = []
            existing_labels = []
        
        for label, values in get_duration_seconds.durations.items():
                if len(values) > 0:
                    # Five recent samples smooth spikes but still expose a slowdown quickly.
                    moving_avg = sum(values) / len(values)
                    if label in existing_labels:
                        index_of_existing_labels = existing_labels.index(label)
                        existing_lines[index_of_existing_labels] = f"{label}: {moving_avg:.6f}\n"
                    else:
                        existing_lines.append(f"{label}: {moving_avg:.6f}\n")
        with open(get_duration_seconds.file_path, "w") as f:
            f.writelines(existing_lines)
