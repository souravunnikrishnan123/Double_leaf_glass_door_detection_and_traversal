#!/usr/bin/env python3
"""Save debug images from several topics into a new numbered run folder.

Every launch gets its own folder, ``<results_dir>/run<N>``, numbered one past
the highest run already there, so earlier runs are never overwritten. Each
topic is saved as ``<name>NNNN.png`` with its own counter.
"""

import os
import re

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def create_next_run_dir(results_dir):
    """
    Create the next ``run<N>`` folder under ``results_dir``.

    Args:
        results_dir:
            Folder holding the runs, created if missing.

    Returns:
        Path of the new, empty run folder.

    Notes:
        Numbering continues from the highest existing run, so a deleted run
        leaves a gap rather than being reused. ``os.mkdir`` fails if the
        folder already exists, so two launches started together still get
        different folders.
    """
    os.makedirs(results_dir, exist_ok=True)
    runs = [
        int(m.group(1))
        for m in (re.fullmatch(r"run(\d+)", d) for d in os.listdir(results_dir))
        if m
    ]
    n = max(runs, default=0) + 1
    while True:
        path = os.path.join(results_dir, f"run{n}")
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            n += 1


class ImageRecorder:
    """
    Write every image received on the configured topics into one run folder.

    Attributes:
        run_dir:
            Folder this launch writes to.

        counts:
            Number of images saved so far, per file name prefix.
    """

    def __init__(self):
        """
        Create the run folder and subscribe to the configured topics.

        Notes:
            ``~results_dir`` is the folder holding the runs. ``~topics`` maps
            each file name prefix to the image topic saved under it.
        """
        results_dir = rospy.get_param("~results_dir")
        topics = rospy.get_param("~topics")
        self.run_dir = create_next_run_dir(results_dir)
        self.bridge = CvBridge()
        self.counts = {}
        for name, topic in topics.items():
            self.counts[name] = 0
            # A deeper queue than a live viewer needs: a saved frame should not
            # be dropped because the previous PNG was still being written.
            # buff_size must exceed one stacked debug image.
            rospy.Subscriber(topic, Image, self.save, callback_args=name,
                             queue_size=10, buff_size=2**24)
        rospy.loginfo("image_recorder: saving %s to %s",
                      ", ".join(sorted(topics.values())), self.run_dir)

    def save(self, msg, name):
        """
        Write one image as ``<name>NNNN.png`` in the run folder.

        Args:
            msg:
                ROS image message.

            name:
                File name prefix configured for the message's topic.
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = os.path.join(self.run_dir, f"{name}{self.counts[name]:04d}.png")
        if cv2.imwrite(path, image):
            self.counts[name] += 1
        else:
            rospy.logwarn_throttle(10.0, "image_recorder: could not write %s", path)


if __name__ == "__main__":
    rospy.init_node("image_recorder")
    ImageRecorder()
    rospy.spin()
