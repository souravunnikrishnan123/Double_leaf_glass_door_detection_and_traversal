#!/usr/bin/env python3
"""Save debug images from several topics into a new numbered run folder.

Every launch gets its own folder, ``<results_dir>/run<N>``, numbered one past
the highest run already there, so earlier runs are never overwritten. Each
topic is saved as ``<name>NNNN.png`` with its own counter.

A paired topic is not saved on its own. Each time the topic it is paired with
saves ``<with>NNNN.png``, the paired frame whose header stamp is closest to
that image's stamp is saved as ``<name>NNNN.png`` with the same number, so the
two views of one instant line up by file name.
"""

import os
import re
import threading
from collections import deque

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


class PairedTopic:
    """
    Buffer the recent frames of one topic so the frame nearest a stamp can be saved.

    Attributes:
        name:
            File name prefix the matched frames are saved under.

        topic:
            Image topic being buffered.

        buffer:
            How much stamp time of frames is kept, as a ``rospy.Duration``.

        frames:
            Buffered messages, oldest first. Guarded by ``lock``, because the
            subscriber filling it and the recorder reading it run in different
            threads.
    """

    def __init__(self, name, topic, buffer_s):
        """
        Subscribe to ``topic`` and start buffering.

        Args:
            name:
                File name prefix for the matched frames.

            topic:
                Image topic to buffer.

            buffer_s:
                Seconds of stamp time to keep. It must cover the delay between
                a frame being captured and the image it is paired with being
                published, or the matching frame is already gone.
        """
        self.name = name
        self.topic = topic
        self.buffer = rospy.Duration.from_sec(buffer_s)
        self.frames = deque()
        self.lock = threading.Lock()
        rospy.Subscriber(topic, Image, self.store, queue_size=10, buff_size=2**24)

    def store(self, msg):
        """
        Add one frame and drop the frames older than the buffer.

        Args:
            msg:
                ROS image message.
        """
        with self.lock:
            # Time going backwards means the simulation was reset; the old
            # frames can no longer match anything.
            if self.frames and msg.header.stamp < self.frames[-1].header.stamp:
                self.frames.clear()
            self.frames.append(msg)
            while msg.header.stamp - self.frames[0].header.stamp > self.buffer:
                self.frames.popleft()

    def closest(self, stamp):
        """
        Return the buffered frame whose stamp is closest to ``stamp``.

        Args:
            stamp:
                ``rospy.Time`` to match. A zero stamp matches the newest frame.

        Returns:
            The matching message, or ``None`` if nothing is buffered yet.
        """
        with self.lock:
            frames = list(self.frames)
        if not frames:
            return None
        if stamp.is_zero():
            return frames[-1]
        return min(frames, key=lambda m: abs((m.header.stamp - stamp).to_sec()))


class ImageRecorder:
    """
    Write every image received on the configured topics into one run folder.

    Attributes:
        run_dir:
            Folder this launch writes to.

        bridge:
            ``CvBridge`` used to convert messages to BGR images.

        counts:
            Number of images saved so far, per file name prefix.

        paired:
            Paired topics to save alongside each saved image, keyed by the
            file name prefix of the topic they are paired with.

        max_pair_offset:
            Stamp difference in seconds above which a pair is saved with a
            warning.
    """

    def __init__(self):
        """
        Create the run folder and subscribe to the configured topics.

        Notes:
            ``~results_dir`` is the folder holding the runs. ``~topics`` maps
            each file name prefix to the image topic saved under it.
            ``~paired`` maps a file name prefix to ``{topic, with}``: frames of
            ``topic`` are saved only next to images saved under the ``with``
            prefix. ``~paired_buffer_s`` and ``~paired_max_offset_s`` tune the
            matching.
        """
        results_dir = rospy.get_param("~results_dir")
        topics = rospy.get_param("~topics")
        paired = rospy.get_param("~paired", {})
        buffer_s = rospy.get_param("~paired_buffer_s", 2.0)
        self.max_pair_offset = rospy.get_param("~paired_max_offset_s", 0.1)
        self.run_dir = create_next_run_dir(results_dir)
        self.bridge = CvBridge()
        self.counts = {}
        self.paired = {}
        for name, cfg in paired.items():
            if cfg["with"] not in topics:
                rospy.logerr("image_recorder: %s is paired with %s, which is not a saved topic; not saving it",
                             name, cfg["with"])
                continue
            self.paired.setdefault(cfg["with"], []).append(
                PairedTopic(name, cfg["topic"], buffer_s))
        for name, topic in topics.items():
            self.counts[name] = 0
            # A deeper queue than a live viewer needs: a saved frame should not
            # be dropped because the previous PNG was still being written.
            # buff_size must exceed one stacked debug image.
            rospy.Subscriber(topic, Image, self.save, callback_args=name,
                             queue_size=10, buff_size=2**24)
        rospy.loginfo("image_recorder: saving %s to %s",
                      ", ".join(sorted(topics.values())), self.run_dir)
        for name, pairs in self.paired.items():
            for p in pairs:
                rospy.loginfo("image_recorder: saving %s as %sNNNN.png next to each %sNNNN.png",
                              p.topic, p.name, name)

    def write(self, msg, filename):
        """
        Write one image into the run folder.

        Args:
            msg:
                ROS image message.

            filename:
                File name inside the run folder.

        Returns:
            Whether the file was written.
        """
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = os.path.join(self.run_dir, filename)
        if cv2.imwrite(path, image):
            return True
        rospy.logwarn_throttle(10.0, "image_recorder: could not write %s", path)
        return False

    def save(self, msg, name):
        """
        Write one image as ``<name>NNNN.png``, plus its paired frames.

        Args:
            msg:
                ROS image message.

            name:
                File name prefix configured for the message's topic.
        """
        index = self.counts[name]
        if not self.write(msg, f"{name}{index:04d}.png"):
            return
        self.counts[name] += 1
        for paired in self.paired.get(name, []):
            self.save_paired(paired, msg.header.stamp, index)

    def save_paired(self, paired, stamp, index):
        """
        Write the ``paired`` frame closest to ``stamp`` as ``<paired.name>NNNN.png``.

        Args:
            paired:
                ``PairedTopic`` to take the frame from.

            stamp:
                Stamp of the image just saved.

            index:
                Number of the image just saved, reused so the pair shares it.

        Notes:
            A pair further apart than ``max_pair_offset`` is still saved, so
            numbering stays aligned, but logs a warning; raise
            ``~paired_buffer_s`` if that happens often.
        """
        frame = paired.closest(stamp)
        if frame is None:
            rospy.logwarn_throttle(10.0, "image_recorder: no frame on %s yet, %s%04d.png not saved",
                                   paired.topic, paired.name, index)
            return
        offset = abs((frame.header.stamp - stamp).to_sec())
        if not stamp.is_zero() and offset > self.max_pair_offset:
            rospy.logwarn_throttle(10.0, "image_recorder: %s%04d.png is %.3f s away from the image it is paired with",
                                   paired.name, index, offset)
        self.write(frame, f"{paired.name}{index:04d}.png")


if __name__ == "__main__":
    rospy.init_node("image_recorder")
    ImageRecorder()
    rospy.spin()
