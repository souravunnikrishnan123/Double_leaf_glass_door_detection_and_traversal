#!/usr/bin/env python3
"""Create a RealSense capture pipeline and depth-to-color aligner."""

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2

def setup_realsense_pipeline(bag_file=None):
    """
    Start a RealSense pipeline from a bag file or the default live device.

    The SDK chooses the recorded/default streams. Returned depth frames can be
    passed through the aligner so their pixels correspond to the color stream.

    Args:
        bag_file:
            Optional path to a recorded RealSense bag. ``None`` selects the
            SDK's default live-device configuration.

    Returns:
        Tuple ``(pipeline, config, align)`` containing the running pipeline,
        its configuration, and a depth-to-color aligner.

    Raises:
        RuntimeError:
            If the bag/device cannot be opened or streaming cannot start.

    Notes:
        The caller owns the running pipeline and must eventually call
        ``pipeline.stop()``.
    """
    # -------------------------------
    # Initialize RealSense Pipeline
    # -------------------------------
    # Keep playback and live capture behind the same pipeline interface so the
    # rest of the detector does not care where frames originated.
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable depth stream (z16 = 16-bit grayscale)
    #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
    # Enable color stream (bgr8 = standard OpenCV format)
    #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

    if bag_file is not None:
        # A recorded bag already declares its stream formats and frame rates.
        config.enable_device_from_file(bag_file)


    # Start streaming
    pipeline.start(config)

    # Create RealSense post-processing filters
    #spatial = rs.spatial_filter()
    #temporal = rs.temporal_filter()
    #hole_filling = rs.hole_filling_filter()

    # Align depth to color stream so depth and color pixels correspond
    # All later ROI operations assume depth pixels line up with the color image.
    align = rs.align(rs.stream.color)

    return pipeline,config,align
