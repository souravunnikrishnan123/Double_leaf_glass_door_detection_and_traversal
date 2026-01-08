#!/usr/bin/env python3
"""
realsense_bag_bridge.py

Reads a RealSense .bag via pyrealsense2, aligns depth->color using rs.align,
and publishes:
 - color image -> ROS topic (bgr8)
 - aligned depth -> ROS topic (16UC1 or 32FC1 passthrough)
 - color CameraInfo -> ROS topic

Params (namespace ~):
  bag                : path to bag file (string)
  color_topic        : topic name to publish color image (default '/camera/color/image_raw')
  depth_topic        : topic name to publish aligned depth (default '/camera/aligned_depth_to_color/image_raw')
  color_info_topic   : topic name to publish CameraInfo (default '/camera/color/camera_info')
  loop               : bool, repeat playback (default False)

Usage:
  rosrun <pkg> realsense_bag_bridge.py _bag:=/path/to/bag.bag
  or via roslaunch using the launch you posted.
"""
import cv2
import rospy
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
import time

# replace the old intrinsics_to_camera_info with this corrected one
def intrinsics_to_camera_info(intr, frame_id):
    """
    Build a sensor_msgs.msg.CameraInfo from a pyrealsense2 intrinsics object.
    """
    ci = CameraInfo()
    ci.header.frame_id = frame_id
    ci.width = int(intr.width)
    ci.height = int(intr.height)

    # K (3x3 row-major)
    K = [0.0]*9
    K[0] = float(intr.fx)
    K[1] = 0.0
    K[2] = float(intr.ppx)
    K[3] = 0.0
    K[4] = float(intr.fy)
    K[5] = float(intr.ppy)
    K[6] = 0.0
    K[7] = 0.0
    K[8] = 1.0
    ci.K = K

    # Distortion
    ci.distortion_model = 'plumb_bob'
    try:
        # intr.coeffs is typically a list-like of length 5
        ci.D = [float(c) for c in intr.coeffs] if intr.coeffs is not None else [0.0,0.0,0.0,0.0,0.0]
    except Exception:
        ci.D = [0.0,0.0,0.0,0.0,0.0]

    # R (rectification) default to identity
    ci.R = [1.0,0.0,0.0,
            0.0,1.0,0.0,
            0.0,0.0,1.0]

    # P (3x4 projection matrix)
    P = [0.0]*12
    P[0] = float(intr.fx)
    P[1] = 0.0
    P[2] = float(intr.ppx)
    P[3] = 0.0
    P[4] = 0.0
    P[5] = float(intr.fy)
    P[6] = float(intr.ppy)
    P[7] = 0.0
    P[8] = 0.0
    P[9] = 0.0
    P[10] = 1.0
    P[11] = 0.0
    ci.P = P

    return ci


def main():
    rospy.init_node('realsense_bag_bridge', anonymous=False)
    rospy.loginfo("realsense_bag_bridge node started.")

    bag_file = rospy.get_param('~bag', rospy.get_param('bag', None))
    if bag_file is None:
        rospy.logerr("No bag file provided. Set ~bag param or bag param.")
        return

    color_topic = rospy.get_param('~color_topic', '/camera/color/image_raw')
    depth_topic = rospy.get_param('~depth_topic', '/camera/aligned_depth_to_color/image_raw')
    color_info_topic = rospy.get_param('~color_info_topic', '/camera/color/camera_info')
    loop_bag = rospy.get_param('~loop', False)

    pub_color = rospy.Publisher(color_topic, Image, queue_size=2)
    pub_depth = rospy.Publisher(depth_topic, Image, queue_size=2)
    pub_color_info = rospy.Publisher(color_info_topic, CameraInfo, queue_size=1)

    bridge = CvBridge()

    # Setup RealSense pipeline
    loop_bag = rospy.get_param('~loop', True)  
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(bag_file, repeat_playback=bool(loop_bag))
    # Let the SDK pick streams from bag (recorded streams)
    rospy.loginfo("Starting RealSense pipeline for bag: %s", bag_file)
    profile = pipeline.start(cfg)

    align_to = rs.stream.color
    align = rs.align(align_to)
    rate = rospy.Rate(30.0)  # fallback loop rate
    # -------------------------
    # FPS REDUCTION CONTROL
    # -------------------------
    frame_skip = rospy.get_param("~frame_skip", 0)  
    frame_counter = 0
    resize_scale = rospy.get_param("~resize_scale", 1.0)


    try:
        while not rospy.is_shutdown():
            # Get frames (blocks until next frame set)
            try:
            # wait longer (10s) to avoid spurious timeouts
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                rospy.logwarn("wait_for_frames timeout / runtime error: %s", e)
                # If bag is not looping, we probably reached the end -> restart or exit
                if not loop_bag:
                    rospy.loginfo("End of bag reached and repeat_playback=False -> exiting loop.")
                    break
                # If loop_bag True, attempt to restart the pipeline once
                rospy.loginfo("Attempting to restart pipeline for looped playback...")
                try:
                    pipeline.stop()
                except Exception:
                    pass
                time.sleep(0.2)
                try:
                    profile = pipeline.start(cfg)
                    rospy.loginfo("Pipeline restarted.")
                    continue
                except Exception as e2:
                    rospy.logerr("Failed to restart pipeline: %s", e2)
                    rospy.loginfo("Sleeping 1s before next retry.")
                    time.sleep(1.0)
                    continue

            frame_counter += 1
            if frame_counter % (frame_skip + 1) != 0:
                continue
            # Process alignment
            aligned_frames = align.process(frames)

            # Fetch color and depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            

            if not color_frame or not depth_frame:
                rospy.logwarn_throttle(5.0, "No color or depth frame in this iteration")
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())  # typically RGB or BGR depending on bag
            depth_image = np.asanyarray(depth_frame.get_data())  # typically uint16 (mm) or float32 (m)


            if resize_scale != 1.0:
                new_w = int(color_image.shape[1] * resize_scale)
                new_h = int(color_image.shape[0] * resize_scale)
                color_image = cv2.resize(color_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                depth_image = cv2.resize(depth_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Try to detect color encoding; many bags store color as RGB8 -> cv_bridge expects 'rgb8' or convert to bgr8
            # We'll publish as bgr8 for OpenCV compatibility. If color image is RGB, swap channels.
            # A quick heuristic: check number of channels
            if color_image.shape[2] == 3:
                # Heuristic: if colors look like RGB (not guaranteed). We'll convert RGB->BGR to be safe.
                color_bgr = color_image[..., ::-1].copy()  # RGB->BGR
            else:
                color_bgr = color_image

            # Prepare image messages
            # For color: publish bgr8
            color_msg = bridge.cv2_to_imgmsg(color_bgr, encoding='bgr8')

            # For depth: preserve raw encoding as passthrough so downstream knows actual type
            # If depth array dtype is uint16 -> publish as '16UC1'; if float32 -> '32FC1'
            if depth_image.dtype == np.uint16:
                depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')  # keep 16UC1
            elif depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
                depth_msg = bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding='32FC1')
            else:
                # fallback: publish as passthrough and let downstream interpret
                depth_msg = bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')

            # Timestamping: use SDK frame timestamp (milliseconds) converted to ROS time
            # frames.get_timestamp() returns ms since start of bag; use that if available
            try:
                ts_ms = frames.get_timestamp()  # milliseconds
                ros_time = rospy.Time.from_sec(float(ts_ms) / 1000.0)
            except Exception:
                ros_time = rospy.Time.now()

            color_msg.header.stamp = ros_time
            depth_msg.header.stamp = ros_time
            color_msg.header.frame_id = "camera_color_frame"
            depth_msg.header.frame_id = "camera_color_frame"  # aligned depth uses color frame

            # Publish CameraInfo derived from the color frame intrinsics (once every frame)
            try:
                color_intr = color_frame.profile.as_video_stream_profile().intrinsics
                ci = intrinsics_to_camera_info(color_intr, frame_id="camera_color_frame")
                ci.header.stamp = ros_time
                pub_color_info.publish(ci)
            except Exception as e:
                rospy.logwarn_throttle(10.0, "Failed to build CameraInfo: %s", e)


            # Publish images
            pub_color.publish(color_msg)
            pub_depth.publish(depth_msg)

            # small sleep to avoid busy spin if pipeline is faster than consumer
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        rospy.loginfo("realsense_bag_bridge stopped.")


if __name__ == "__main__":
    main()
