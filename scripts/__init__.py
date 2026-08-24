"""
Glass-door detection and traversal components for the Robodog ROS package.

The implementation remains in the ROS ``scripts`` directory. This package
marker lets documentation tools such as pdoc discover the modules together
without changing the layout used by the ROS nodes.
"""

import os as _os
import sys as _sys


# These modules can be imported as a package or launched directly by ROS. The
# path shim keeps the older direct imports working in both cases.
# The existing modules use imports such as ``from duration import ...`` because
# they are also launched directly as ROS scripts. Keep those imports working
# when a documentation tool loads the files through the package name.
_scripts_directory = _os.path.dirname(__file__)
if _scripts_directory not in _sys.path:
    _sys.path.insert(0, _scripts_directory)


# Keep this list explicit: importing every file blindly can start hardware or
# a long-running processing loop as a side effect.
# pdoc respects a package's __all__ when discovering submodules. The legacy
# door_frame_detection script is intentionally omitted because importing it
# starts the RealSense processing loop. draft.py is an incomplete code fragment
# rather than an importable module, so it is omitted as well.
__all__ = [
    "Frame_data",
    "check_if_passable",
    "color_door_detector",
    "depth_door_detector",
    "detect_glass_door_plane",
    "door_status_detector",
    "door_traversal_controller",
    "door_type_detector",
    "duration",
    "evaluate_detected_ransac_planes",
    "extra_functions",
    "find_glass_frame_lines",
    "main",
    "odom_bridge",
    "passability_check_main",
    "post_processing_of_detected_vertical_lines",
    "processing_classes",
    "realsense_bag_bridge",
    "ros_frame_adapter",
    "setup_realsense_pipeline",
    "state_combine_door",
    "state_final",
    "state_full_image_passability_check",
    "state_idle",
    "state_machine_base_classes",
    "state_parallel_detection",
    "state_searching_door_plane",
    "temporal_smoothing_of_frame_line_candidates",
    "visualization_utils",
]
