# AI Coding Agent Guide for robodog_glass_door_detection

This repository is a ROS (catkin) package, but the current pipeline runs as a plain Python app that processes Intel RealSense frames (from a recorded .bag or live camera) and visualizes results. The core logic is a small state machine that fuses color- and depth-based detections to infer a glass door state.

## Big Picture
- Architecture: Single-process loop in `scripts/main.py` reads frames, updates a state machine, and draws OpenCV windows.
- State Machine: `StateMachine` in `scripts/base_state_machine_classes.py` orchestrates 4 top-level states:
  - `idle_state` → `searching_door_plane_state` → `parallel_detection_state` → `combine_door_state` → back to `parallel_detection_state`.
- Data Flow: A shared `FrameContext` (`scripts/Frame_data.py`) carries inputs (depth/color/intrinsics) and intermediate artifacts between states.
- Fusion: Color branch (Canny+Hough on RGB) and Depth branch (Sobel+Hough on Z) each propose ROIs and a door-side label; `combine_door_state` reconciles them.

## Entrypoint & Run
- Entrypoint: `scripts/main.py`.
  - RealSense setup via `setup_realsense_pipeline(bag_file=...)`. Adjust the `.bag` path in `main.py` to run with your data.
- Quick run (no ROS topics are used by default):
  ```bash
  cd /root/catkin_ws/src/robodog_glass_door_detection
  python3 scripts/main.py
  ```
- Catkin context (optional):
  ```bash
  cd /root/catkin_ws
  catkin_make  # or catkin build
  source devel/setup.bash
  rosrun robodog_glass_door_detection main.py  # if installed via catkin_install_python
  ```

## Key Dependencies
- Python: `numpy`, `opencv-python`, `pyrealsense2`, `open3d`.
- ROS msgs (declared in `package.xml`): `rospy`, `sensor_msgs`, `geometry_msgs`, `std_msgs` (not actively published/subscribed in current code).
- RealSense: A working librealsense/pyrealsense2 with access to device or a `.bag` file.

## State Machine Pattern
- Base types in `scripts/Frame_data.py`:
  - `BaseState`: implement `entry_action(ctx)`, `do_action(ctx) -> Optional[str]`, `exit_action(ctx)`.
  - `FrameContext`: shared per-frame data container; mutate fields in states to pass results downstream.
- Transitions: `do_action` returns the next state name or `None` to stay. See `scripts/base_state_machine_classes.py` for timing and logging.
- Current states:
  - Plane gate: `searching_door_plane_state` uses RANSAC (Open3D) to find a vertical plane; only proceeds when the plane is at ~2m (±0.3m).
  - Parallel branch: `parallel_detection_state` advances both branches each frame and renders via `visualization_utils.show_stacked_visualization`.
  - Fusion: `combine_door_state` writes `ctx.door_state_label` and loops back.

## Context Contract (used across states)
- Inputs set in `main.py` each frame: `ctx.depth_image_in_meters`, `ctx.color_image`, `ctx.fx, ctx.fy, ctx.cx, ctx.cy`, `ctx.depth_frame`.
- Color branch sets: `ctx.roi_left_color_based`, `ctx.roi_right_color_based`, `ctx.door_depth_m_color_based`, `ctx.door_state_color_based`, `ctx.edges`.
- Depth branch sets: `ctx.roi_left_depth_based`, `ctx.roi_right_depth_based`, `ctx.door_depth_m_depth_based`, `ctx.door_state_depth_based`, `ctx.sobel_vis_color`.
- Final: `ctx.door_state_label`.

## Algorithm Touchpoints
- Plane finding: `scripts/detect_glass_door_plane.py` → backproject (`create_3d_points_and_detect_ransac_plane.py`), voxel-downsample, normal filter, RANSAC planes, distance + “glassness” heuristics; outlines plane on RGB.
- Color frame lines: `scripts/color_image_based_frame_detection.py` → Canny+Hough, depth checks along lines, line pairing (`find_frame_line_pair.py`), ROI building (`roi.py`).
- Depth frame lines: `scripts/depth_based_detection.py` → Sobel gradient in Z, Hough, robust Z via side ROIs, clustering/merge, pairing, ROI building.
- Door state: `scripts/door_status.py` → offsets ROIs away from frame, checks ROI depth consistency vs door depth, optional passability via `check_if_passable`.
- Viz & tooling: `scripts/visualization_utils.py` (depth colormap, mouse click prints depth), `scripts/duration.py` (simple timers).

## Conventions & Gotchas
- No ROS topics right now: `rospy` is imported but unused; avoid introducing pubs/subs unless you also document topics.
- Keep `do_action` lightweight; main loop runs every frame and must call `cv2.waitKey(1)` (already done in `parallel_detection_state`).
- Intrinsics: Read from depth frame each loop; update `ctx.fx,fy,cx,cy` before calling states.
- RealSense input: When switching from `.bag` to live, enable streams in `setup_realsense_pipeline` (commented lines) and remove `config.enable_device_from_file`.

## Extending the System (examples)
- Add a new state: subclass `BaseState`, register via `sm.add_state(...)`, and return a transition name from `do_action`.
- Add a third branch (e.g., learning-based): follow `parallel_detection_state` pattern—hold child state instances, call `do_action(ctx)`, and write results to new `ctx.*` fields; extend `combine_door_state` to fuse.
- Tuning: Depth ranges and thresholds are passed as args in `color_image_based_frame_detection` and `depth_based_edge_detection`; prefer parameterizing rather than hardcoding in states.

## Where to Start
- For quick iteration, edit `scripts/main.py` bag path and run.
- To debug geometry, set breakpoints in `detect_glass_door_plane` and use the plane overlays and the mouse-depth tool in viz windows.
