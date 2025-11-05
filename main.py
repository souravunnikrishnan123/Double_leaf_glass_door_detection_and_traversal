import cv2
import numpy as np
from base_state_machine_classes import StateMachine
from setup_realsense_pipeline import setup_realsense_pipeline
from Frame_data import FrameContext
from idle_state import idle_state
from searching_door_plane_state import searching_door_plane_state
from combine_door_state import combine_door_state
from parallel_detection_state import parallel_detection_state




def main():
    pipeline, config, align = setup_realsense_pipeline(bag_file="/app/realsense_camera_feed/grey_door/grey_door_opening_night_from_IAS_lab_side.bag")
    try:

        sm = StateMachine(ctx=None)
        sm.add_state(idle_state())
        sm.add_state(searching_door_plane_state())
        sm.add_state(parallel_detection_state())
        sm.add_state(combine_door_state())
    

        sm.set_state("idle")

        while True:

            # Wait for next set of frames
            frames = pipeline.wait_for_frames()

            # Align depth to color
            aligned = align.process(frames)

            # Extract depth and color frames
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            # If either frame is not available, skip this loop
            if not depth_frame or not color_frame:
                continue

            # --- Apply RealSense filters ---
            #depth_frame = spatial.process(depth_frame)
            #depth_frame = temporal.process(depth_frame)
            #depth_frame = hole_filling.process(depth_frame)

            # Convert depth to numpy arrays for OpenCV
            # Get depth sensor from the pipeline
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()

            # Depth: must convert to float meters
            depth_scale = depth_sensor.get_depth_scale()
            depth_image_raw = np.asanyarray(depth_frame.get_data())
            # Convert to meters
            depth_image_in_meters = depth_image_raw.astype(float) * depth_scale
            
            # Convert colour image to numpy arrays for OpenCV
            #raw color (8-bit RGB values, already fine for OpenCV, no need of any conversion)
            color_image = np.asanyarray(color_frame.get_data())
            color_image_for_depth_line = color_image.copy()
            color_image_for_ransac = color_image.copy()
            color_image_for_bev = color_image.copy()
            

            # Get intrinsics
            intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy

            # Reuse a persistent context so state outputs persist across frames
            if not hasattr(sm, "ctx") or sm.ctx is None:
                sm.ctx = FrameContext(
                    depth_image_in_meters=depth_image_in_meters,
                    color_image=color_image,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    depth_frame=depth_frame
                )
            else:
                sm.ctx.depth_image_in_meters = depth_image_in_meters
                sm.ctx.color_image = color_image
                sm.ctx.color_image_color_based = color_image.copy()
                sm.ctx.color_image_depth_based = color_image.copy()
                sm.ctx.fx, sm.ctx.fy, sm.ctx.cx, sm.ctx.cy = fx, fy, cx, cy
                sm.ctx.depth_frame = depth_frame


            sm.update(sm.ctx)

            state_name = sm.current_state.name if sm.current_state else "none"
            cv2.putText(color_image, f"State: {state_name}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Robot Vision", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()