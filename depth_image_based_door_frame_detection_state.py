from Frame_data import BaseState
from depth_based_detection import depth_based_edge_detection
from Frame_data import FrameContext
import numpy as np

class depth_image_based_door_frame_detection_state(BaseState):
    def __init__(self):
        super().__init__("depth_image_based_door_frame_detection_state")

    def do_action(self, ctx: FrameContext):

        roi_polygon_left_based_on_depth_image, roi_polygon_right_based_on_depth_image, mean_z_depth_to_frame_based_on_depth_image, sobel_vis_color = depth_based_edge_detection(ctx.depth_frame, ctx.depth_image_in_meters, ctx.color_image_depth_based, MIN_DEPTH =1.8, MAX_DEPTH = 2.5, DEPTH_RANGE=(1.7, 2.3), detected_plane=ctx.detected_plane, PHYSICAL_GRADIENT_THRESHOLD=0.25)
        ctx.sobel_vis_color = sobel_vis_color

        if roi_polygon_left_based_on_depth_image is not None and roi_polygon_right_based_on_depth_image is not None:
            ctx.roi_left_depth_based = roi_polygon_left_based_on_depth_image
            ctx.roi_right_depth_based = roi_polygon_right_based_on_depth_image
            ctx.door_depth_m_depth_based = float(mean_z_depth_to_frame_based_on_depth_image) if mean_z_depth_to_frame_based_on_depth_image is not None else None
            return "depth_image_based_detecting_door_state"
        return None  # Stay in the current state if frame detection fails
        