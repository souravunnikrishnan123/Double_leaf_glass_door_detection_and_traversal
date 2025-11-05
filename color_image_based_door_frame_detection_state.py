from Frame_data import BaseState
from color_image_based_frame_detection import color_image_based_frame_detection
from Frame_data import FrameContext

class color_image_based_door_frame_detection_state(BaseState):
    def __init__(self):
        super().__init__("color_image_based_door_frame_detection_state")

    def do_action(self, ctx: FrameContext):

        if ctx.detected_plane is None:
            return None

        roi_polygon_left_based_on_color_image, roi_polygon_right_based_on_color_image, mean_z_depth_to_frame_based_on_color_image, edges = color_image_based_frame_detection(ctx.detected_plane, ctx.color_image_color_based, ctx.depth_frame, DEPTH_RANGE = (1.7, 2.3))
        
        ctx.edges = edges
        if roi_polygon_left_based_on_color_image is not None and roi_polygon_right_based_on_color_image is not None:
            ctx.roi_left_color_based = roi_polygon_left_based_on_color_image
            ctx.roi_right_color_based = roi_polygon_right_based_on_color_image
            ctx.door_depth_m_color_based = float(mean_z_depth_to_frame_based_on_color_image) if mean_z_depth_to_frame_based_on_color_image is not None else None
            return "color_image_based_detecting_door_state"

        return None  # Stay in the current state if frame detection fails
        