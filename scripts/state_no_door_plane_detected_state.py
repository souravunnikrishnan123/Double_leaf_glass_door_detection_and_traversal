

from Frame_data import FrameContext,BaseState
import numpy as np
from check_if_passable import BirdsEyePassabilityPipeline


class create_full_view_bird_eye_view_state(BaseState):
    def __init__(self):
        super().__init__("create_full_view_bird_eye_view_state")

    def do_action(self, ctx: FrameContext):
        full_image_roi = np.array([[0,0],[ctx.color_image.shape[1]-1,0],[ctx.color_image.shape[1]-1,ctx.color_image.shape[0]-1],[0,ctx.color_image.shape[0]-1]])
        keyword = "full_image_view"
        pipeline_full_image = BirdsEyePassabilityPipeline()
        point_cloud_vertical_ratio, passability_view, bird_eye_view = pipeline_full_image.run(ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, ctx.color_image, full_image_roi, door_depth=2.0, keyword=keyword)



        # Persist visualizations in context
        setattr(ctx, f"passability_view_{keyword}", passability_view)
        setattr(ctx, f"bird_eye_view_{keyword}", bird_eye_view)