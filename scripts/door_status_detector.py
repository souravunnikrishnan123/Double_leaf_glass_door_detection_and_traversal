


#!/usr/bin/env python3
import rospy
from door_status import detect_door_state


class Door_Status_Detector:
    def __init__(self, keyword: str = "color_based"):
        ns = "~door_status_detector"
        self.keyword = keyword
        # Core ROI params
        self.roi_width = rospy.get_param(f"{ns}/roi_width",240)
        self.margin = rospy.get_param(f"{ns}/margin", 10)
        self.threshold = rospy.get_param(f"{ns}/threshold", 0.05)


    def detect(self, ctx):

        if getattr(ctx, f"roi_left_{self.keyword}") is None or getattr(ctx, f"roi_right_{self.keyword}") is None:
            return None

        # Call detector and then assign views to context separately
        door_state, passability_view, bird_eye_view = detect_door_state(
            ctx.depth_image_in_meters,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            getattr(ctx, f"color_image_{self.keyword}"),
            getattr(ctx, f"roi_left_{self.keyword}"),
            getattr(ctx, f"roi_right_{self.keyword}"),
            roi_width=self.roi_width,
            margin=self.margin,
            threshold=self.threshold,
            z_door_depth=getattr(ctx, f"door_depth_m_{self.keyword}"),
            keyword=self.keyword 
        )

        # Persist visualizations in context
        setattr(ctx, f"passability_view_{self.keyword}", passability_view)
        setattr(ctx, f"bird_eye_view_{self.keyword}", bird_eye_view)

        if door_state is not None:
            setattr(ctx, f"door_state_{self.keyword}", door_state)
            return door_state

        return None  # Stay in the current state if door state cannot be determined