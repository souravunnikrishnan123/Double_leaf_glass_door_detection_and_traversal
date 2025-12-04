



from ast import keyword
from door_status import detect_door_state


class Door_Status_Detector:
    def __init__(self):
        ns = "door_status_detector"

    def detect(self, ctx, keyword: str = "color_based"):

        if getattr(ctx, f"roi_left_{keyword}") is None or getattr(ctx, f"roi_right_{keyword}") is None:
            return None

        # Call detector and then assign views to context separately
        door_state, passability_view, bird_eye_view = detect_door_state(
            ctx.depth_image_in_meters,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            getattr(ctx, f"color_image_{keyword}"),
            getattr(ctx, f"roi_left_{keyword}"),
            getattr(ctx, f"roi_right_{keyword}"),
            roi_width=240,
            margin=10,
            threshold=0.05,
            z_door_depth=getattr(ctx, f"door_depth_m_{keyword}"),
            keyword = keyword
        )

        # Persist visualizations in context
        setattr(ctx, f"passability_view_{keyword}", passability_view)
        setattr(ctx, f"bird_eye_view_{keyword}", bird_eye_view)

        if door_state is not None:
            setattr(ctx, f"door_state_{keyword}", door_state)
            return door_state

        return None  # Stay in the current state if door state cannot be determined