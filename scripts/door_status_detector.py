"""Infer whether each side of a detected glass door is open or closed."""

#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from duration import get_duration_seconds


from processing_classes import backproject_depth_to_points




class Door_Status_Detector:
    """
    Classify a color- or depth-derived frame pair using side ROI depths.

    ``keyword`` selects the matching fields on ``FrameContext``; for example,
    ``"color_based"`` reads ``roi_left_color_based`` and draws on
    ``color_image_color_based``.

    Attributes:
        keyword:
            Branch suffix used for dynamic context attribute lookup.

        roi_width:
            Configured ROI width retained for the status detector namespace.

        margin:
            Horizontal pixel offset applied away from each frame edge.

        threshold:
            Minimum fraction of near-door points required to call a side
            consistent with a closed pane.

        timer:
            Named duration recorder used for computation and visualization.
    """

    def __init__(self, keyword: str = "color_based"):
        """
        Initialize one branch-specific door-status detector.

        Args:
            keyword:
                Context suffix, normally ``"color_based"`` or
                ``"depth_based"``.

        Notes:
            Separate instances are used by the current dual-branch state so
            branch selection remains immutable during detection.
        """
        ns = "~door_status_detector"
        self.keyword = keyword
        # Core ROI params
        self.roi_width = rospy.get_param(f"{ns}/roi_width",240)
        self.margin = rospy.get_param(f"{ns}/margin", 10)
        self.threshold = rospy.get_param(f"{ns}/threshold", 0.05)
        # Backprojection stride for the side ROIs. Exposed as a parameter so it can
        # track the input resolution instead of being fixed in code.
        self.roi_subsample = rospy.get_param(f"{ns}/roi_subsample", 4)
        # Diagnostic overlays cost a full scatter write per ROI, four times per
        # frame, so they are gated here rather than relying on no-op drawing calls.
        self.enable_visualization = rospy.get_param("~enable_visualization", True)

        #duration timer
        self.timer = get_duration_seconds()

    
    def offset_roi_polygon(self, roi_polygon, side="left"):
        """
        Shift a side ROI away from the detected frame line.

        Args:
            roi_polygon:
                ``N x 2`` polygon of image coordinates, or ``None``.

            side:
                ``"left"`` applies a negative x offset; any other value applies
                a positive offset.

        Returns:
            Shifted copy of the polygon, or ``None`` when no polygon is given.

        Notes:
            Moving the ROI away from the frame reduces contamination when a
            Hough line lies slightly inside the physical frame boundary.
        """
        if roi_polygon is None:
            return None
        # Move outward from the center frame, not merely in a fixed image direction.
        offset = - self.margin if side == "left" else self.margin
        roi_polygon_offset = roi_polygon.copy()
        roi_polygon_offset[:, 0] += offset  # Shift x-coordinates
        return roi_polygon_offset



    def check_side_roi_against_door(self, depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, color = (0, 255, 0)):
        """
        Return the fraction of an ROI that lies near the door reference depth.

        Points are backprojected from the polygon, compared with the frame
        depth, and filtered by surface normal to reduce floor influence.

        Args:
            depth_image_in_meters:
                Aligned ``H x W`` metric depth image.

            fx:
                Horizontal focal length in pixels.

            fy:
                Vertical focal length in pixels.

            cx:
                Horizontal principal point in pixels.

            cy:
                Vertical principal point in pixels.

            color_image:
                Optional BGR image modified with polygon and point overlays.

            roi_polygon:
                Image-space polygon defining the side region.

            door_depth:
                Reference frame depth in meters.

            color:
                BGR color used to outline the ROI.

        Returns:
            Fraction of backprojected ROI points that remain within five
            percent of ``door_depth`` after normal filtering. Empty
            backprojection returns ``0.0``.

        Notes:
            Backprojection begins at 90 percent of the door depth and uses a
            stride of four. Candidate points are blue and matching points are
            cyan in the diagnostic image.
        """

        self.timer.start(f"check_side_roi_against_door {self.keyword}")
        H, W = depth_image_in_meters.shape
        
        # Include everything behind the frame while dropping foreground people
        # or robot parts that should not decide whether the pane is closed.
        filtered_points, filtered_uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=door_depth * 20, min_depth=door_depth * 0.9, subsample=self.roi_subsample, roi_polygon = roi_polygon)

        if len(filtered_points) == 0:
            return 0.0

        # Backprojection already returns float32 points and integer pixel indices;
        # keeping the indices integral avoids a float round trip before drawing.
        filtered_uv = np.asarray(filtered_uv, dtype=np.int32)

        # A closed pane or frame returns points close to the confirmed door range;
        # an opening mostly exposes surfaces much farther away.
        close_mask = np.abs(filtered_points[:, 2] - door_depth) <= door_depth * 0.05
        close_points = filtered_points[close_mask]
        close_uv = filtered_uv[close_mask]

        # Range alone can include the floor where it crosses the door plane.
        # Surface orientation removes that false support.
        #filter out the points from floor using normal calculation. because we only have few points in close_points
        pc_nf = o3d.geometry.PointCloud()
        pc_nf.points = o3d.utility.Vector3dVector(close_points)
        try:
            pc_nf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=40))
            pc_nf.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
            normals = np.asarray(pc_nf.normals)

        except Exception:
            normals = None

        if normals is not None:
            ny_thr = 0.7 
            vertical_mask = np.abs(normals[:, 1]) < ny_thr #(cos^-1(0.7) ≈ 45°)
            close_points = close_points[vertical_mask]
            close_uv = close_uv[vertical_mask]


            
        #print(len(bottom_points))
        # Normalize by all sampled ROI points so the threshold is independent of ROI size.
        fraction_close = close_points.shape[0] / filtered_points.shape[0]

        #print(f"Fraction close: {fraction_close:.3f}")
        self.timer.stop(f"check_side_roi_against_door {self.keyword}")
        
        self.timer.start(f"check_side_roi_against_door--> visualization {self.keyword}")
        # Visualization. The scatter writes below are plain NumPy, so unlike the
        # cv2 drawing calls they are not disabled by the no-op patching and must be
        # skipped explicitly when diagnostics are off.
        if color_image is not None and self.enable_visualization:
            # Draw ROI polygon
            cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)

            # Validity masks to avoid out-of-bounds
            valid_filtered = (filtered_uv[:, 0] >= 0) & (filtered_uv[:, 0] < W) & (filtered_uv[:, 1] >= 0) & (filtered_uv[:, 1] < H)
            valid_close = (close_uv[:, 0] >= 0) & (close_uv[:, 0] < W) & (close_uv[:, 1] >= 0) & (close_uv[:, 1] < H)

            # Draw "all filtered points" as blue
            color_image[filtered_uv[valid_filtered, 1], filtered_uv[valid_filtered, 0]] = (255, 0, 0)
            # Draw "close to door depth" points as cyan
            color_image[close_uv[valid_close, 1], close_uv[valid_close, 0]] = (0, 255, 255)


        self.timer.stop(f"check_side_roi_against_door--> visualization {self.keyword}")                 
        return fraction_close




    def detect(self, ctx):

        """
        Classify the door from left- and right-side depth consistency.

        A side is consistent when enough points remain near the door plane.
        Two consistent sides mean closed; one inconsistent side identifies the
        opening direction.

        Args:
            ctx:
                Shared frame context containing branch-specific ROIs, frame
                depth, visualization image, metric depth, and intrinsics.

        Returns:
            Tuple ``(door_state, open_side_roi)``. ``door_state`` is one of
            ``"closed"``, ``"open_left"``, ``"open_right"``, or ``"unknown"``.
            The polygon is returned only for an open-side label.

        Raises:
            AttributeError:
                If context fields for the configured ``keyword`` do not exist.

        Notes:
            The method draws the decision and both side-match fractions on the
            branch visualization image.
        """

        #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
        #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
        roi_left_offset = self.offset_roi_polygon(getattr(ctx, f"roi_left_{self.keyword}"), side="left")
        roi_right_offset = self.offset_roi_polygon(getattr(ctx, f"roi_right_{self.keyword}"), side="right")


        #check depth consistency
        left_match = self.check_side_roi_against_door(ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{self.keyword}"), roi_left_offset, getattr(ctx, f"door_depth_m_{self.keyword}"), color=(255, 0, 255))
        right_match = self.check_side_roi_against_door(ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{self.keyword}"), roi_right_offset, getattr(ctx, f"door_depth_m_{self.keyword}"), color=(0, 255, 255))

        rospy.logdebug("Left match: %.2f, Right match: %.2f", left_match, right_match)

        # Each side is judged independently; the asymmetric cases tell us which
        # half of the doorway is open.
        #decision
        left_consistent = left_match > self.threshold
        right_consistent = right_match > self.threshold

        if left_consistent and right_consistent:
            door_state = "closed"
        elif left_consistent and not right_consistent:
            door_state = "open_right"
        elif not left_consistent and right_consistent:
            door_state = "open_left"
        else:
            door_state = "unknown"



        roi_open_side = None
        if door_state == "open_left":
            roi_open_side = roi_left_offset
        elif door_state == "open_right":
            roi_open_side = roi_right_offset

        # Overlay decision text
        cv2.putText(getattr(ctx, f"color_image_{self.keyword}"), f"Door State: {door_state}", (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        
        cv2.putText(getattr(ctx, f"color_image_{self.keyword}"), f"Left match: {left_match:.2f}", (30, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(getattr(ctx, f"color_image_{self.keyword}"), f"Right match: {right_match:.2f}", (30, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return door_state, roi_open_side
