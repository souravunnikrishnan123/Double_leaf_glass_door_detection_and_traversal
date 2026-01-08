


#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
from duration import get_duration_seconds


from processing_classes import backproject_depth_to_points




def offset_roi_polygon(roi_polygon, margin, side="left"):
    """
    Offset the ROI polygon horizontally by margin.
    For left ROI, shift left; for right ROI, shift right.
    """
    if roi_polygon is None:
        return None
    offset = -margin if side == "left" else margin
    roi_polygon_offset = roi_polygon.copy()
    roi_polygon_offset[:, 0] += offset  # Shift x-coordinates
    return roi_polygon_offset



def check_side_roi_against_door(depth_image_in_meters, fx, fy, cx, cy, color_image, roi_polygon, door_depth, color = (0, 255, 0), keyword="color"):
    """
    Check how much of ROI depth matches door reference depth.
    """
    timer = get_duration_seconds()
    timer.start(f"check_side_roi_against_door {keyword}")
    H, W = depth_image_in_meters.shape
    
    filtered_points, filtered_uv,_ = backproject_depth_to_points(depth_image_in_meters, fx, fy, cx, cy, max_depth=door_depth * 20, min_depth=door_depth * 0.9, subsample=4, roi_polygon = roi_polygon)
    
    if len(filtered_points) == 0:
        return 0.0

    filtered_points = np.asarray(filtered_points,dtype=np.float32)
    filtered_uv = np.asarray(filtered_uv,dtype=np.float32)

    close_mask = np.abs(filtered_points[:, 2] - door_depth) <= door_depth * 0.05
    close_points = filtered_points[close_mask]
    close_uv = filtered_uv[close_mask]

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
    fraction_close = close_points.shape[0] / filtered_points.shape[0]

    #print(f"Fraction close: {fraction_close:.3f}")
    timer.stop(f"check_side_roi_against_door {keyword}")
    
    timer.start(f"check_side_roi_against_door--> visualization {keyword}")
    # Visualization
    if color_image is not None:
        # Draw ROI polygon
        cv2.polylines(color_image, [roi_polygon.astype(np.int32)], isClosed=True, color=color, thickness=2)
        
        # Prepare coordinate arrays
        uv_filtered = filtered_uv.astype(np.int32)
        uv_close = close_uv.astype(np.int32)
        #uv_removed = removed_close_uv.astype(np.int32)

        # Validity masks to avoid out-of-bounds
        valid_filtered = (uv_filtered[:, 0] >= 0) & (uv_filtered[:, 0] < W) & (uv_filtered[:, 1] >= 0) & (uv_filtered[:, 1] < H)
        valid_close = (uv_close[:, 0] >= 0) & (uv_close[:, 0] < W) & (uv_close[:, 1] >= 0) & (uv_close[:, 1] < H)
        #valid_removed = (uv_removed[:, 0] >= 0) & (uv_removed[:, 0] < W) & (uv_removed[:, 1] >= 0) & (uv_removed[:, 1] < H)

        # Draw "all filtered points" as blue
        color_image[uv_filtered[valid_filtered][:, 1], uv_filtered[valid_filtered][:, 0]] = (255, 0, 0)
        # Draw "close to door depth" points as cyan
        color_image[uv_close[valid_close][:, 1], uv_close[valid_close][:, 0]] = (0, 255, 255)
        # Draw "removed (coplanar/floor)" points as green
        #color_image[uv_removed[valid_removed][:, 1], uv_removed[valid_removed][:, 0]] = (0, 255, 0)


    timer.stop(f"check_side_roi_against_door--> visualization {keyword}")                 
    return fraction_close




class Door_Status_Detector:
    def __init__(self, keyword: str = "color_based"):
        ns = "~door_status_detector"
        self.keyword = keyword
        # Core ROI params
        self.roi_width = rospy.get_param(f"{ns}/roi_width",240)
        self.margin = rospy.get_param(f"{ns}/margin", 10)
        self.threshold = rospy.get_param(f"{ns}/threshold", 0.05)


    def detect(self, ctx):

        """
        Decide OPEN/CLOSED based on ROIs and door depth reference.
        """

        #reuse roi_polygon_left and roi_polygon_right from door_frame_detection.py but with a margin offset. becuase we dont want to
        #include the vertical door frame line pixels in the ROI for depth checking to know the status of door( especially when the detected RGB houghline are not at the frame end but slightly inward)
        roi_left_offset = offset_roi_polygon(getattr(ctx, f"roi_left_{self.keyword}"), self.margin, side="left")
        roi_right_offset = offset_roi_polygon(getattr(ctx, f"roi_right_{self.keyword}"), self.margin, side="right")


        #check depth consistency
        left_match = check_side_roi_against_door(ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{self.keyword}"), roi_left_offset, getattr(ctx, f"door_depth_m_{self.keyword}"), color=(255, 0, 255),keyword=self.keyword)
        right_match = check_side_roi_against_door(ctx.depth_image_in_meters, ctx.fx, ctx.fy, ctx.cx, ctx.cy, getattr(ctx, f"color_image_{self.keyword}"), roi_right_offset, getattr(ctx, f"door_depth_m_{self.keyword}"), color=(0, 255, 255),keyword=self.keyword)

        print(f"Left match: {left_match:.2f}, Right match: {right_match:.2f}")

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
