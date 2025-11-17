import pyrealsense2 as rs   # RealSense SDK for Python
import numpy as np
import cv2


def get_strip_avg_z(depth_image_in_meters, line_points, side="left", roi_width=20, min_depth=1.7):
    """
    Calculate the average Z coordinate in a strip (ROI) parallel to a given line.
    """
    # Take only x, y (ignore any extra fields like depth)
    pts = np.array([(p[0], p[1]) for p in line_points], dtype=np.int32)

    # Compute direction of the line
    dx = pts[-1][0] - pts[0][0]
    dy = pts[-1][1] - pts[0][1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return None, None

    dx /= length
    dy /= length

    if side == "left":
        offset_vec = np.array([-dy, dx])
    else:
        offset_vec = np.array([dy, -dx])

    offset_vec = offset_vec / np.linalg.norm(offset_vec) * roi_width

    # Shift points
    pts_offset = pts + offset_vec
    roi_polygon = np.vstack((pts, pts_offset[::-1]))

    # Create mask for ROI
    mask = np.zeros(depth_image_in_meters.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 255)

    roi_depth = depth_image_in_meters[mask == 255]

    # Vectorized filtering
    valid_mask = (roi_depth >= min_depth) & (roi_depth != 0) & ~np.isnan(roi_depth)
    valid_depths = roi_depth[valid_mask]

    avg_z = float(np.mean(valid_depths)) if len(valid_depths) > 10 else None
    return avg_z, roi_polygon




def extrapolate_line_to_y_range(line_points, y_start, y_end, step=1):
    """
    Fit a line to line_points and sample all (x, y) points along the extrapolated line
    from y_start to y_end (inclusive), with given step.
    Returns a list of (x, y) points.
    """
    pts = np.array([(p[0], p[1]) for p in line_points])
    # Fit line: y = m*x + c, but we want x as a function of y for vertical lines
    # Use np.polyfit with degree 1
    if len(pts) < 2:
        return  []  # Not enough points

    # Fit x = a*y + b (since vertical lines)
    fit = np.polyfit(pts[:,1], pts[:,0], 1)
    #fitting a straight line (least squares regression) to all points in line_points.
    a, b = fit

    ys = np.arange(y_start, y_end + 1, step)
    sampled_points = [(int(a * y + b), int(y)) for y in ys]
    return sampled_points


def process_filtered_lines(filtered_lines, depth_image_in_meters, color_image):
    """
    For each pair of lines, create ROIs:
    - Left ROI: to the left of the left line in the pair
    - Right ROI: to the right of the right line in the pair
    Compute average Z depth inside each ROI and draw them.

    Args:
        filtered_lines: list of tuples [(left_line_points, right_line_points), ...]
        depth_frame: RealSense depth frame
        color_image: BGR image for visualization

    Returns:
        (avg_z_left_roi_list, avg_z_right_roi_list)
    """

    correction_factor=1.1


    avg_z_left_roi_list = []
    avg_z_right_roi_list = []
    filtered_pairs = []
    mean_z_depth_along_frame_lines_list = []
    valid_roi_polygon_left_list = []
    valid_roi_polygon_right_list = []

    for i, (left_line_points, right_line_points, left_depth, right_depth) in enumerate(filtered_lines):
        """
        
        y_bottom = max(np.max([pt[1] for pt in left_line_points]), 
                       np.max([pt[1] for pt in right_line_points]),
                       np.max([pt[1] for pt in detected_plane]) if (detected_plane is not None and len(detected_plane) > 0) else depth_image_in_meters.shape[0] - 1
                        )
        """
        y_bottom = depth_image_in_meters.shape[0] - 1
        y_top = 0

        extrapolated_left_line_points = extrapolate_line_to_y_range(left_line_points, y_top, y_bottom)
        extrapolated_right_line_points = extrapolate_line_to_y_range(right_line_points, y_top, y_bottom)


        # Compute average Z for left ROI
        avg_z_left_roi, roi_polygon_left = get_strip_avg_z(
            depth_image_in_meters, extrapolated_left_line_points, side="left", roi_width=240, min_depth=1.7
        )
        

        # Compute average Z for right ROI
        avg_z_right_roi, roi_polygon_right = get_strip_avg_z(
            depth_image_in_meters, extrapolated_right_line_points, side="right", roi_width=240, min_depth=1.7
        ) # minimum depth used to ignore the depth info from human who is between the door and robodog
        

        # Calculate mean Z along the left line. we are not using extrapolated_left_line_points because 
        # more reliable depth info comes from the original line points. extrapolated points shall be used only for defining the ROIs
        # to get the maximum area where we expect lot of zero and non zero depth values
        #z_left_line = [pt[2] for pt in left_line_points if pt[2] > 0 and not np.isnan(pt[2])]
        #mean_z_left_line = float(np.mean(z_left_line)) if z_left_line else None
        mean_z_left_line = left_depth

        # Calculate mean Z along the original right line points
        #z_right_line = [pt[2] for pt in right_line_points if pt[2] > 0 and not np.isnan(pt[2])]
        #mean_z_right_line = float(np.mean(z_right_line)) if z_right_line else None
        mean_z_right_line = right_depth
        if mean_z_left_line is not None and mean_z_right_line is not None:
            mean_z_depth_along_frame_lines = 0.5 * (mean_z_left_line + mean_z_right_line)
        else:
            mean_z_depth_along_frame_lines = None

                # Filter pairs based on your criteria
        if (mean_z_left_line is not None and avg_z_left_roi is not None and mean_z_left_line*correction_factor < avg_z_left_roi) and (mean_z_right_line is not None and avg_z_right_roi is not None and mean_z_right_line*correction_factor < avg_z_right_roi):
            avg_z_left_roi_list.append(avg_z_left_roi)
            avg_z_right_roi_list.append(avg_z_right_roi)
            filtered_pairs.append((extrapolated_left_line_points, extrapolated_right_line_points, left_depth, right_depth))
            valid_roi_polygon_left_list.append(roi_polygon_left)
            valid_roi_polygon_right_list.append(roi_polygon_right)
            mean_z_depth_along_frame_lines_list.append(mean_z_depth_along_frame_lines)


            # Draw ROIs if available
            if roi_polygon_left is not None:
                cv2.polylines(color_image, [roi_polygon_left.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
            if roi_polygon_right is not None:
                cv2.polylines(color_image, [roi_polygon_right.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
            
            # Annotate average Z values
            cv2.putText(color_image, f"Left ROI {i+1} Z: {avg_z_left_roi:.2f} m", (30, 30 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(color_image, f"Right ROI {i+1} Z: {avg_z_right_roi:.2f} m", (30, 50 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(color_image, f"Left Line {i+1} Z: {mean_z_left_line:.2f} m", (300, 30 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(color_image, f"Right Line {i+1} Z: {mean_z_right_line:.2f} m", (300, 50 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




    return valid_roi_polygon_left_list, valid_roi_polygon_right_list, filtered_pairs, mean_z_depth_along_frame_lines_list

