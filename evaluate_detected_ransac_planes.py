import numpy as np
import cv2  



# ---------------------------------------------------------
# Evaluate plane in image space and compute heuristics for glass-door
# ---------------------------------------------------------
def evaluate_plane_candidate(depth_m, uv_all, uv_inliers, points_inliers, roi, H, W):
    """
    Given inlier pixel coordinates and points, compute:
      - bounding box in pixels
      - pixel-level hole fraction inside bbox (fraction of pixels with invalid depth)
      - inlier density (inliers / nonzero pixels in ROI)
      - largest connected inlier component area fraction vs bbox
      - physical extents of inlier points (width, height in meters)
    Returns a dict of statistics used for decision heuristics.
    """
    # prepare full image-level valid mask (non-zero depth)
    nonzero_mask_full = (depth_m > 0.0) & np.isfinite(depth_m)

    # If no inliers, return empty stats
    if len(uv_inliers) == 0:
        return {"valid": False}

    # Project inliers to integer pixel coords (u,x ; v,y)
    u_in = uv_inliers[:,0]
    v_in = uv_inliers[:,1]

    # compute bounding rectangle of inlier pixels
    min_u = int(np.clip(np.min(u_in), 0, W-1))
    max_u = int(np.clip(np.max(u_in), 0, W-1))
    min_v = int(np.clip(np.min(v_in), 0, H-1))
    max_v = int(np.clip(np.max(v_in), 0, H-1))

    # bounding box width and height in pixels
    bbox_w = max_u - min_u + 1
    bbox_h = max_v - min_v + 1

    # number of non-zero depth pixels inside bbox (these include background objects)
    bbox_nonzero = np.count_nonzero(nonzero_mask_full[min_v:max_v+1, min_u:max_u+1])

    # total bbox pixels
    bbox_area = bbox_w * bbox_h

    # hole fraction: fraction of pixels in bbox that are invalid (zeros/nans)
    hole_fraction = 1.0 - (bbox_nonzero / float(bbox_area + 1e-6))

    # make an image mask of the inliers for connected component analysis
    inlier_mask = np.zeros((H, W), dtype=np.uint8)
    inlier_mask[v_in, u_in] = 255  # mark inlier pixels (value 255)

    # morphological closing to join fragmented inlier blobs (use a small kernel)
    kernel = np.ones((5,5), dtype=np.uint8)
    closed = cv2.morphologyEx(inlier_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # find contours (connected components) and area of largest component
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area

    # fraction of bbox covered by largest connected inlier region
    largest_comp_fraction = largest_area / float(bbox_area + 1e-6)

    # physical extents (in meters) from 3D inlier points
    # width = lateral span (X axis), height = vertical span (Y axis)
    Xs = points_inliers[:,0]
    Ys = points_inliers[:,1]
    Zs = points_inliers[:,2]
    physical_width = float(np.max(Xs) - np.min(Xs))
    physical_height = float(np.max(Ys) - np.min(Ys))

    stats = {
        "valid": True,
        "bbox": (min_u, min_v, max_u, max_v),
        "bbox_w_px": bbox_w,
        "bbox_h_px": bbox_h,
        "bbox_area_px": bbox_area,
        "bbox_nonzero_px": int(bbox_nonzero),
        "hole_fraction": float(hole_fraction),
        "largest_comp_fraction": float(largest_comp_fraction),
        "physical_width_m": physical_width,
        "physical_height_m": physical_height,
        "inlier_pixel_count": int(len(u_in))
    }
    return stats