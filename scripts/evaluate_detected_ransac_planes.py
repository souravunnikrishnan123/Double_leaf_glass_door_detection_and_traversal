#!/usr/bin/env python3
"""Image- and metric-space quality measurements for RANSAC planes."""

import rospy
import numpy as np
import cv2  



# ---------------------------------------------------------
# Evaluate plane in image space and compute heuristics for glass-door
# ---------------------------------------------------------
def evaluate_plane_candidate(depth_m, uv_all, uv_inliers, points_inliers, roi, H, W):
    """
    Measure the image-space and physical quality of one RANSAC plane.

    The function builds the plane's pixel bounding box, measures missing depth,
    closes small gaps in its inlier mask, finds the largest connected region,
    and reports the 3D width and height of the inlier cloud.

    Args:
        depth_m:
            Full ``H x W`` depth image in meters.

        uv_all:
            Pixel coordinates for the source point cloud. Retained for API
            compatibility; the current implementation does not use it.

        uv_inliers:
            ``N x 2`` integer pixel coordinates for plane inliers.

        points_inliers:
            ``N x 3`` camera-frame coordinates corresponding to
            ``uv_inliers``.

        roi:
            Plane ROI supplied by the caller. Retained for API compatibility;
            the current implementation evaluates the inlier bounding box.

        H:
            Image height in pixels.

        W:
            Image width in pixels.

    Returns:
        Dictionary of quality statistics. An empty inlier set returns
        ``{"valid": False}``; otherwise the dictionary includes the bounding
        box, hole and connectivity fractions, physical extents, inlier count,
        and a copy of the inlier pixels.

    Notes:
        Physical width uses camera X and physical height uses camera Y.
        Morphological closing uses a fixed 5-by-5 kernel.
    """
    # Missing depth is useful evidence here: clean glass often returns fewer
    # measurements than the wall or objects visible around it.
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
    #print(hole_fraction)

    # make an image mask of the inliers for connected component analysis
    inlier_mask = np.zeros((H, W), dtype=np.uint8)
    inlier_mask[v_in, u_in] = 255  # mark inlier pixels (value 255)

    # RANSAC inliers are usually peppered with small gaps, so close only those
    # gaps before asking whether the plane is one coherent image region.
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

    # Pixel coverage alone favors nearby clutter. Metric extents make the
    # candidate comparable with the expected size of a doorway.
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
        "inlier_pixel_count": int(len(u_in)),
        "uv_inliers": uv_inliers.copy()
    }
    return stats
