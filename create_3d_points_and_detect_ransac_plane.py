import numpy as np
import open3d as o3d

# ---------------------------------------------------------
# Utility: backproject depth image to 3D points and uv coords
# ---------------------------------------------------------
def backproject_depth_to_points(depth_image_in_meters,
                                fx, fy, cx, cy,
                                max_depth=5.0,
                                roi=None,
                                subsample=1):
    """
    Convert a single-channel depth image (in meters) to an (N,3) point array
    and corresponding (N,2) pixel coordinates (u,v).
    - depth_image_in_meters: 2D numpy array (H,W) with depths in meters; invalid = 0 or np.nan
    - fx,fy,cx,cy: camera intrinsics (focal lengths, principal point)
    - max_depth: ignore points with depth > max_depth
    - roi: optional tuple (y0,y1,x0,x1) to crop image before projection
    - subsample: integer stride to reduce point count (1 = full or no subsampling)
    Returns:
      points: (N,3) numpy array (X,Y,Z) in camera frame (meters)
      uv: (N,2) integer pixel coordinates (u=x, v=y)
      mask_full: boolean 2D mask of valid depth used for additional image-space stats
    """
    # image shape
    H, W = depth_image_in_meters.shape

    # apply ROI cropping if provided (y0,y1,x0,x1) in pixel coords
    if roi is None:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = roi
        # clamp ROI to image bounds
        y0, y1 = max(0, y0), min(H, y1)
        x0, x1 = max(0, x0), min(W, x1)

    # create pixel grid for the ROI (v = rows, u = cols)
    u = np.arange(x0, x1, subsample)
    v = np.arange(y0, y1, subsample)
    uu, vv = np.meshgrid(u, v) # with subsampling value of 1, this is full grid.
    #but if the subsampling value is 2, then it will be every second pixel in both directions.

    #The function samples the depth image at the grid points defined by uu, vv.
    # esults in a 2D array of the same shape as uu and vv.
    depth_sample = depth_image_in_meters[vv, uu]  # shape (h_roi, w_roi)


    # invalid depth mask: zeros or NaNs or depth > max_depth
    valid_mask = (depth_sample > 0.0) & np.isfinite(depth_sample) & (depth_sample <= max_depth)

    # extract Z values of valid pixels
    Z = depth_sample[valid_mask].astype(np.float64)  # (N,)
    if Z.size == 0:
        # no valid depth in ROI, return empty arrays for points and uv
        return np.zeros((0,3), dtype=np.float64), np.zeros((0,2), dtype=np.int32), valid_mask

    # corresponding pixel coordinates for valid points with depth
    uu_valid = uu[valid_mask].astype(np.float64)
    vv_valid = vv[valid_mask].astype(np.float64)

    # for each valid point, backproject to 3D camera coordinates (standard pinhole model)
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    X = (uu_valid - cx) * Z / fx
    Y = (vv_valid - cy) * Z / fy
    # Z remains the same

    points = np.stack((X, Y, Z), axis=1)  # (N,3). N is the number of valid points

    # uv pixel integer coords used later for masks/contours
    uv = np.stack((uu_valid.astype(np.int32), vv_valid.astype(np.int32)), axis=1)  # (N,2)

    return points, uv, valid_mask



# ---------------------------------------------------------
# Utility: run Open3D RANSAC plane segmentation on point cloud
# RANSAC works by randomly sampling minimal sets of points (here, 3 points per sample)
# and iteratively searching for the plane that has the most inliers—points that lie within a specified distance from the plane.

# ---------------------------------------------------------
def ransac_plane_from_points(points,
                             distance_threshold=0.02,
                             ransac_n=3,
                             num_iterations=1000):
    """
    Fit a plane to a set of 3D points using Open3D's RANSAC.
    - points: (N,3) numpy array
    - distance_threshold: in meters, maximum distance from model to be considered an inlier
    - ransac_n: minimal points to estimate plane (3)
    - num_iterations: RANSAC iterations
    Returns:
      plane_model: [a,b,c,d] as floats for plane ax+by+cz+d=0
      inlier_indices: list of indices into the input points that are inliers
    """
    # convert numpy points to Open3D point cloud
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # segment_plane returns plane_model and list of inlier indices
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    return plane_model, inliers
