import numpy as np
import open3d as o3d
import cv2

# ---------------------------------------------------------
# Utility: backproject depth image to 3D points and uv coords
# ---------------------------------------------------------
def backproject_depth_to_points(depth_image_in_meters,
                                fx, fy, cx, cy,
                                max_depth=5.0,
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

    y0, y1, x0, x1 = 0, H, 0, W

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



def find_vertical_planes(points,
                         distance_threshold=0.05,
                         ransac_n=3,
                         num_iterations=1000,
                         vertical_tol=0.3,
                         min_inliers=10000,
                         max_planes=5):
  """
  Iteratively find up to max_planes vertical planes in the point cloud.
  Returns a list of (plane_model, inlier_indices) for vertical planes.
  """

  remaining_points = points.copy()
  remaining_indices = np.arange(points.shape[0])  # Track original indices
  found_planes = []

  for _ in range(max_planes):
    if len(remaining_points) < ransac_n:
          break

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(remaining_points)
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    if len(inliers) < min_inliers:
        break
    
    
    #to get edge points for the next iteration of RANSAC plane detection
    #It prevents dominant planes (like the floor) from "stealing" all points near boundaries
    inlier_points = remaining_points[inliers]
    a, b, c, d = plane_model
    distances = np.abs(a*inlier_points[:,0] + b*inlier_points[:,1] + c*inlier_points[:,2] + d) / np.linalg.norm([a,b,c])
    #  a tighter threshold is defined for "core" inliers. this is to get edge points for the next iteration of RANSAC plane detection
    core_threshold = 0.02  # 2cm

    # Indices of core inliers (to remove)
    core_inlier_mask = distances < core_threshold
    core_inlier_indices = np.array(inliers)[core_inlier_mask]

    # Indices of edge inliers (to keep for next iteration)
    edge_inlier_mask = ~core_inlier_mask
    edge_inlier_indices = np.array(inliers)[edge_inlier_mask]

    # Check if the plane is vertical
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    #print("Detected plane normal:", normal)
    if abs(abs(normal[2]) - 1.0) < vertical_tol:
        # Map inliers to original indices
        orig_inlier_indices = remaining_indices[inliers]  
        # Save vertical plane
        found_planes.append((plane_model, orig_inlier_indices, remaining_points[inliers]))
        print(f"Found vertical plane with {len(inliers)} inliers.")
    # Remove inliers from remaining_points for next iteration
    mask = np.ones(len(remaining_points), dtype=bool)
    mask[core_inlier_indices] = False
    remaining_points = remaining_points[mask]
    remaining_indices = remaining_indices[mask]

  return found_planes


def highlight_planes_on_image(color_image, uv, found_planes):
  """
  Overlays each detected plane's inlier pixels on the color_image in a unique color.
  - color_image: (H, W, 3) numpy array (will be modified in-place)
  - uv: (N, 2) array of pixel coordinates corresponding to the original points
  - found_planes: list of (plane_model, inlier_indices, inlier_points)
  """
  # Define a list of distinct colors (BGR for OpenCV)
  plane_colors = [
      (0, 0, 255),    # Red
      (0, 255, 0),    # Green
      (255, 0, 0),    # Blue
      (0, 255, 255),  # Yellow
      (255, 0, 255),  # Magenta
      (255, 255, 0),  # Cyan
      (128, 128, 255),# Pinkish
      (0, 128, 255),  # Orange
  ]
  for i, (_, inlier_indices, _) in enumerate(found_planes):
      color = plane_colors[i % len(plane_colors)]
      for idx in inlier_indices:
          u, v = uv[idx]
          u, v = int(u), int(v)
          """
          if 0 <= v < color_image.shape[0] and 0 <= u < color_image.shape[1]:
              cv2.circle(color_image, (u, v), 1, color, -1)  # Draw a small dot
        """


def draw_plane_outline_on_image(color_image, plane_model, inlier_points, fx, fy, cx, cy, color=(0,255,255), thickness=2, lower_pct=5, upper_pct=95, min_width_m=0.8, min_height_m=0.5):
    """
    Draws a robust outline of the detected plane as a quadrilateral on the color image.
    Uses percentiles to avoid outlier influence.
    """
    # Use percentiles to avoid outliers
    x_min = np.percentile(inlier_points[:, 0], lower_pct)
    x_max = np.percentile(inlier_points[:, 0], upper_pct)
    y_min = np.percentile(inlier_points[:, 1], lower_pct)
    y_max = np.percentile(inlier_points[:, 1], upper_pct)

    # to avoid detecting small planes including the human body
    width = x_max - x_min
    height = y_max - y_min
    if width < min_width_m or height < min_height_m:
        return []  # Plane too small to consider
    
    corners_3d = np.array([
        [x_min, y_min, 0],
        [x_max, y_min, 0],
        [x_max, y_max, 0],
        [x_min, y_max, 0]
    ])
    a, b, c, d = plane_model
    for i in range(4):
        x, y, _ = corners_3d[i]
        z = -(a*x + b*y + d) / c
        corners_3d[i, 2] = z

    corners_uv = []
    for x, y, z in corners_3d:
        if z <= 0:
            continue
        u = int(round(x * fx / z + cx))
        v = int(round(y * fy / z + cy))
        corners_uv.append((u, v))
    
    if len(corners_uv) == 4:
        pts = np.array(corners_uv, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(color_image, [pts], isClosed=True, color=color, thickness=thickness)
        return corners_uv  # Return the list of (u, v) tuples

    return []  # Return empty if not enough valid corners