import numpy as np
import open3d as o3d
import cv2

# ---------------------------------------------------------
# Utility: backproject depth image to 3D points and uv coords
# ---------------------------------------------------------
def backproject_depth_to_points(
    depth_image_in_meters,
    fx, fy, cx, cy,
    max_depth=5.0,
    subsample=1,
    roi_polygon=None
):
    """
    Convert a depth image to 3D points and pixel coordinates, optionally within a polygonal ROI.
    If roi_polygon is None, uses the whole image.
    roi_polygon should be a Nx2 array/list of (x, y) pixel coordinates.
    Returns:
        points: (N, 3) array of 3D coordinates
        uv: (N, 2) array of image pixel coordinates
        mask: boolean 2D mask of valid pixels inside ROI
    """
    H, W = depth_image_in_meters.shape
    mask = np.ones((H, W), dtype=np.uint8) * 255

    if roi_polygon is not None:
        mask[:] = 0
        cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)

    # Find valid pixels inside ROI and with valid depth
    valid_mask = (
        (mask == 255)
        & (depth_image_in_meters > 0)
        & np.isfinite(depth_image_in_meters)
        & (depth_image_in_meters < max_depth)
    )

    ys, xs = np.where(valid_mask)
    if subsample > 1:
        ys = ys[::subsample]
        xs = xs[::subsample]

    zs = depth_image_in_meters[ys, xs]
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)

    Xs = (xs_f - cx) * zs / fx
    Ys = (ys_f - cy) * zs / fy
    points = np.stack([Xs, Ys, zs], axis=-1)
    uv = np.stack([xs, ys], axis=-1)

    return points, uv, valid_mask

# ---------------------------------------------------------
# Utility: run Open3D RANSAC plane segmentation on point cloud
# RANSAC works by randomly sampling minimal sets of points (here, 3 points per sample)
# and iteratively searching for the plane that has the most inliers—points that lie within a specified distance from the plane.

# ---------------------------------------------------------
def ransac_plane_from_points(points,
                             distance_threshold=0.05,
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
    if points is None or len(points) == 0 or points.shape[1] != 3:
        return None, []
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
                         horizontal_tol = 0.3,
                         min_inliers=10000,
                         max_planes=4):
  """
  Iteratively find up to max_planes vertical planes in the point cloud.
  Returns a list of (plane_model, inlier_indices) for vertical planes.
  """

  remaining_points = points.copy()
  remaining_indices = np.arange(points.shape[0])  # Track original indices
  found_vertical_planes = []
  found_horizontal_planes = []
  all_planes = []

  for _ in range(max_planes):
    if len(remaining_points) < ransac_n:
          break

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(remaining_points)
    plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    
    all_planes.append((plane_model, remaining_indices[inliers], remaining_points[inliers]))
    
    if len(inliers) < min_inliers:
        break
    
    
    #to get edge points for the next iteration of RANSAC plane detection
    #It prevents dominant planes (like the floor) from "stealing" all points near boundaries
    inlier_points = remaining_points[inliers]
    a, b, c, d = plane_model
    distances = np.abs(a*inlier_points[:,0] + b*inlier_points[:,1] + c*inlier_points[:,2] + d) / np.linalg.norm([a,b,c])
    #  a tighter threshold is defined for "core" inliers. this is to get edge points for the next iteration of RANSAC plane detection
    core_threshold = 0.02     # 2cm

    # Indices of core inliers (to remove)
    core_inlier_mask = distances < core_threshold
    core_inlier_indices = np.array(inliers)[core_inlier_mask]

    # Indices of edge inliers (to keep for next iteration)
    edge_inlier_mask = ~core_inlier_mask
    edge_inlier_indices = np.array(inliers)[edge_inlier_mask]

    mask = np.ones(len(remaining_points), dtype=bool)
    # Remove inliers from remaining_points for next iteration. we are only removig the core inliers to get edge points for the next iteration of RANSAC plane detection.
    #because, for slim door planes, floor is taking lot of points near the door boundary
    
    mask[core_inlier_indices] = False  # Remove core inliers


    # Check if the plane is vertical
    normal = np.array(plane_model[:3])
    normal = normal / np.linalg.norm(normal)
    #print("Detected plane normal:", normal)
    if abs(abs(normal[2]) - 1.0) < vertical_tol:
        # Map inliers to original indices
        orig_inlier_indices = remaining_indices[inliers]  
        # Save vertical plane
        found_vertical_planes.append((plane_model, orig_inlier_indices, remaining_points[inliers]))
        print(f"Found vertical plane with {len(inliers)} inliers.")
        mask[edge_inlier_indices] = False  # remove the edge inliers for next iteration
        #here because we detected a potential door plane, we remove all inliers (core + edge) to get to avoid getting the same plane again
    
    elif abs(abs(normal[1]) - 1.0) < horizontal_tol:
        # Map inliers to original indices
        orig_inlier_indices = remaining_indices[inliers]  
        found_horizontal_planes.append((plane_model, core_inlier_indices, remaining_points[inliers]))
        print(f"Found horizontal plane with {len(inliers)} inliers.")
    

    remaining_points = remaining_points[mask]
    remaining_indices = remaining_indices[mask]

  return found_vertical_planes, found_horizontal_planes, all_planes


def highlight_planes_on_image(color_image, uv, found_vertical_planes):
  """
  Overlays each detected plane's inlier pixels on the color_image in a unique color.
  - color_image: (H, W, 3) numpy array (will be modified in-place)
  - uv: (N, 2) array of pixel coordinates corresponding to the original points
  - found_vertical_planes: list of (plane_model, inlier_indices, inlier_points)
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
  for i, (_, inlier_indices, _) in enumerate(found_vertical_planes):
      color = plane_colors[i % len(plane_colors)]
      for idx in inlier_indices:
          u, v = uv[idx]
          u, v = int(u), int(v)
          
          if 0 <= v < color_image.shape[0] and 0 <= u < color_image.shape[1]:
              cv2.circle(color_image, (u, v), 1, color, -1)  # Draw a small dot
            


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
    print(f"width={width:.2f}, height={height:.2f}")
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