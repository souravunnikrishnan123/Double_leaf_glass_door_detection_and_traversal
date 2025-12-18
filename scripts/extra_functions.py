import cv2
import numpy as np
import open3d as o3d

def extract_roi_corners(roi_polygon):
    """
    Extract the 4 corner points of a parallelogram-shaped ROI.
    Works even if roi_polygon has many intermediate points.
    """
    roi = np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
    if roi.shape[0] < 4:
        return roi.astype(np.int32)

    # Step 1: Get convex hull (robust to noisy edges)
    hull = cv2.convexHull(roi)
    hull = hull.reshape(-1, 2)

    # Step 2: Approximate hull to 4-point polygon
    peri = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * peri, True).reshape(-1, 2)

    # Fallback: if we didn’t get 4 corners, pick extreme ones
    if approx.shape[0] != 4:
        x, y = hull[:, 0], hull[:, 1]
        pts = np.array([
            [np.min(x), np.min(y)],
            [np.max(x), np.min(y)],
            [np.max(x), np.max(y)],
            [np.min(x), np.max(y)],
        ], dtype=np.float32)
    else:
        pts = approx

    # Step 3: Order points: top-left, top-right, bottom-right, bottom-left
    # sort by y (top to bottom)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top2, bottom2 = y_sorted[:2], y_sorted[2:]

    # left/right among top and bottom
    top_left = top2[np.argmin(top2[:, 0])]
    top_right = top2[np.argmax(top2[:, 0])]
    bottom_left = bottom2[np.argmin(bottom2[:, 0])]
    bottom_right = bottom2[np.argmax(bottom2[:, 0])]

    ordered = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    return ordered





def extend_roi_polygon_to_full_height(roi_polygon, image_height):
    """
    Robustly extend a 4-point ROI (parallelogram) to full image height.
    - Handles arbitrary ordering of the 4 points.
    - Preserves the left/right slant by using x of the top pair for y=0
      and x of the bottom pair for y=image_height-1.
    """
    roi = np.asarray(roi_polygon).reshape(-1, 2)

    # sort points by y (top -> bottom)
    sorted_idx = np.argsort(roi[:, 1])
    top2 = roi[sorted_idx[:2]] # selects the two points with the smallest y values
    bot2 = roi[sorted_idx[2:]] #selects the two points with the largest y values

    # determine left/right for top and bottom pairs by x
    top_left_x = int(round(np.min(top2[:, 0])))
    top_right_x = int(round(np.max(top2[:, 0])))
    bot_left_x = int(round(np.min(bot2[:, 0])))
    bot_right_x = int(round(np.max(bot2[:, 0])))

    extended = np.array([
        [top_left_x, 0],
        [top_right_x, 0],
        [bot_right_x, image_height - 1],
        [bot_left_x, image_height - 1]
    ], dtype=np.int32)

    return extended



def build_side_rect_roi(line_points, side="left", roi_width=40, margin=10, image_height=None):
    """
    Build rectangular ROI offset from a vertical line.
    - margin: how many pixels to leave empty next to the line
    - roi_width: width of ROI strip
    """
    xs = [p[0] for p in line_points]
    ys = [p[1] for p in line_points]
    x_min, x_max = int(min(xs)), int(max(xs))
    #y_min = int(min(ys))
    #y_min = 0  # start from top of image
    # Use image_height if provided, else use max y from line
    #y_max = image_height - 1 if image_height is not None else int(max(ys))
    y_min, y_max = int(min(ys)), int(max(ys))


    if side == "left":
        x1 = x_min - margin - roi_width
        x2 = x_min - margin
    else:  # right
        x1 = x_max + margin
        x2 = x_max + margin + roi_width

    roi_polygon = np.array([
        [x1, y_min],
        [x2, y_min],
        [x2, y_max],
        [x1, y_max]
    ], dtype=np.int32)

    return roi_polygon



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
            



def find_planes(points,
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

