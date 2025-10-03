import cv2
import numpy as np
from get_z_depth import get_z_depth


def show_stacked_visualization(color_image, depth_image, MIN_DEPTH, MAX_DEPTH, edges, depth_frame, window_name="Color | Depth | Edges+ Lines", screen_width=1920, screen_height=1080):
    """
    Stack color, depth, and edge images horizontally, resize to fit screen, display in a window,
    attach mouse callback for depth, and handle ESC key to exit.
    Returns True if ESC is pressed, else False.
    """
    depth_colormap = depth_to_colormap(depth_image, MIN_DEPTH, MAX_DEPTH)

    target_height, target_width = color_image.shape[:2]
    if len(edges.shape) == 2 or edges.shape[2] == 1:
        edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        edges_vis = edges
    edges_resized = cv2.resize(edges_vis, (target_width, target_height))
    
    stacked = np.hstack((color_image, depth_colormap, edges_resized))

    scale_w = screen_width / stacked.shape[1]
    scale_h = screen_height / stacked.shape[0]
    scale_factor = min(scale_w, scale_h)

    stacked_resized = cv2.resize(stacked, None, fx=scale_factor, fy=scale_factor)
    cv2.imshow(window_name, stacked_resized)

    cv2.setMouseCallback(window_name, click_event, param=(depth_frame, scale_factor))

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        return True
    return False


# -------------------------------
# Convert raw depth image to color for visualization
# -------------------------------
def depth_to_colormap(depth_image, MIN_DEPTH, MAX_DEPTH):
    # Clip depth image to desired range
    depth_scaled = np.clip(depth_image, MIN_DEPTH*1000, MAX_DEPTH*1000)
    # Convert depth to 8-bit for color mapping
    depth_scaled = cv2.convertScaleAbs(depth_scaled, alpha=0.03)
    # Apply colormap (Jet: Blue → Red gradient)
    return cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)

# -------------------------------
# Mouse click event for checking depth at pixel
# -------------------------------
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame, scale_factor = param  # unpack parameters
        # Map coordinates back to original resolution
        orig_x = int(x / scale_factor)
        orig_y = int(y / scale_factor)

        depth = get_z_depth(depth_frame, orig_x, orig_y)
        print(f"Clicked at ({orig_x}, {orig_y}) → Depth: {depth:.2f} m")
