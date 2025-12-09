import numpy as np
from typing import Optional, Tuple
import cv2


class Preprocessor:

    def resize_by_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        if scale != 1.0:
            return cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            return image
        
    def gaussian_blur_filter(self, image: np.ndarray, blur_cfg: dict) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.equalizeHist(gray)
        k = int(blur_cfg.get("ksize", 3))
        sigma = float(blur_cfg.get("sigma", 0.8))
        filtered_grey_image = cv2.GaussianBlur(contrast, (k, k), sigma)
        return filtered_grey_image

    def bilateral_filter(self, image: np.ndarray, bilateral_cfg: dict) -> np.ndarray:
    
        # Smooth Z-depth (bilateral preserves edges better than Gaussian)
        d = int(bilateral_cfg.get("d", 5))
        sigma_color = float(bilateral_cfg.get("sigma_color", 50))
        sigma_space = float(bilateral_cfg.get("sigma_space", 75))

        filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered_image
    
    def restore_size(self, image: np.ndarray, target_w: int, target_h: int, scale: float) -> np.ndarray:
        if scale != 1.0:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:   
            return image



class LineFilter:
    def angle_filter(self, lines: np.ndarray, angle_threshold: float) -> np.ndarray:
        if lines is None or len(lines) == 0:
            return lines
        x1_np = lines[:, 0, 0]
        y1_np = lines[:, 0, 1]
        x2_np = lines[:, 0, 2]
        y2_np = lines[:, 0, 3]
        angles = np.arctan2(y2_np - y1_np, x2_np - x1_np)
        vertical_mask = (np.abs(np.cos(angles)) < angle_threshold)
        return lines[vertical_mask]

    def is_depth_valid(self, center_depth: Optional[float], depth_range: Tuple[float, float]) -> bool:
        if center_depth is None:
            return False
        return depth_range[0] <= center_depth <= depth_range[1]


class EdgeDetector:
    def canny_edge_detection(self, filtered: np.ndarray, canny_cfg: dict) -> np.ndarray:
        median_val = np.median(filtered[::4, ::4])
        lf = float(canny_cfg.get("lower_factor", 0.7))
        uf = float(canny_cfg.get("upper_factor", 2.0))
        lower = int(max(0, lf * median_val))
        upper = int(min(255, uf * median_val))
        edges = cv2.Canny(filtered, lower, upper)
        return edges
    
    def sobel_edge_detection(self, filtered: np.ndarray, sobel_cfg: dict) -> np.ndarray:
                # Gradient along X (detect vertical edges in depth)

        ksize = int(sobel_cfg.get("ksize", 3))

        depth_grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=ksize)
        depth_grad_x = np.abs(depth_grad_x)
        return depth_grad_x
    
    def adaptive_threshold(self, depth_grad_x: np.ndarray, valid_grad_vals: np.ndarray, adaptive_cfg: dict) -> np.ndarray:

        # Adaptive threshold: mean + k*std of valid gradients
        
        k_factor = float(adaptive_cfg.get("k_factor", 2.0))
        fallback = float(adaptive_cfg.get("fallback_threshold", 0.1))

        if len(valid_grad_vals) > 0:
            mean_val = np.mean(valid_grad_vals)
            std_val = np.std(valid_grad_vals)
            thresh_val = mean_val + k_factor * std_val
        else:
            thresh_val = fallback  # fallback threshold

        _, depth_edges = cv2.threshold(depth_grad_x, thresh_val, 255, cv2.THRESH_BINARY)
        depth_edges = depth_edges.astype(np.uint8)
        return depth_edges



class HoughPLineDetector:
        # Hough Line Transform to detect lines
    # Detect lines using Probabilistic Hough Transform
    # Parameters:
    #  - 1: pixel resolution of the Hough grid
    #  - np.pi / 180: angle resolution in radians (1 degree)
    #  - threshold=100: minimum number of intersections to detect a line
    #  - minLineLength=100: minimum length of line in pixels to be considered
    #  - maxLineGap=10: maximum allowed gap between line segments to link them
    def detect(self, edges: np.ndarray, hough_cfg: dict, scale: float) -> Optional[np.ndarray]:
        base_threshold = int(hough_cfg.get("threshold", 100))
        base_min_len = int(hough_cfg.get("min_line_length", 100))
        base_max_gap = int(hough_cfg.get("max_line_gap", 20))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=int(base_threshold * scale),
            minLineLength=int(base_min_len * scale),
            maxLineGap=int(base_max_gap * scale),
        )
        if lines is not None and len(lines) > 0:
            # bring back to original scale
            lines[..., :4] = np.rint(lines[..., :4] / scale).astype(np.int32)
        return lines