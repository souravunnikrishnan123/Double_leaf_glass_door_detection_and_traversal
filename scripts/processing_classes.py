"""Reusable image, line, edge, and depth-backprojection operations."""

import numpy as np
from typing import Optional, Tuple
import cv2
import math

class Preprocessor:
    """
    Prepare color and depth images for edge detection.

    This stateless helper centralizes image resizing and the branch-specific
    smoothing operations used before Canny or Sobel processing.
    """

    def resize_by_scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """
        Resize an image by a uniform scale factor.

        Args:
            image:
                Source NumPy image.

            scale:
                Horizontal and vertical resize factor.

        Returns:
            Resized image. When ``scale`` equals ``1.0``, the original array is
            returned without copying.
        """
        if scale != 1.0:
            return cv2.resize(
                image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        else:
            return image
        
    def gaussian_blur_filter(self, image: np.ndarray, blur_cfg: dict) -> np.ndarray:
        """
        Convert BGR to grayscale, equalize contrast, and apply Gaussian blur.

        Args:
            image:
                BGR uint8 image.

            blur_cfg:
                Mapping containing ``ksize`` and ``sigma``. Missing values
                default to 3 and 0.8.

        Returns:
            Smoothed single-channel image suitable for Canny detection.

        Raises:
            cv2.error:
                If the image format is unsupported or the configured kernel is
                invalid.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = cv2.equalizeHist(gray)
        k = int(blur_cfg.get("ksize", 3))
        sigma = float(blur_cfg.get("sigma", 0.8))
        filtered_grey_image = cv2.GaussianBlur(contrast, (k, k), sigma)
        return filtered_grey_image

    def bilateral_filter(self, image: np.ndarray, bilateral_cfg: dict) -> np.ndarray:
        """
        Smooth a depth image while preserving strong discontinuities.

        Args:
            image:
                Single-channel metric depth image.

            bilateral_cfg:
                Mapping containing OpenCV bilateral-filter parameters ``d``,
                ``sigma_color``, and ``sigma_space``.

        Returns:
            Bilaterally filtered depth image.
        """
    
        # Smooth Z-depth (bilateral preserves edges better than Gaussian)
        d = int(bilateral_cfg.get("d", 5))
        sigma_color = float(bilateral_cfg.get("sigma_color", 50))
        sigma_space = float(bilateral_cfg.get("sigma_space", 75))

        filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return filtered_image
    
    def restore_size(self, image: np.ndarray, target_w: int, target_h: int, scale: float) -> np.ndarray:
        """
        Restore a processed image to the source frame dimensions.

        Args:
            image:
                Image processed at a reduced scale.

            target_w:
                Required output width in pixels.

            target_h:
                Required output height in pixels.

            scale:
                Scale used during preprocessing.

        Returns:
            Image resized to ``(target_w, target_h)`` when ``scale`` is not
            one; otherwise the original array.
        """
        if scale != 1.0:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:   
            return image



class LineFilter:
    """
    Filter Hough segments by orientation and aligned depth evidence.

    The methods accept the ``(N, 1, 4)`` layout returned by
    ``cv2.HoughLinesP`` and use aligned metric depth to reject lines that do
    not belong to the confirmed door-distance band.
    """

    def angle_filter(self, lines: np.ndarray, angle_threshold: float) -> np.ndarray:
        """
        Keep segments whose direction is sufficiently close to vertical.

        Args:
            lines:
                Hough segments shaped ``(N, 1, 4)``.

            angle_threshold:
                Maximum absolute horizontal component, expressed as
                ``abs(cos(angle))``.

        Returns:
            Filtered Hough array. ``None`` and empty inputs are returned
            unchanged.
        """
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
        """
        Check that a depth estimate lies in an inclusive interval.

        Args:
            center_depth:
                Metric depth estimate, or ``None`` when sampling failed.

            depth_range:
                Two-element ``(minimum, maximum)`` interval in meters.

        Returns:
            ``True`` only when a depth exists inside the interval.
        """
        if center_depth is None:
            return False
        return depth_range[0] <= center_depth <= depth_range[1]


    def get_median_depth_along_line(self, depth_image_in_meters, line, min_num_of_valid_depths):
        """
        Estimate line depth from samples taken directly along the segment.

        Args:
            depth_image_in_meters:
                ``H x W`` aligned metric depth image.

            line:
                Hough segment in ``[[x1, y1, x2, y2]]`` layout.

            min_num_of_valid_depths:
                Minimum number of finite positive samples required.

        Returns:
            Median depth in meters, or ``None`` when too few valid samples are
            available.

        Notes:
            The number of samples is the segment's Euclidean pixel length.
        """
        x1, y1, x2, y2 = line[0]
        pixel_length = math.hypot(x2 - x1, y2 - y1)
        num_samples = int(pixel_length)
        H, W = depth_image_in_meters.shape


        t = np.linspace(0, 1, num_samples, dtype=np.float32)
        xs = np.rint(x1 + (x2 - x1) * t).astype(int)
        ys = np.rint(y1 + (y2 - y1) * t).astype(int)

        # --- Filter out-of-bounds ---
        inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        if not inb.any():
            return None

        xs, ys = xs[inb], ys[inb]
        d = depth_image_in_meters[ys, xs]

        # --- Keep only valid depths (finite & positive) ---
        valid = np.isfinite(d) & (d > 0)

        d_valid = d[valid]

        return float(np.median(d_valid)) if len(d_valid) >= min_num_of_valid_depths else None
    

    def get_median_depth_by_roi_around(self,depth_image_in_meters, line , roi_width):
        """
        Estimate line depth from narrow strips on both sides of the segment.

        Args:
            depth_image_in_meters:
                ``H x W`` aligned metric depth image.

            line:
                Primarily vertical Hough segment in ``[[x1, y1, x2, y2]]``
                layout.

            roi_width:
                Width in pixels sampled on each side of the line center.

        Returns:
            The nearer nonzero side median in meters. If only one side has
            valid depth, that median is returned; if neither side is valid,
            returns ``None``.

        Notes:
            A nonvertical input is rejected because the strips are built as
            horizontal offsets from a vertical line.
        """
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > abs(y1 - y2):
            print("Warning: Line is not primarily vertical. This method assumes vertical lines.")
            return None
        
        H, W = depth_image_in_meters.shape

        y_start, y_end = sorted((y1, y2))
        y_start= max(0, y_start)
        y_end = min(H, y_end)

        if y_end <= y_start:
            return None
        
        ys = np.arange(y_start, y_end)

        x_center = int((x1 + x2) / 2)

        x_left_roi_start = max(0, x_center - roi_width)
        x_left_roi_end = min(W, x_center)

        x_right_roi_start = max(0, x_center + 1)
        x_right_roi_end = min(W, x_center + roi_width + 1)

        # List to store valid Z-depths for each ROI
        z_depths1 = depth_image_in_meters[ys, x_left_roi_start:x_left_roi_end] if x_left_roi_end > x_left_roi_start else np.empty((0, 0), dtype=np.float32)
        z_depths2 = depth_image_in_meters[ys, x_right_roi_start:x_right_roi_end] if x_right_roi_end > x_right_roi_start else np.empty((0, 0), dtype=np.float32)   

        z_depths1 = z_depths1.ravel() #to convert to 1D array
        z_depths2 = z_depths2.ravel() #to convert to 1D array

        z_depths1 = z_depths1[np.isfinite(z_depths1) & (z_depths1 > 0)]
        z_depths2 = z_depths2[np.isfinite(z_depths2) & (z_depths2 > 0)]


        # Compute medians
        med_z_depth1 = np.median(z_depths1) if len(z_depths1) > 0 else 0
        med_z_depth2 = np.median(z_depths2) if len(z_depths2) > 0 else 0

        # Apply your filtering logic
        if med_z_depth1 == 0 and med_z_depth2 == 0:
            return None
        elif med_z_depth1 == 0:
            return med_z_depth2
        elif med_z_depth2 == 0:
            return med_z_depth1
        else:
            return min(med_z_depth1, med_z_depth2)




class EdgeDetector:
    """
    Produce edge responses from color intensity or metric depth.

    The color branch uses median-adaptive Canny thresholds. The depth branch
    computes the absolute x-gradient and converts it to a binary edge mask with
    a statistical threshold.
    """

    def canny_edge_detection(self, filtered: np.ndarray, canny_cfg: dict) -> np.ndarray:
        """
        Run Canny with thresholds derived from the image's sampled median.

        Args:
            filtered:
                Smoothed single-channel uint8 image.

            canny_cfg:
                Mapping containing ``lower_factor`` and ``upper_factor``.

        Returns:
            Binary uint8 Canny edge image.
        """
        median_val = np.median(filtered[::4, ::4])
        lf = float(canny_cfg.get("lower_factor", 0.7))
        uf = float(canny_cfg.get("upper_factor", 2.0))
        lower = int(max(0, lf * median_val))
        upper = int(min(255, uf * median_val))
        edges = cv2.Canny(filtered, lower, upper)
        return edges
    
    def sobel_edge_detection(self, filtered: np.ndarray, sobel_cfg: dict) -> np.ndarray:
        """
        Compute the absolute horizontal Sobel gradient.

        Args:
            filtered:
                Smoothed single-channel depth image.

            sobel_cfg:
                Mapping containing odd Sobel kernel size ``ksize``.

        Returns:
            Float32 gradient-magnitude image. Horizontal differentiation makes
            vertical depth boundaries respond strongly.
        """
                # Gradient along X (detect vertical edges in depth)

        ksize = int(sobel_cfg.get("ksize", 3))

        depth_grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=ksize)
        depth_grad_x = np.abs(depth_grad_x)
        return depth_grad_x
    
    def adaptive_threshold(self, depth_grad_x: np.ndarray, valid_grad_vals: np.ndarray, adaptive_cfg: dict) -> np.ndarray:
        """
        Convert depth gradients to a binary edge mask.

        Args:
            depth_grad_x:
                Absolute Sobel gradient image.

            valid_grad_vals:
                Gradient samples from the accepted door-depth band.

            adaptive_cfg:
                Mapping with ``k_factor`` and ``fallback_threshold``.

        Returns:
            Binary uint8 edge mask.

        Notes:
            The normal threshold is ``mean + k_factor * standard_deviation``.
            The fallback is used when the valid sample set is empty.
        """

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
    """
    Detect probabilistic Hough segments at a configurable image scale.

    Configuration values are specified for the original image and scaled before
    calling OpenCV. Detected endpoints are then mapped back to original-image
    coordinates.
    """

        # Hough Line Transform to detect lines
    # Detect lines using Probabilistic Hough Transform
    # Parameters:
    #  - 1: pixel resolution of the Hough grid
    #  - np.pi / 180: angle resolution in radians (1 degree)
    #  - threshold=100: minimum number of intersections to detect a line
    #  - minLineLength=100: minimum length of line in pixels to be considered
    #  - maxLineGap=10: maximum allowed gap between line segments to link them
    def detect(self, edges: np.ndarray, hough_cfg: dict, scale: float) -> Optional[np.ndarray]:
        """
        Run the probabilistic Hough transform.

        Args:
            edges:
                Binary edge image, possibly processed at reduced resolution.

            hough_cfg:
                Mapping containing ``threshold``, ``min_line_length``, and
                ``max_line_gap`` in original-image pixels.

            scale:
                Processing scale used to create ``edges``.

        Returns:
            Hough segments shaped ``(N, 1, 4)`` in original-image coordinates,
            or ``None`` when no segment is detected.

        Raises:
            ZeroDivisionError:
                If lines are detected while ``scale`` is zero.
        """
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

# ---------------------------------------------------------
# Utility: backproject depth image to 3D points and uv coords
# ---------------------------------------------------------
def backproject_depth_to_points(
    depth_image_in_meters,
    fx, fy, cx, cy,
    max_depth,min_depth,
    subsample,
    roi_polygon=None
):
    """
    Backproject valid depth pixels into camera-frame 3D points.

    Pinhole projection is applied to a configurable depth interval, optionally
    restricted by an image-space polygon. Pixel coordinates remain paired with
    their resulting points.

    Args:
        depth_image_in_meters:
            ``H x W`` metric depth image.

        fx:
            Horizontal focal length in pixels.

        fy:
            Vertical focal length in pixels.

        cx:
            Horizontal principal point in pixels.

        cy:
            Vertical principal point in pixels.

        max_depth:
            Exclusive far-depth limit in meters.

        min_depth:
            Exclusive near-depth limit in meters.

        subsample:
            Keep every ``subsample``-th valid pixel after mask extraction.

        roi_polygon:
            Optional ``N x 2`` polygon of ``(x, y)`` image coordinates. The
            entire image is used when omitted.

    Returns:
        Tuple ``(points, uv, valid_mask)`` where ``points`` is an ``N x 3``
        array of camera-frame ``(X, Y, Z)``, ``uv`` is the corresponding
        ``N x 2`` pixel array, and ``valid_mask`` is the full-resolution
        boolean selection mask.

    Notes:
        Camera coordinates follow the optical convention: X right, Y down,
        and Z forward.
    """
    H, W = depth_image_in_meters.shape
    mask = np.ones((H, W), dtype=np.uint8) * 255

    if roi_polygon is not None:
        mask[:] = 0
        cv2.fillPoly(mask, [np.array(roi_polygon, dtype=np.int32)], 255)

    # Find valid pixels inside ROI and with valid depth
    valid_mask = (
        (mask == 255)
        & (depth_image_in_meters > min_depth)
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



