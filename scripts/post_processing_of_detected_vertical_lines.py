#!/usr/bin/env python3
import rospy
import numpy as np


def extrapolate_along_line_segment(depth_image_in_meters, start_point, direction_vector, center_depth,
                                   gradient_threshold=0.1, window=5, max_steps=100):
    """
    Fully vectorized extrapolation along a line in a depth frame.
    Skips invalid depths entirely (no zero fill).
    Stops when smoothed depth deviates from center_depth beyond threshold.

    Returns: list of (x, y, depth) tuples.
    """
    


    H, W = depth_image_in_meters.shape

    # --- Define ray coordinates ---
    x0, y0 = start_point
    dx, dy = direction_vector
    t = np.arange(1, max_steps + 1, dtype=np.float32)
    xs = np.rint(x0 + dx * t).astype(int)
    ys = np.rint(y0 + dy * t).astype(int)

    # --- Filter out-of-bounds ---
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    if not inb.any():
        return []

    xs, ys = xs[inb], ys[inb]
    d = depth_image_in_meters[ys, xs]

    # --- Keep only valid depths (finite & positive) ---
    valid = np.isfinite(d) & (d > 0)
    if not np.any(valid):
        return []

    xs, ys, d = xs[valid], ys[valid], d[valid]
    n = len(d)
    if n == 0:
        return []

    # --- Compute trailing moving average using convolution ---
    # Kernel for a simple mean over `window` samples
    kernel = np.ones(window, dtype=np.float32) / float(window)

    # Perform convolution with 'same' mode so output aligns with input
    smoothed = np.convolve(d, kernel, mode='same')


    #NumPy implicitly uses zero-padding, so edge values are influenced by partial windows.
    #Finally, the result is prepended with window−1 samples using edge padding, which repeats the first smoothed value at the front.
    #This creates extra leading samples without changing the computed smoothing
    smoothed = np.pad(smoothed, (window - 1, 0), mode='edge')


    # --- Find first deviation beyond threshold ---
    deviation = np.abs(smoothed - float(center_depth))
    cond_break = deviation > float(gradient_threshold)

    if np.any(cond_break):
        brk = np.argmax(cond_break)
        xs, ys, d = xs[:brk], ys[:brk], d[:brk]

    # --- Return list of tuples ---
    return [(int(x), int(y)) for x, y, z in zip(xs, ys, d)]





def get_median_depth_along_detected_line(depth_image_in_meters, x1, y1, x2, y2, num_samples, min_num_of_valid_depths=10):
    """Samples depth values along the line segment from (x1, y1) to (x2, y2) and returns the median."""

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