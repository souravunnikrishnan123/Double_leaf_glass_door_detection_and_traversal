#!/usr/bin/env python3
import rospy
import numpy as np
import cv2


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





def cluster_and_merge_lines(color_image,lines, depth_of_valid_lines, x_thresh, min_merged_line_length ):
    """
    Cluster vertical lines by proximity in x-coordinate and merge into one line per cluster.
    Args:
        lines: list of (x1, y1, x2, y2)
        x_thresh: max horizontal distance (in pixels) to group lines
    Returns:
        merged_lines: list of merged (x1, y1, x2, y2)
    """
    if not lines:
        return [], []

        # Sort lines and depths together by average x
    lines_with_depths = sorted(
        zip(lines, depth_of_valid_lines),
        key=lambda ld: (ld[0][0] + ld[0][2]) / 2
    )
    clusters = []
    depth_of_clusters = []
    current_cluster = [lines_with_depths[0][0]]
    depths_of_lines_in_current_cluster = [lines_with_depths[0][1]]

    for line, depth in lines_with_depths[1:]:
        x_avg = (line[0] + line[2]) / 2
        x_avg_cluster = (current_cluster[-1][0] + current_cluster[-1][2]) / 2
        if abs(x_avg - x_avg_cluster) < x_thresh:
            current_cluster.append(line)
            depths_of_lines_in_current_cluster.append(depth)
        else:
            clusters.append(current_cluster)
            depth_of_clusters.append(depths_of_lines_in_current_cluster)
            current_cluster = [line]
            depths_of_lines_in_current_cluster = [depth]

    clusters.append(current_cluster)  # to add the last cluster
    depth_of_clusters.append(depths_of_lines_in_current_cluster)

    # Merge each cluster into a single line spanning min y to max y
    merged_lines = []
    merged_lines_depths = []
    for cluster, depths in zip(clusters, depth_of_clusters):
        xs = [(l[0] + l[2]) / 2 for l in cluster]
        ys = [l[1] for l in cluster] + [l[3] for l in cluster]
        x_avg = int(np.mean(xs))
        y_min, y_max = min(ys), max(ys)
        if abs(y_max - y_min) >= min_merged_line_length:
            merged_lines.append(((x_avg, y_min),(x_avg, y_max)))
            merged_lines_depths.append(float(np.mean(depths)))
            cv2.line(color_image, (x_avg, y_min), (x_avg, y_max), (0, 255, 0), 2)  # green for filtered merged lines

    return merged_lines, merged_lines_depths