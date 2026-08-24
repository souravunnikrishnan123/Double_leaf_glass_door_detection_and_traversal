#!/usr/bin/env python3
"""Extend, cluster, and merge vertical door-frame line candidates."""

import rospy
import numpy as np
import cv2


def extrapolate_along_line_segment(depth_image_in_meters, start_point, direction_vector, center_depth,
                                   gradient_threshold=0.1, window=5, max_steps=100):
    """
    Extrapolate a line while sampled depth remains near a reference depth.

    Candidate pixels are generated in one vectorized operation. Out-of-bounds
    and invalid-depth samples are removed, the remaining depth sequence is
    smoothed, and the segment is truncated at its first large deviation.

    Args:
        depth_image_in_meters:
            ``H x W`` aligned depth image in meters.

        start_point:
            Starting ``(x, y)`` pixel. Sampling begins one step beyond it.

        direction_vector:
            Per-step ``(dx, dy)`` direction, normally a unit vector.

        center_depth:
            Reference line depth in meters.

        gradient_threshold:
            Maximum allowed absolute difference from ``center_depth``.

        window:
            Width of the moving-average kernel.

        max_steps:
            Maximum number of pixels sampled along the ray.

    Returns:
        List of accepted ``(x, y)`` pixels in sampling order. Returns an empty
        list when no in-bounds pixel has valid depth.

    Notes:
        The current implementation returns pixel ``(x, y)`` pairs; depth is
        used only to decide where extrapolation should stop. Convolution uses
        zero-padding and the result is front-padded with its edge value.
    """
    


    H, W = depth_image_in_meters.shape

    # --- Define ray coordinates ---
    x0, y0 = start_point
    dx, dy = direction_vector
    # Start one step past the endpoint so the original segment is not duplicated.
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
    # The first sustained range change is treated as the end of the same surface.
    deviation = np.abs(smoothed - float(center_depth))
    cond_break = deviation > float(gradient_threshold)

    if np.any(cond_break):
        brk = np.argmax(cond_break)
        xs, ys, d = xs[:brk], ys[:brk], d[:brk]

    # --- Return list of tuples ---
    return [(int(x), int(y)) for x, y, z in zip(xs, ys, d)]





def cluster_and_merge_lines(color_image,lines, depth_of_valid_lines, x_thresh, min_merged_line_length ):
    """
    Cluster nearby vertical lines and merge each cluster into one segment.

    Lines are sorted by average x-coordinate. Adjacent lines closer than
    ``x_thresh`` join the same cluster, whose output spans the minimum to
    maximum endpoint y-coordinate and carries the mean input depth.

    Args:
        color_image:
            BGR image modified with green merged-line overlays.

        lines:
            List of ``((x1, y1), (x2, y2))`` segments.

        depth_of_valid_lines:
            Metric depth associated one-to-one with each input line.

        x_thresh:
            Maximum horizontal distance in pixels used to group adjacent lines.

        min_merged_line_length:
            Minimum vertical span in pixels retained after merging.

    Returns:
        Tuple ``(merged_lines, merged_depths)``. Each merged line is
        ``((average_x, minimum_y), (average_x, maximum_y))`` and its matching
        depth is the cluster mean.

    Notes:
        ``lines`` and ``depth_of_valid_lines`` are paired with ``zip``; extra
        elements in the longer input are ignored.
    """
    if not lines:
        return [], []

    # Keep depths attached while sorting; the parallel arrays must stay aligned.
    # Sort lines and depths together by average x
    lines_with_depths = sorted(
        zip(lines, depth_of_valid_lines),
        key=lambda ld: (ld[0][0][0] + ld[0][1][0]) / 2
    )
    clusters = []
    depth_of_clusters = []
    current_cluster = [lines_with_depths[0][0]]
    depths_of_lines_in_current_cluster = [lines_with_depths[0][1]]

    for line, depth in lines_with_depths[1:]:
        x_avg = (line[0][0] + line[1][0]) / 2
        x_avg_cluster = (
            current_cluster[-1][0][0] + current_cluster[-1][1][0]
        ) / 2
        # Adjacency after sorting is enough for these nearly vertical segments.
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
        xs = [(line[0][0] + line[1][0]) / 2 for line in cluster]
        ys = (
            [line[0][1] for line in cluster]
            + [line[1][1] for line in cluster]
        )
        x_avg = int(np.mean(xs))
        y_min, y_max = min(ys), max(ys)
        # Short clusters are usually texture or sensor noise rather than an upright.
        if abs(y_max - y_min) >= min_merged_line_length:
            merged_lines.append(((x_avg, y_min),(x_avg, y_max)))
            merged_lines_depths.append(float(np.mean(depths)))
            cv2.line(color_image, (x_avg, y_min), (x_avg, y_max), (0, 255, 0), 2)  # green for filtered merged lines

    return merged_lines, merged_lines_depths
