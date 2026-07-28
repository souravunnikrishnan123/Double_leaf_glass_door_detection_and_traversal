#!/usr/bin/env python3
"""Track approximately vertical line candidates across recent frames."""

import rospy
import numpy as np
import cv2



def update_line_history(filtered_lines, line_history, DISTANCE_THRESHOLD, MAX_LINES_TO_TRACK, color_image):
    """
    Add the current line candidates to history and draw stable tracks.

    Each line is reduced to its mean x-coordinate for cross-frame clustering,
    while the original point samples are retained for visualization.

    Args:
        filtered_lines:
            Iterable of lines, where each line is a sequence of ``(x, y)``
            pixel coordinates.

        line_history:
            Bounded deque updated in place with the current frame's lines.

        DISTANCE_THRESHOLD:
            Maximum horizontal distance in pixels for one line cluster.

        MAX_LINES_TO_TRACK:
            Maximum number of stable clusters to draw.

        color_image:
            BGR image modified by the stable-line overlay.

    Returns:
        ``None``. History and ``color_image`` are modified in place.
    """
    frame_lines = []
    for pts in filtered_lines:
        pts = np.array(pts)  # Convert list of tuples to Nx2 array
        avg_x = int(np.mean(pts[:, 0]))
        frame_lines.append((avg_x, pts))
    line_history.append(frame_lines)

    stable_lines = get_stable_lines(line_history, DISTANCE_THRESHOLD, MAX_LINES_TO_TRACK)

    draw_stable_lines(color_image, stable_lines)

def get_stable_lines(line_history, DISTANCE_THRESHOLD, MAX_LINES_TO_TRACK):
    """
    Cluster historical lines by horizontal position and rank their stability.

    Args:
        line_history:
            Sequence of per-frame ``(mean_x, points)`` collections.

        DISTANCE_THRESHOLD:
            Maximum x-distance in pixels for adding a line to an existing
            cluster.

        MAX_LINES_TO_TRACK:
            Maximum number of ranked clusters to return.

    Returns:
        List of ``(average_x, recent_line_points, confidence)`` tuples ordered
        by decreasing confidence. Fewer than five history entries returns an
        empty list.

    Notes:
        Confidence is the number of lines assigned to a cluster divided by the
        number of frames in history.
    """
    if len(line_history) < 5:
        return []  # Not enough history

    # Flatten all historical lines into one list
    all_lines = []
    for frame_lines in line_history:
        all_lines.extend(frame_lines)

    # Group lines by proximity in X
    clusters = []
    for avg_x, pts in all_lines:
        placed = False
        for cluster in clusters:
            if abs(cluster["center_x"] - avg_x) < DISTANCE_THRESHOLD:
                cluster["lines"].append((avg_x, pts))
                cluster["center_x"] = np.mean([l[0] for l in cluster["lines"]])
                placed = True
                break
        if not placed:
            clusters.append({"center_x": avg_x, "lines": [(avg_x, pts)]})

    # Compute stability (frequency) for each cluster
    stable_lines = []
    for cluster in clusters:
        # Count how many frames this cluster appeared in
        frame_counts = set()
        for avg_x, _ in cluster["lines"]:
            frame_counts.add(avg_x)  # Rough frame-based uniqueness
        confidence = len(cluster["lines"]) / (len(line_history))  # normalized
        # Take the most recent line points from this cluster
        recent_line = cluster["lines"][-1][1]
        stable_lines.append((int(cluster["center_x"]), recent_line, confidence))

    # Sort clusters by confidence (most stable first)
    stable_lines.sort(key=lambda x: x[2], reverse=True)

    # Keep top N stable lines
    return stable_lines[:MAX_LINES_TO_TRACK]



def draw_stable_lines(image, stable_lines, color=(0, 255, 255)):
    """
    Draw stable line tracks and confidence labels on an image.

    Args:
        image:
            BGR image modified in place.

        stable_lines:
            Iterable of ``(average_x, point_array, score)`` tuples.

        color:
            BGR line and text color.

    Returns:
        ``None``. Invalid or single-point line arrays are skipped.
    """
    for avg_x, pts, score in stable_lines:
        # Ensure pts is a numpy array of points
        pts = np.array(pts)
        if pts.ndim != 2 or pts.shape[1] < 2:
            continue
        if len(pts) < 2:
            continue
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[-1][0]), int(pts[-1][1])
        cv2.line(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"x={avg_x}, s={score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
