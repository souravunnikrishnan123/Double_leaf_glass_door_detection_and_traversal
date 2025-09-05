import numpy as np


def cluster_and_merge_lines(lines, x_thresh=20):
    """
    Cluster vertical lines by proximity in x-coordinate and merge into one line per cluster.
    Args:
        lines: list of (x1, y1, x2, y2)
        x_thresh: max horizontal distance (in pixels) to group lines
    Returns:
        merged_lines: list of merged (x1, y1, x2, y2)
    """
    if not lines:
        return []

    # Sort lines by average x
    lines_sorted = sorted(lines, key=lambda l: (l[0] + l[2]) / 2)
    clusters = []
    current_cluster = [lines_sorted[0]]

    for line in lines_sorted[1:]:
        x_avg = (line[0] + line[2]) / 2
        x_avg_cluster = (current_cluster[-1][0] + current_cluster[-1][2]) / 2
        if abs(x_avg - x_avg_cluster) < x_thresh:
            current_cluster.append(line)
        else:
            clusters.append(current_cluster)
            current_cluster = [line]
    clusters.append(current_cluster)

    # Merge each cluster into a single line spanning min y to max y
    merged_lines = []
    for cluster in clusters:
        xs = [(l[0] + l[2]) / 2 for l in cluster]
        ys = [l[1] for l in cluster] + [l[3] for l in cluster]
        x_avg = int(np.mean(xs))
        y_min, y_max = min(ys), max(ys)
        merged_lines.append((x_avg, y_min, x_avg, y_max))

    return merged_lines