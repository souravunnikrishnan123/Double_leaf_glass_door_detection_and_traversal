import numpy as np


def cluster_and_merge_lines(lines, depth_of_valid_lines, x_thresh=10):
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
        merged_lines.append((x_avg, y_min, x_avg, y_max))
        merged_lines_depths.append(float(np.mean(depths)))

    return merged_lines, merged_lines_depths