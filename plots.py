import numpy as np
import matplotlib.pyplot as plt

# Keep global references
fig_uv, ax_uv, sc_uv = None, None, None
fig_topdown, ax_topdown, sc_topdown = None, None, None
fig_inliers, ax_inliers = None, None
fig_plane, ax_plane = None, None


def debug_visualize(points=None, uv=None, plane_model=None, inlier_indices=None):
    global fig_uv, ax_uv, sc_uv
    global fig_topdown, ax_topdown, sc_topdown
    global fig_inliers, ax_inliers
    global fig_plane, ax_plane

    """
    # --- 1. Pixel positions (uv) ---
    if uv is not None and uv.shape[0] > 0:
        if fig_uv is None:
            fig_uv, ax_uv = plt.subplots()
            sc_uv = ax_uv.scatter(uv[:, 0], uv[:, 1], s=1, c='blue')
            ax_uv.invert_yaxis()
            ax_uv.set_title("Pixel positions (uv)")
            ax_uv.set_xlabel("u (cols)")
            ax_uv.set_ylabel("v (rows)")
        else:
            sc_uv.set_offsets(uv)
    """
    # --- 2. Top-down view (X vs Z) ---
    if points is not None and points.shape[0] > 0:
        if fig_topdown is None:
            fig_topdown, ax_topdown = plt.subplots()
            sc_topdown = ax_topdown.scatter(points[:, 0], points[:, 2], s=1, c='red')
            ax_topdown.set_title("Top-down view (X vs Z)")
            ax_topdown.set_xlabel("X [m]")
            ax_topdown.set_ylabel("Z [m]")
            ax_topdown.axis("equal")
        else:
            sc_topdown.set_offsets(points[:, [0, 2]])

    # --- 3. Inliers vs Outliers ---
    if (
        points is not None and points.shape[0] > 0 and
        inlier_indices is not None and len(inlier_indices) > 0
    ):
        inlier_pts = points[inlier_indices]
        outlier_mask = np.ones(points.shape[0], dtype=bool)
        outlier_mask[inlier_indices] = False
        outlier_pts = points[outlier_mask]

        if fig_inliers is None:
            fig_inliers = plt.figure()
            ax_inliers = fig_inliers.add_subplot(111, projection="3d")
            ax_inliers.set_title("RANSAC Inliers vs Outliers")
            ax_inliers.set_xlabel("X [m]")
            ax_inliers.set_ylabel("Y [m]")
            ax_inliers.set_zlabel("Z [m]")

        ax_inliers.cla()
        ax_inliers.scatter(outlier_pts[:, 0], outlier_pts[:, 1], outlier_pts[:, 2], s=1, c="gray", label="Outliers")
        ax_inliers.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], s=3, c="red", label="Inliers")
        ax_inliers.legend()
    """
    # --- 4. Plane Surface + Inliers ---
    if (
        points is not None and points.shape[0] > 0 and
        inlier_indices is not None and len(inlier_indices) > 0 and
        plane_model is not None
    ):
        inlier_pts = points[inlier_indices]

        if fig_plane is None:
            fig_plane = plt.figure()
            ax_plane = fig_plane.add_subplot(111, projection="3d")
            ax_plane.set_title("Plane Fit (surface)")
            ax_plane.set_xlabel("X [m]")
            ax_plane.set_ylabel("Y [m]")
            ax_plane.set_zlabel("Z [m]")

        ax_plane.cla()
        ax_plane.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c="gray", alpha=0.3)
        ax_plane.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], s=3, c="red")

        a, b, c, d = plane_model
        xx, yy = np.meshgrid(
            np.linspace(points[:, 0].min(), points[:, 0].max(), 10),
            np.linspace(points[:, 1].min(), points[:, 1].max(), 10),
        )
        zz = (-a * xx - b * yy - d) / c
        ax_plane.plot_surface(xx, yy, zz, alpha=0.3, color="cyan")
    """

    # Refresh all plots
    plt.pause(0.001)
