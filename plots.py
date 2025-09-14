import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def debug_visualize(points=None, uv=None, plane_model=None, inlier_indices=None):

    # Visualize uv (pixel positions of valid depth points)
    if uv is not None and uv.shape[0] > 0:
        plt.figure()
        plt.scatter(uv[:, 0], uv[:, 1], s=1, c='blue')
        plt.gca().invert_yaxis()  # flip because image coords have origin top-left
        plt.title("Pixel positions (uv) of valid depth points")
        plt.xlabel("u (cols)")
        plt.ylabel("v (rows)")
        plt.show()

    # Visualize 3D points (X, Z) as top-down view
    if points is not None and points.shape[0] > 0:
        plt.figure()
        plt.scatter(points[:, 0], points[:, 2], s=1, c='red')
        plt.title("Top-down view of 3D points (X vs Z)")
        plt.xlabel("X [m]")
        plt.ylabel("Z [m]")
        plt.axis("equal")
        plt.show()

    # Only proceed if inlier_indices and plane_model are valid
    if (
        points is not None and points.shape[0] > 0 and
        inlier_indices is not None and len(inlier_indices) > 0 and
        plane_model is not None
    ):
        inlier_pts = points[inlier_indices]
        outlier_mask = np.ones(points.shape[0], dtype=bool)
        outlier_mask[inlier_indices] = False
        outlier_pts = points[outlier_mask]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(outlier_pts[:, 0], outlier_pts[:, 1], outlier_pts[:, 2], s=1, c='gray', label="Outliers")
        ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], s=1, c='red', label="Inliers")
        ax.set_title("RANSAC Plane Fit: Inliers vs Outliers")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.legend()
        plt.show()

        # Get plane coefficients
        a, b, c, d = plane_model

        # Pick a patch of X, Y
        xx, yy = np.meshgrid(
            np.linspace(points[:,0].min(), points[:,0].max(), 10),
            np.linspace(points[:,1].min(), points[:,1].max(), 10)
        )
        zz = (-a * xx - b * yy - d) / c  # solve for z

        # Overlay with inliers
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='gray')
        ax.scatter(inlier_pts[:, 0], inlier_pts[:, 1], inlier_pts[:, 2], s=3, c='red')
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='cyan')
        ax.set_title("Plane model with inliers")
        plt.show()
    else:
        print("No valid plane/inliers for 3D visualization.")

