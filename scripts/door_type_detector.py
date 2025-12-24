import numpy as np
import rospy
import cv2



class DoorTypeDetector():
    """
    Estimate glass and frame widths from RANSAC plane inliers only.
    Robust to glass transparency, bottom metal frames, depth clipping, and camera skew.
    """
    def __init__(self):
        self.bin_width = rospy.get_param("~door_type_detector/bin_width", 0.02)              # meters (2 cm)
        self.min_vertical_support = rospy.get_param("~door_type_detector/min_vertical_support", 0.6)    # fraction of door height
        self.min_points_per_bin = rospy.get_param("~door_type_detector/min_points_per_bin", 5)
        self.min_glass_width = rospy.get_param("~door_type_detector/min_glass_width", 0.16)   #  segments
        self.max_glass_width = rospy.get_param("~door_type_detector/max_glass_width", 1.5)
        self.num_vertical_slices = rospy.get_param("~door_type_detector/num_vertical_slices", 40)
        self.min_points_per_vertical_slice = rospy.get_param("~door_type_detector/min_points_per_vertical_slice", 2)
    
    def estimate_glass_and_frame_widths(self,
        inlier_points
    ):
        """
        Estimate glass and frame widths using ONLY RANSAC plane inliers,
        robust to:
        - glass transparency
        - bottom metal frames
        - depth clipping (e.g. 1–3 m)
        - camera skew / robot orientation

        The key idea:
        A true frame supports the plane over MOST of the door height.
        Glass supports the plane only locally (bottom / sides).

        Parameters
        ----------
        plane_model : tuple (a, b, c, d)
            Plane equation ax + by + cz + d = 0 from RANSAC
        inlier_points : (N, 3) ndarray
            XYZ coordinates of RANSAC plane inliers
        bin_width : float
            Width of bins along the door horizontal axis (meters)
        min_vertical_support : float
            Minimum fraction of door height that must be supported
            to classify a bin as "frame"
        min_points_per_bin : int
            Minimum inlier points required to evaluate a bin

        Returns
        -------
        result : dict with keys
            - bins : list of (x_start, x_end)
            - bin_labels : list of "frame" or "glass"
            - segments : list of (x_start, x_end, label)
            - glass_widths_m : list of widths in meters
            - frame_widths_m : list of widths in meters
        """

        # Horizontal (width) coordinate
        xs = inlier_points[:, 0]

        # Vertical (height) coordinate
        ys = inlier_points[:,1]  # Y in world coords is already vertical

        # Total door height supported by inliers
        door_y_min = np.percentile(ys, 2)
        door_y_max = np.percentile(ys, 98)

        door_x_min = np.percentile(xs, 2)
        door_x_max = np.percentile(xs, 98)

        door_height = door_y_max - door_y_min
        if door_height <= 0:
            return None

        # ------------------------------------------------------------
        # STEP 3: Create bins along door width (PHYSICAL bins)
        # ------------------------------------------------------------


        bin_edges = np.arange(door_x_min, door_x_max + self.bin_width, self.bin_width)

        bin_labels = []
        bins = []

        # ------------------------------------------------------------
        # STEP 4: Classify each bin using VERTICAL SUPPORT
        # ------------------------------------------------------------

        for i in range(len(bin_edges) - 1):
            x0, x1 = bin_edges[i], bin_edges[i + 1]
            bins.append((x0, x1))

            # Inlier points whose horizontal coordinate falls in this bin
            in_bin = (xs >= x0) & (xs < x1)

            if np.count_nonzero(in_bin) < self.min_points_per_bin:
                # No or very few inliers → no vertical support → glass
                bin_labels.append("glass")
                continue

            y_vals = ys[in_bin]


            vertical_support_ratio = self.compute_vertical_occupancy_ratio(
                                        y_vals=y_vals,
                                        door_y_min=door_y_min,
                                        door_y_max=door_y_max
                                    )

            if vertical_support_ratio >= self.min_vertical_support:
                bin_labels.append("frame")
            else:
                bin_labels.append("glass")

        # ------------------------------------------------------------
        # STEP 5: Merge consecutive bins into segments
        # ------------------------------------------------------------

        segments = []
        start_idx = 0
        current_label = bin_labels[0]

        for i in range(1, len(bin_labels)):
            if bin_labels[i] != current_label:
                segments.append((
                    bin_edges[start_idx],
                    bin_edges[i],
                    current_label
                ))
                start_idx = i
                current_label = bin_labels[i]

        # Add final segment
        segments.append((
            bin_edges[start_idx],
            bin_edges[len(bin_labels)],
            current_label
        ))

        # ------------------------------------------------------------
        # STEP 6: Extract glass and frame widths
        # ------------------------------------------------------------

        glass_widths_m = []
        frame_widths_m = []

        for x0, x1, label in segments:
            width = x1 - x0
            if label == "glass":
                glass_widths_m.append(width)
            else:
                frame_widths_m.append(width)

        merged_segments = self.consolidate_door_segments(segments)

        # ------------------------------------------------------------
        # STEP 7: Select final frame/glass widths per adjacency rules
        # ------------------------------------------------------------
        final_frame_width_m = None
        final_glass_width_m = None
        final_frame_index = None
        final_glass_index = None

        if merged_segments and len(merged_segments) >= 1:
            widths = [x1 - x0 for (x0, x1, _) in merged_segments]
            labels = [label for (_, _, label) in merged_segments]

            # Filter glass segments by max width constraint
            glass_ok = set([i for i, label in enumerate(labels)
                            if label == "glass" and widths[i] <= self.max_glass_width])

            # Case A: a frame flanked by glass on both sides
            candidate_frames = []
            for i in range(len(merged_segments)):
                if labels[i] != "frame":
                    continue
                left_ok = (i - 1) in glass_ok if i - 1 >= 0 and labels[i - 1] == "glass" else False
                right_ok = (i + 1) in glass_ok if i + 1 < len(merged_segments) and labels[i + 1] == "glass" else False
                if left_ok and right_ok:
                    candidate_frames.append(i)

            if candidate_frames:
                # Choose the widest frame among candidates
                i_best = max(candidate_frames, key=lambda k: widths[k])
                final_frame_width_m = widths[i_best]
                final_frame_index = i_best
                final_glass_width_m = min(widths[i_best - 1], widths[i_best + 1])
                final_glass_index = i_best - 1 if widths[i_best - 1] <= widths[i_best + 1] else i_best + 1
            else:
                # Case B: exactly one valid glass segment, use adjacent frames
                if len(glass_ok) == 1:
                    g_idx = list(glass_ok)[0]
                    final_glass_width_m = widths[g_idx]
                    final_glass_index = g_idx

                    adj_frame_widths = []
                    if g_idx - 1 >= 0 and labels[g_idx - 1] == "frame":
                        adj_frame_widths.append(widths[g_idx - 1])

                    if g_idx + 1 < len(merged_segments) and labels[g_idx + 1] == "frame":
                        adj_frame_widths.append(widths[g_idx + 1])
                    
                    if adj_frame_widths:
                        final_frame_width_m = min(adj_frame_widths)
                        final_frame_index = g_idx - 1 if final_frame_width_m == widths[g_idx - 1] else g_idx + 1

        result = {
            "bins": bins,
            "bin_labels": bin_labels,
            "segments": merged_segments,
            "final_frame_width_m": final_frame_width_m,
            "final_glass_width_m": final_glass_width_m,
            "final_frame_index": final_frame_index,
            "final_glass_index": final_glass_index
        }

        return result



    def compute_vertical_occupancy_ratio(self,
        y_vals,
        door_y_min,
        door_y_max
    ):
        """
        Robust vertical support estimation using OCCUPANCY RATIO.

        Parameters
        ----------
        y_vals : (N,) ndarray
            Door-local vertical coordinates of inlier points in ONE width bin
        door_y_min : float
            Robust lower bound of door vertical extent
        door_y_max : float
            Robust upper bound of door vertical extent
        num_slices : int
            Number of vertical slices across door height
        min_points_per_slice : int
            Minimum inlier points required to mark a slice as occupied

        Returns
        -------
        vertical_occupancy_ratio : float in [0, 1]
        """


        # Divide the FULL door height into equal slices
        slice_edges = np.linspace(
            door_y_min,
            door_y_max,
            self.num_vertical_slices + 1
        )

        occupied = 0

        for i in range(self.num_vertical_slices):
            y0 = slice_edges[i]
            y1 = slice_edges[i + 1]

            # Count inliers in this vertical slice
            in_slice = (y_vals >= y0) & (y_vals < y1)

            if np.count_nonzero(in_slice) >= self.min_points_per_vertical_slice:
                occupied += 1

        return occupied / self.num_vertical_slices


    def consolidate_door_segments(self,
    segments):
        """
        Consolidate over-segmented door geometry into
        meaningful glass and frame widths.

        Parameters
        ----------
        segments : list of (x_start, x_end, label)
            Output from vertical-support algorithm
        min_segment_width : float
            Minimum width to consider a GLASS segment meaningful.
            Frame segments are kept regardless of width.


        Returns
        -------
        dict with:
            glass_widths_m : list
            frame_widths_m : list
        """

        # --------------------------------------------------
        # STEP 1: Remove tiny GLASS segments (keep all frames)
        # --------------------------------------------------
        filtered = []
        for seg in segments:
            x0, x1, label = seg
            width = x1 - x0
            if label != "glass" or width >= self.min_glass_width:
                filtered.append(seg)

        if not filtered:
            return [], []

        # --------------------------------------------------
        # STEP 2: Merge adjacent segments with same label
        # --------------------------------------------------
        merged = [filtered[0]]

        for x0, x1, label in filtered[1:]:
            px0, px1, plabel = merged[-1]

            if label == plabel:
                # Extend previous segment
                merged[-1] = (px0, x1, label)
            else:
                merged.append((x0, x1, label))


        return merged
        
    


    def visualize_door_bins_and_widths(self,
        inlier_points,
        bin_edges,
        bin_labels,
        segments,
        final_glass_index,
        final_frame_index,
        fx, cx,
        color_image
    ):
        """
        Visualize:
        - Width bins (based on world X)
        - Bin classification (glass / frame)
        - Final detected glass & frame widths

        Output: OpenCV BGR image for ROS publishing
        """

        vis = color_image.copy()
        H, W, _ = vis.shape

        # --------------------------------------------------
        # STEP 1: Map world X → image u (pixels)
        # --------------------------------------------------
        a, b = self.map_world_x_to_image_u(inlier_points, fx, cx)

        def world_x_to_img_u(x):
            return int(a * x + b)

        # --------------------------------------------------
        # STEP 3: Draw per-bin visualization
        # --------------------------------------------------
        for (x0, x1), label in zip(bin_edges, bin_labels):
            u0 = world_x_to_img_u(x0)
            u1 = world_x_to_img_u(x1)

            u0, u1 = np.clip([u0, u1], 0, W - 1)

            if label == "frame":
                color = (255, 0, 0)     # blue
            else:
                color = (0, 255, 255)   # yellow

            overlay = vis.copy()
            cv2.rectangle(
                overlay,
                (u0, 0),
                (u1, H),
                color,
                -1
            )
            vis = cv2.addWeighted(overlay, 0.20, vis, 0.80, 0)

            # Bin boundary
            cv2.line(vis, (u0, 0), (u0, H), (0, 0, 255), 1)

        # --------------------------------------------------
        # STEP 4: Draw consolidated segments (thick boxes)
        # --------------------------------------------------
        for i, (x0, x1, label) in enumerate(segments):
            u0 = world_x_to_img_u(x0)
            u1 = world_x_to_img_u(x1)

            u0, u1 = np.clip([u0, u1], 0, W - 1)

            width_cm = (x1 - x0) * 100.0

            if label == "frame":
                if i == final_frame_index:
                    color = (0, 0, 255)   # red
                    text = f"Final Frame {width_cm:.1f} cm"
                else:
                    color = (0, 255, 0)   # green
                    text = f"Frame {width_cm:.1f} cm"
            else:
                if i == final_glass_index:
                    color = (255, 0, 0)   # blue
                    text = f"Final Glass {width_cm:.1f} cm"
                else:
                    color = (0, 165, 255) # orange
                    text = f"Glass {width_cm:.1f} cm"

            cv2.rectangle(vis, (u0, 5), (u1, H - 5), color, 3)

            cv2.putText(
                vis,
                text,
                (u0 + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return vis

    def map_world_x_to_image_u(self, inlier_points, fx, cx):
        """
        Build a linear mapping from world X coordinate
        to image u pixel coordinate using pinhole projection.

        This avoids door-plane coordinates and uses raw X,Z.
        """

        xs = inlier_points[:, 0]
        zs = inlier_points[:, 2]

        # Project points to image u coordinate
        us = (xs * fx / zs) + cx

        # Fit linear mapping: u ≈ a * X + b
        a, b = np.polyfit(xs, us, 1)
        return a, b



