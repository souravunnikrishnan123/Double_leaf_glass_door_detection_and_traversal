"""Pair vertical candidates and build the side ROIs used for door status."""

import cv2
import numpy as np
from duration import get_duration_seconds


class GlassFrameLineProcessor:
    """
    Turn candidate vertical lines into one likely center-frame pair.

    The processor compares physical glass/frame widths in pixel space, pairs
    nearby frame edges, and verifies that the regions outside the pair are
    deeper than the candidate frame itself.

    Attributes:
        keyword:
            Branch label used in duration names.

        depth_image_in_meters:
            Metric depth image for the frame currently being processed.

        color_image:
            Branch image receiving ROI and line-pair overlays.

        fx:
            Horizontal camera focal length in pixels.

        lines:
            Candidate vertical lines for the current frame.

        depth_of_each_lines:
            Metric depth associated one-to-one with ``lines``.

        door_geometry:
            Mapping containing physical glass/frame widths and ROI settings.

        DEPTH_RANGE:
            Accepted door-depth interval in meters.

        timer:
            Named duration recorder.
    """

    def __init__(self, keyword: str = "color_based"):
        """
        Initialize an empty frame-line processing context.

        Args:
            keyword:
                Intended branch label, normally ``"color_based"`` or
                ``"depth_based"``.

        Notes:
            Per-frame inputs are assigned by
            :meth:`find_left_right_roi_and_door_depth`. The current
            implementation resets ``keyword`` to ``None`` later in this
            constructor, so timing labels use ``None`` unless that assignment
            is changed.
        """
        self.keyword = keyword
        self.depth_image_in_meters = None
        self.color_image = None
        self.fx = None
        self.lines = []
        self.depth_of_each_lines = []
        self.door_geometry = []
        self.DEPTH_RANGE = []
        self.keyword = None

        #duration timer
        self.timer = get_duration_seconds()


    def filter_vertical_lines_glass_contact(self):
        """
        Filter and pair lines that could border a glass pane.

        Keeps lines that:
        - Have no neighbor on one side (left or right), OR
        - The neighbor is farther than `glass_width_cm` (in centimeters)

        Returns:
            List of paired candidates. Each item contains
            ``(left_line, right_line, left_depth, right_depth)``.

        Notes:
            Physical glass and center-frame widths are converted to expected
            pixel gaps independently for each line depth. A middle candidate
            must have one wide neighbor gap and one frame-width gap.
        """
        if not self.lines:
            return []

        # Door uprights are nearly vertical, so their mean x is a reliable and
        # inexpensive coordinate for ordering and pairing them.
        mean_x_unsorted = np.array([
            (np.mean([p[0] for p in ln]) if len(ln) > 0 else np.inf)
            for ln in self.lines
        ], dtype=np.float32)

        sorted_indices = np.argsort(mean_x_unsorted)
        lines_sorted = [self.lines[i] for i in sorted_indices]
        depth_of_each_lines_sorted = [self.depth_of_each_lines[i] for i in sorted_indices]

        # After sorting lines and depths
        line_to_depth = {id(line): depth for line, depth in zip(lines_sorted, depth_of_each_lines_sorted)}
        # Vectorized mean x for sorted lines
        mean_x_coords = np.array([
            (np.mean([p[0] for p in ln]) if len(ln) > 0 else np.inf)
            for ln in lines_sorted
        ], dtype=np.float32)

        # Expected pixel width shrinks with distance. Compute it per line instead
        # of assuming every candidate sits on exactly the same depth plane.
        # Convert depth to centimeters and compute per-line pixel gaps
        depth_cm_arr = np.array(depth_of_each_lines_sorted, dtype=np.float32) * 100.0
        valid_mask = depth_cm_arr > 0
        required_pixel_gap_arr = np.where(valid_mask, (self.door_geometry["glass_width_cm"] / depth_cm_arr) * self.fx, np.nan)
        frame_pixel_gap_arr = np.where(valid_mask, (self.door_geometry["center_frame_width_cm"] / depth_cm_arr) * self.fx, np.nan)

        # Vectorized neighbor checks and selection
        n = len(lines_sorted)
        filtered = []
        if n:
            mean_x_left = np.roll(mean_x_coords, 1)
            mean_x_right = np.roll(mean_x_coords, -1)
            dist_left = np.abs(mean_x_coords - mean_x_left)
            dist_right = np.abs(mean_x_coords - mean_x_right)

            idx = np.arange(n)
            has_left = idx > 0
            has_right = idx < (n - 1)

            both_mask = has_left & has_right
            # A center frame edge should have its partner close by and glass-width
            # clearance on the other side. That pattern is more useful than line
            # strength alone in a corridor full of vertical edges.
            # --- Case 1: Both left and right neighbors exist ---
            # One must be wide, one must be narrow
            cond_both = (
                ((dist_left >= required_pixel_gap_arr) & (dist_right <= frame_pixel_gap_arr)) |
                ((dist_right >= required_pixel_gap_arr) & (dist_left <= frame_pixel_gap_arr))
            ) & both_mask & valid_mask
            # --- Case 2: Only one neighbor exists ---
            cond_left_only = (dist_left <= frame_pixel_gap_arr) & has_left & (~has_right) & valid_mask
            cond_right_only = (dist_right <= frame_pixel_gap_arr) & has_right & (~has_left) & valid_mask

            selected_idx = np.where(cond_both | cond_left_only | cond_right_only)[0]
            filtered = [lines_sorted[i] for i in selected_idx.tolist()]

        # Use scalar gap compatible with downstream usage
        frame_pixel_gap_scalar = float(np.nanmean(frame_pixel_gap_arr)) if np.any(valid_mask) else 0.0
        paired_lines = self.get_paired_lines(filtered, frame_pixel_gap_scalar, line_to_depth)

        #print("Filtered lines:", filtered)
        #print("Paired lines:", paired_lines)

        return paired_lines

    def get_paired_lines(self, filtered, frame_pixel_gap, line_to_depth):
        """
        Pair horizontally neighboring lines within the expected frame width.

        Args:
            filtered:
                Candidate line point sequences.

            frame_pixel_gap:
                Maximum average-x separation in pixels for one frame pair.

            line_to_depth:
                Mapping from ``id(line)`` to metric line depth.

        Returns:
            Deduplicated list of
            ``(left_line, right_line, left_depth, right_depth)`` tuples.

        Notes:
            Lines are sorted from left to right. When both neighbors qualify,
            the left neighbor has priority. Pair identity is deduplicated
            without regard to order.
        """
        paired_lines = []

        if not filtered:
            return []

        # Sort by mean x (vectorized) to preserve left-to-right assumptions
        mean_x = np.array([
            (np.mean([pt[0] for pt in line]) if len(line) > 0 else np.inf)
            for line in filtered
        ], dtype=np.float32)
        sort_idx = np.argsort(mean_x)
        filtered = [filtered[i] for i in sort_idx]
        mean_x = mean_x[sort_idx]

        n = len(filtered)
        # Neighbor means via roll; mask out invalid edges
        mean_left = np.roll(mean_x, 1)
        mean_right = np.roll(mean_x, -1)
        dist_left = np.abs(mean_x - mean_left)
        dist_right = np.abs(mean_x - mean_right)

        idx = np.arange(n)
        has_left = idx > 0
        has_right = idx < (n - 1)

        # Build selection masks
        both = has_left & has_right
        left_within = dist_left <= frame_pixel_gap
        right_within = dist_right <= frame_pixel_gap

        # Priority: if both neighbors within gap, choose left (as original behavior)
        choose_left = both & left_within & (~right_within | right_within)  # both within still chooses left
        # When only right within (and left not), choose right
        choose_right = both & (~left_within) & right_within
        # Only-left case (no right neighbor)
        choose_left_only = has_left & (~has_right) & left_within
        # Only-right case (no left neighbor)
        choose_right_only = has_right & (~has_left) & right_within

        left_indices = np.where(choose_left | choose_left_only)[0]
        right_indices = np.where(choose_right | choose_right_only)[0]

        # Build pairs from indices
        for i in left_indices.tolist():
            l1 = filtered[i - 1]
            l2 = filtered[i]
            paired_lines.append((l1, l2, line_to_depth[id(l1)], line_to_depth[id(l2)]))
        for i in right_indices.tolist():
            l1 = filtered[i]
            l2 = filtered[i + 1]
            paired_lines.append((l1, l2, line_to_depth[id(l1)], line_to_depth[id(l2)]))

        # The neighbor pass can discover the same physical pair from either edge.
        # Deduplicate pairs (order-insensitive) using ids
        unique_pairs = []
        seen = set()
        for l1, l2, d1, d2 in paired_lines:
            # Use tuple of sorted ids to avoid (A,B) and (B,A) duplicates
            key = tuple(sorted([id(l1), id(l2)]))
            if key not in seen:
                unique_pairs.append((l1, l2, d1, d2))
                seen.add(key)
        return unique_pairs

    def sort_lines_by_y(self, line_points):
        """
        Sort line samples from the top of the image downward.

        Args:
            line_points:
                Iterable of points whose second coordinate is image y.

        Returns:
            New list ordered by ascending y-coordinate.
        """
        # Sort points by y-coordinate (ascending)
        return sorted(line_points, key=lambda pt: pt[1])

    def get_strip_avg_z(self, depth_image_in_meters, line_points, side="left", roi_width=20, min_depth=1.7):
        """
        Measure average depth in a strip beside a sampled line.

        Args:
            depth_image_in_meters:
                Aligned ``H x W`` metric depth image.

            line_points:
                Ordered line samples containing at least x and y coordinates.

            side:
                ``"left"`` or ``"right"`` side on which to build the strip.

            roi_width:
                Perpendicular strip width in pixels.

            min_depth:
                Minimum accepted depth in meters.

        Returns:
            Tuple ``(average_depth, roi_polygon)``. A degenerate line returns
            ``(None, None)``; fewer than eleven valid pixels returns a polygon
            with ``average_depth`` set to ``None``.

        Notes:
            The ROI follows the line direction rather than assuming a perfectly
            vertical segment.
        """
        # Take only x, y (ignore any extra fields like depth)
        pts = np.array([(p[0], p[1]) for p in line_points], dtype=np.int32)

        # Compute direction of the line
        dx = float(pts[-1][0] - pts[0][0])
        dy = float(pts[-1][1] - pts[0][1])
        length = np.hypot(dx, dy)
        if length == 0:
            return None, None

        dx /= length
        dy /= length

        # Rotating the unit tangent by 90 degrees gives a strip that follows a
        # slightly tilted frame instead of assuming an axis-aligned rectangle.
        if side == "left":
            offset_vec = np.array([-dy, dx])
        else:
            offset_vec = np.array([dy, -dx])

        # Normalize once then scale to roi_width
        offset_vec = (offset_vec / np.linalg.norm(offset_vec)) * roi_width

        # Shift points
        pts_offset = pts + offset_vec
        roi_polygon = np.vstack((pts, pts_offset[::-1]))

        # Create mask for ROI
        mask = np.zeros(depth_image_in_meters.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [roi_polygon.astype(np.int32)], 255)

        roi_depth = depth_image_in_meters[mask == 255]

        # Vectorized filtering
        valid_mask = (roi_depth >= min_depth) & (roi_depth != 0) & ~np.isnan(roi_depth)
        valid_depths = roi_depth[valid_mask]

        # A handful of isolated returns is not enough to describe a whole side ROI.
        avg_z = float(np.mean(valid_depths)) if len(valid_depths) > 10 else None
        return avg_z, roi_polygon

    def extrapolate_line_to_y_range(self, line_points, y_start, y_end, step=1):
        """
        Fit x as a function of y and sample an extended vertical line.

        Args:
            line_points:
                At least two ``(x, y)`` samples.

            y_start:
                First output image row.

            y_end:
                Last output image row, included when reached by ``step``.

            step:
                Row interval between output samples.

        Returns:
            List of integer ``(x, y)`` points. Fewer than two input samples
            returns an empty list.

        Notes:
            Least-squares fitting uses ``x = a*y + b``, which is stable for
            nearly vertical lines.
        """
        pts = np.array([(p[0], p[1]) for p in line_points], dtype=np.float32)
        # Fit line: y = m*x + c, but we want x as a function of y for vertical lines
        # Use np.polyfit with degree 1
        if len(pts) < 2:
            return  []  # Not enough points

        # Fit x = a*y + b (since vertical lines)
        #fitting a straight line (least squares regression) to all points in line_points.
        a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)



        ys = np.arange(y_start, y_end + 1, step, dtype=np.int32)
        xs = (a * ys + b).astype(np.int32)
        sampled_points = list(zip(xs.tolist(), ys.tolist()))
        return sampled_points

    def process_filtered_lines(self, filtered_lines):
        """
        Build and validate side ROIs for every candidate frame pair.

        Args:
            filtered_lines:
                Iterable of
                ``(left_line, right_line, left_depth, right_depth)`` tuples.

        Returns:
            Tuple ``(left_roi, right_roi, mean_frame_depth)`` for the selected
            pair, or ``(None, None, None)`` when no pair passes validation.

        Notes:
            Lines are extrapolated to full image height only for constructing
            ROIs; original line depths remain the frame-depth estimates. A pair
            is accepted when both outside ROIs are deeper than their frame
            edges by ``correction_factor``. If several pairs pass, the pair
            nearest the image center is selected.
        """

        avg_z_left_roi_list = []
        avg_z_right_roi_list = []
        filtered_pairs = []
        mean_z_depth_along_frame_lines_list = []
        valid_roi_polygon_left_list = []
        valid_roi_polygon_right_list = []

        for i, (left_line_points, right_line_points, left_depth, right_depth) in enumerate(filtered_lines):
            """

            y_bottom = max(np.max([pt[1] for pt in left_line_points]),
                           np.max([pt[1] for pt in right_line_points]),
                           np.max([pt[1] for pt in detected_plane]) if (detected_plane is not None and len(detected_plane) > 0) else depth_image_in_meters.shape[0] - 1
                            )
            """
            y_bottom = self.color_image.shape[0] - 1
            y_top = 0


            # Full-height lines define useful side regions even when Hough only
            # observed a short, unobstructed section of the physical frame.
            extrapolated_left_line_points = self.extrapolate_line_to_y_range(left_line_points, y_top, y_bottom)
            extrapolated_right_line_points = self.extrapolate_line_to_y_range(right_line_points, y_top, y_bottom)


            # Compute average Z for left ROI
            avg_z_left_roi, roi_polygon_left = self.get_strip_avg_z(
                self.depth_image_in_meters, extrapolated_left_line_points, side="left", roi_width=self.door_geometry["roi_width"], min_depth=self.DEPTH_RANGE[0]
            )


            # Compute average Z for right ROI
            avg_z_right_roi, roi_polygon_right = self.get_strip_avg_z(
                self.depth_image_in_meters, extrapolated_right_line_points, side="right", roi_width=self.door_geometry["roi_width"], min_depth=self.DEPTH_RANGE[0]
            ) # minimum depth used to ignore the depth info from human who is between the door and robodog


            # Calculate mean Z along the left line. we are not using extrapolated_left_line_points because
            # more reliable depth info comes from the original line points. extrapolated points shall be used only for defining the ROIs
            # to get the maximum area where we expect lot of zero and non zero depth values
            #z_left_line = [pt[2] for pt in left_line_points if pt[2] > 0 and not np.isnan(pt[2])]
            #mean_z_left_line = float(np.mean(z_left_line)) if z_left_line else None
            mean_z_left_line = left_depth

            # Calculate mean Z along the original right line points
            #z_right_line = [pt[2] for pt in right_line_points if pt[2] > 0 and not np.isnan(pt[2])]
            #mean_z_right_line = float(np.mean(z_right_line)) if z_right_line else None
            mean_z_right_line = right_depth
            if mean_z_left_line is not None and mean_z_right_line is not None:
                mean_z_depth_along_frame_lines = 0.5 * (mean_z_left_line + mean_z_right_line)
            else:
                mean_z_depth_along_frame_lines = None

                    # Filter pairs based on your criteria
            # Both outside strips must lie behind their corresponding uprights.
            # This rejects pairs formed by texture lines on a single flat surface.
            if (mean_z_left_line is not None and avg_z_left_roi is not None and mean_z_left_line * self.door_geometry["correction_factor"] < avg_z_left_roi) and (mean_z_right_line is not None and avg_z_right_roi is not None and mean_z_right_line * self.door_geometry["correction_factor"] < avg_z_right_roi):
                avg_z_left_roi_list.append(avg_z_left_roi)
                avg_z_right_roi_list.append(avg_z_right_roi)
                filtered_pairs.append((extrapolated_left_line_points, extrapolated_right_line_points, left_depth, right_depth))
                valid_roi_polygon_left_list.append(roi_polygon_left)
                valid_roi_polygon_right_list.append(roi_polygon_right)
                mean_z_depth_along_frame_lines_list.append(mean_z_depth_along_frame_lines)


                # Draw ROIs if available
                if roi_polygon_left is not None:
                    cv2.polylines(self.color_image, [roi_polygon_left.astype(np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)
                if roi_polygon_right is not None:
                    cv2.polylines(self.color_image, [roi_polygon_right.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

                # Annotate average Z values
                cv2.putText(self.color_image, f"Left ROI {i+1} Z: {avg_z_left_roi:.2f} m", (30, 30 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(self.color_image, f"Right ROI {i+1} Z: {avg_z_right_roi:.2f} m", (30, 50 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(self.color_image, f"Left Line {i+1} Z: {mean_z_left_line:.2f} m", (300, 30 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(self.color_image, f"Right Line {i+1} Z: {mean_z_right_line:.2f} m", (300, 50 + i*40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if filtered_pairs:
            if len(filtered_pairs) > 1:  # more than one pair detected as glass frame.
                #in this case we need to filter out the correct frame line at the center. if we are getting more than one glass -frame candidate means, mostly it is due to the glass area in the  inward opening door
                #so in this case chances are high that ransac detected the whole door plane. hence we can use the width of detected ransac plane to filter out the correct frame line pair
                #correct frame line pair will be close to the center of detected ransac door plane
                # 1. Compute the center x of the detected plane (average of its 4 corners)
                #plane_center_x = np.mean([pt[0] for pt in detected_plane]) if detected_plane is not None else color_image.shape[1] // 2
                # The robot stops facing the doorway, so image center is a stable
                # tie-breaker when reflections create more than one valid pair.
                plane_center_x = self.color_image.shape[1] // 2 #since wer are stopping about 2m far from glass door plane. the center of entire view would be almost same as the center of glass door plane
                #this is to avoid dependancy to glass detection algorithm
                # 2. Find the pair whose center is closest to the plane center
                min_dist = float('inf')
                best_pair = None
                for left_line, right_line, left_depth, right_depth in filtered_pairs:
                    # Compute mean x of left and right line
                    left_x = float(np.mean([pt[0] for pt in left_line]))
                    right_x = float(np.mean([pt[0] for pt in right_line]))
                    pair_center_x = (left_x + right_x) / 2
                    dist = abs(pair_center_x - plane_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (left_line, right_line, left_depth, right_depth)

                idx = filtered_pairs.index(best_pair)

            else:  # filtered pairs has only one pair.
                best_pair = filtered_pairs[0]
                idx = 0



            roi_polygon_left = valid_roi_polygon_left_list[idx]
            roi_polygon_right = valid_roi_polygon_right_list[idx]
            mean_z_depth_along_frame_lines = mean_z_depth_along_frame_lines_list[idx]
            mean_z_depth_along_frame_lines = float(mean_z_depth_along_frame_lines) if mean_z_depth_along_frame_lines is not None else None

            return roi_polygon_left, roi_polygon_right, mean_z_depth_along_frame_lines
        else:
            return None, None, None

    def find_left_right_roi_and_door_depth(self,
                                       depth_image_in_meters,
                                        color_image,
                                        fx,
                                        lines,
                                        depth_of_each_lines,
                                        door_geometry,
                                        DEPTH_RANGE,
                                        ):
        """
        Select a frame pair and derive its left/right status-checking ROIs.

        This is the public entry point. It stores the current frame inputs,
        filters candidates using physical door geometry, orders pair samples,
        draws candidates, and delegates ROI validation.

        Args:
            depth_image_in_meters:
                Aligned metric depth image.

            color_image:
                Branch image modified with diagnostic overlays.

            fx:
                Horizontal focal length in pixels.

            lines:
                Candidate vertical lines represented by endpoint pairs or
                sampled point sequences.

            depth_of_each_lines:
                Metric depth associated one-to-one with every line.

            door_geometry:
                Mapping containing ``glass_width_cm``,
                ``center_frame_width_cm``, ``roi_width``, and
                ``correction_factor``.

            DEPTH_RANGE:
                Accepted door-depth interval in meters.

        Returns:
            Left ROI polygon, right ROI polygon, and mean frame depth. Missing
            detections are reported as ``(None, None, 0)`` or
            ``(None, None, None)`` depending on the rejection stage.

        Notes:
            Candidate pairs are drawn red/cyan and accepted side ROIs are drawn
            magenta/yellow on ``color_image``.
        """

        self.depth_image_in_meters = depth_image_in_meters
        self.color_image = color_image
        self.fx = fx
        self.lines = lines
        self.depth_of_each_lines = depth_of_each_lines
        self.door_geometry = door_geometry
        self.DEPTH_RANGE = DEPTH_RANGE


        if lines is None or len(lines) == 0:
            return None, None, 0

        self.timer.start(f"{self.keyword}_image_based_frame_detection pairing and roi processing")

        # Pairing is shared by both the RGB and depth branches; only the source
        # candidates and their diagnostic image differ.
        paired_lines = self.filter_vertical_lines_glass_contact()
        # Adjust all pairs so each line's points are sorted by y. so that gradient ccan be calculated correctly

        paired_lines_sorted = [
            (self.sort_lines_by_y(left_line), self.sort_lines_by_y(right_line), left_depth, right_depth)
            for left_line, right_line, left_depth, right_depth in paired_lines]


        # Visualize paired lines (glass frame candidates)
        for left_line, right_line, left_depth, right_depth in paired_lines_sorted:
            # Draw left line in red
            if len(left_line) >= 2:
                pt1 = tuple(map(int, left_line[0][:2]))
                pt2 = tuple(map(int, left_line[-1][:2]))
                cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)
            # Draw right line in cyan
            if len(right_line) >= 2:
                pt1 = tuple(map(int, right_line[0][:2]))
                pt2 = tuple(map(int, right_line[-1][:2]))
                cv2.line(color_image, pt1, pt2, (0, 255, 255), 2)



        # Example: Extract full coordinates of each stable line
        #for avg_x, line_pts, confidence in stable_lines:
            # Print the number of points instead of shape
            #print(f"Stable Line X={avg_x}, Confidence={confidence:.2f}, NumPoints={len(line_pts)}")

        roi_polygon_left, roi_polygon_right , mean_z_depth_along_frame_lines = self.process_filtered_lines(paired_lines_sorted)

        self.timer.stop(f"{self.keyword}_image_based_frame_detection pairing and roi processing")
        return roi_polygon_left, roi_polygon_right , mean_z_depth_along_frame_lines
