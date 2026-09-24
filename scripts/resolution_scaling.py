#!/usr/bin/env python3
"""Convert resolution-dependent parameters to the current image scale.

Every pixel-domain value in ``door_detection.launch`` is written for the
reference resolution, meaning ``resize_scale = 1.0``. When the bridge publishes
smaller images the same physical structure covers fewer pixels, so those values
must be converted before use. Keeping the conversion here means the launch file
stays readable as a single calibrated set rather than one set per resolution.

Quantities scale differently depending on what they measure, so each kind has
its own method rather than a single multiplier:

* pixel distances scale with the linear factor,
* pixel areas scale with its square,
* strides over the valid-pixel list scale with its square, because the number
  of valid pixels does,
* per-pixel depth gradients scale inversely, because the same physical step is
  spread over fewer pixels.

Values that are already physical (metres, centimetres), unitless (ratios,
percentiles, angles) or counts of 3D points are not converted.
"""

import rospy


class ResolutionScaler:
    """
    Convert reference-resolution parameters to the active image resolution.

    Attributes:
        resize_scale:
            Linear size of the published images relative to the reference
            resolution. ``1.0`` means the parameters are used unchanged.
    """

    def __init__(self, resize_scale=None):
        """
        Read the active image scale.

        Args:
            resize_scale:
                Explicit scale factor. When omitted it is read from the private
                ``~resize_scale`` parameter, defaulting to ``1.0``.

        Notes:
            A non-positive or unreadable value falls back to ``1.0`` so a
            misconfigured parameter cannot silently disable detection.
        """
        if resize_scale is None:
            try:
                resize_scale = rospy.get_param("~resize_scale", 1.0)
            except Exception:
                resize_scale = 1.0
        try:
            resize_scale = float(resize_scale)
        except (TypeError, ValueError):
            resize_scale = 1.0
        if not resize_scale > 0.0:
            rospy.logwarn(
                "resize_scale %r is not positive; treating parameters as unscaled.",
                resize_scale,
            )
            resize_scale = 1.0
        self.resize_scale = resize_scale

    @property
    def is_identity(self):
        """Whether the active resolution matches the reference resolution."""
        return self.resize_scale == 1.0

    def length(self, value, minimum=1):
        """
        Convert a pixel distance.

        Args:
            value:
                Distance in reference-image pixels, such as a line length, a
                region width, or a Hough accumulator threshold.

            minimum:
                Smallest result allowed, so a small value cannot round to zero
                and disable the test it belongs to.

        Returns:
            Distance in current-image pixels, as an ``int``.
        """
        return max(int(minimum), int(round(float(value) * self.resize_scale)))

    def odd_length(self, value, minimum=3):
        """
        Convert a pixel distance that must stay an odd kernel size.

        Args:
            value:
                Kernel size in reference-image pixels.

            minimum:
                Smallest kernel allowed; also forced odd.

        Returns:
            Odd kernel size in current-image pixels, as an ``int``.

        Notes:
            OpenCV kernels must be odd, so the scaled value is rounded to the
            nearest odd number rather than the nearest integer.
        """
        minimum = int(minimum) | 1
        scaled = int(round(float(value) * self.resize_scale))
        if scaled % 2 == 0:
            scaled += 1
        return max(minimum, scaled)

    def area(self, value, minimum=1):
        """
        Convert a pixel area.

        Args:
            value:
                Area in reference-image square pixels.

            minimum:
                Smallest result allowed.

        Returns:
            Area in current-image square pixels, as an ``int``.
        """
        factor = self.resize_scale * self.resize_scale
        return max(int(minimum), int(round(float(value) * factor)))

    def stride(self, value, minimum=1):
        """
        Convert a stride taken over the list of valid pixels.

        Args:
            value:
                Stride calibrated at the reference resolution.

            minimum:
                Smallest stride allowed. ``1`` keeps every valid pixel.

        Returns:
            Stride for the current resolution, as an ``int``.

        Notes:
            Back-projection strides the flattened list of valid pixels, so the
            number of points it returns is that list's length divided by the
            stride. The list shrinks with the image area, so the stride is
            scaled by the square of the linear factor to hold the resulting
            point count roughly constant. Absolute point-count thresholds such
            as ``min_inliers`` therefore keep their meaning.

            The conversion cannot go below ``minimum``. Once it clamps there the
            point count is no longer held constant and starts falling with the
            image area, so the absolute count thresholds lose their calibration.
            That is reported once per call site rather than passed over silently,
            because the symptom - RANSAC and the frame-width bins quietly running
            out of points - looks like a detection failure, not a configuration
            one.
        """
        factor = self.resize_scale * self.resize_scale
        scaled = int(round(float(value) * factor))
        if scaled < int(minimum):
            try:
                rospy.logwarn_once(
                    "resize_scale=%.3f wants a back-projection stride of %.2f, "
                    "below the floor of %d. The back-projected point count will "
                    "fall by about %.1fx instead of staying constant, so absolute "
                    "count thresholds (ransac/min_inliers, min_points_per_bin, "
                    "min_points_per_vertical_slice, "
                    "minimum_depth_points_after_filtering) are no longer "
                    "calibrated and should be lowered to match.",
                    self.resize_scale,
                    float(value) * factor,
                    int(minimum),
                    float(minimum) / max(1e-9, float(value) * factor),
                )
            except Exception:
                pass
        return max(int(minimum), scaled)

    def gradient(self, value):
        """
        Convert a per-pixel depth gradient threshold.

        Args:
            value:
                Threshold in metres per reference-image pixel.

        Returns:
            Threshold in metres per current-image pixel, as a ``float``.

        Notes:
            Scaling is inverse: a depth step spanning a given physical distance
            covers fewer pixels in a smaller image, so the measured gradient per
            pixel grows as the image shrinks.
        """
        return float(value) / self.resize_scale

    def describe(self):
        """Return a short human-readable summary for start-up logging."""
        if self.is_identity:
            return "resize_scale=1.0 (parameters used at their reference values)"
        return (
            "resize_scale=%.4g (pixel parameters scaled from their reference "
            "values; lengths x%.4g, areas and strides x%.4g, per-pixel "
            "gradients x%.4g)" % (
                self.resize_scale,
                self.resize_scale,
                self.resize_scale ** 2,
                1.0 / self.resize_scale,
            )
        )
