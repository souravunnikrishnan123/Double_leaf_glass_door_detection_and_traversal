import cv2
import numpy as np
import mediapipe as mp



def get_human_mask_mediapipe(color_img, segmentation, threshold=0.5):
    """
    Input: BGR color image (H,W,3)
    Output: binary mask (H,W) uint8 with 1 where person present
    """
    # Debug: verify segmentation object
    try:
        print(f"[segmentation debug] type={type(segmentation)}, exists_attr_process={hasattr(segmentation, 'process')}")
    except Exception as e:
        print("[segmentation debug] error inspecting segmentation:", e)

    img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    try:
        results = segmentation.process(img_rgb)
    except Exception as e:
        print("[segmentation debug] segmentation.process() raised:", e)
        return np.zeros(color_img.shape[:2], dtype=np.uint8)

    if results is None or results.segmentation_mask is None:
        print("[segmentation debug] results.segmentation_mask is None")
        return np.zeros(color_img.shape[:2], dtype=np.uint8)

    seg_mask = results.segmentation_mask.astype(np.float32)  # float32 HxW in [0,1]
    # Debug: show stats of raw segmentation output
    minv, maxv, meanv = float(seg_mask.min()), float(seg_mask.max()), float(seg_mask.mean())
    print(f"[segmentation debug] seg_mask min={minv:.4f} max={maxv:.4f} mean={meanv:.4f}")

    # Adaptive thresholding:
    # - If max is extremely small, assume no person.
    # - Otherwise pick a threshold proportional to the peak confidence but clamped so it's not too small.
    if maxv <= 0.02:
        # Too weak to trust
        print("[segmentation debug] max confidence <= 0.02 -> returning empty mask")
        return np.zeros(color_img.shape[:2], dtype=np.uint8)

    adaptive_thresh = max(0.02, min(0.25, maxv * 0.5))
    print(f"[segmentation debug] adaptive_thresh={adaptive_thresh:.4f} (user threshold param={threshold})")

    mask = (seg_mask >= adaptive_thresh).astype(np.uint8)

    # Postprocess: close small holes and keep only the largest connected component
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Keep largest contour only (remove spurious blobs)
        mask_vis = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            clean = np.zeros_like(mask_vis)
            cv2.drawContours(clean, [largest], -1, 255, thickness=-1)
            mask = (clean > 0).astype(np.uint8)
    except Exception as e:
        print("[segmentation debug] postprocessing failed:", e)

    # Debug: show mask window so you can visually confirm (non-blocking)
    try:
        cv2.imshow("human_mask_debug", (mask * 255).astype(np.uint8))
        cv2.waitKey(1)
    except Exception:
        pass

    return mask