import cv2
import numpy as np
import mediapipe as mp


def get_human_mask_mediapipe(color_img,segmentation, threshold=0.5):
    """
    Input: BGR color image (H,W,3)
    Output: binary mask (H,W) uint8 with 1 where person present
    """
    img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    results = segmentation.process(img_rgb)
    if results.segmentation_mask is None:
        return np.zeros(color_img.shape[:2], dtype=np.uint8)
    seg_mask = results.segmentation_mask  # float32 HxW in [0,1]
    mask = (seg_mask > threshold).astype(np.uint8)
    return mask