import mediapipe as mp


def setup_segmentation_model():
    # -----------------------------
    # MediaPipe setup (selfie segmentation)
    # -----------------------------
    mp_selfie = mp.solutions.selfie_segmentation
    mp_drawing = mp.solutions.drawing_utils


    # Use model_selection=1 for general (0 is landscape? see docs), default is fine.
    segmentation = mp_selfie.SelfieSegmentation(model_selection=1)

    return mp_drawing, segmentation