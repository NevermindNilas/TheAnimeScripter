"""Inference constants for the pretrained DUT weights.

Inlined from upstream configs/config.py at commit 2a013d1 (the resolution the
released checkpoints were trained at; upstream later repointed WIDTH/HEIGHT at
a portrait demo clip without retraining). These are properties of the weights,
not user knobs.
"""

# Model input resolution. RFDet_640 / MotionPro / Smoother were trained here.
WIDTH = 640
HEIGHT = 480

# Mesh grid cell size in model-resolution pixels -> 40x30 grid vertices.
PIXELS = 16

# Optical-flow normalization amplitude.
FLOWC = 20

# Neighbor radius (px) for blending the two cluster homographies.
RADIUS = 200

# Minimum secondary-cluster population before multi-homography kicks in.
THRESHOLDPOINT = 102

# Keypoints kept per frame.
TOPK = 512

# RFDet hyperparameters.
SCORE_COM_STRENGTH = 100.0
SCALE_COM_STRENGTH = 100.0
NMS_THRESH = 0.0
NMS_KSIZE = 5
GAUSSIAN_KSIZE = 15
GAUSSIAN_SIGMA = 0.5
KSIZE = 3
PADDING = 1
DILATION = 1
SCALE_LIST = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

# Motion clustering distance thresholds (magnitude px / rotation deg).
THRESHOLD_MANG = 2
THRESHOLD_ROT = 5
