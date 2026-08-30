"""Pipeline configuration.

Every constant governing detection, zoom and rigid matching lives here. The
values are calibrated on the reference dataset: changing them changes the
numbers the pipeline produces.
"""

# --- RF-DETR class ids ---
CLASS_BALL = 1
CLASS_DOT = 2
CLASS_HAND = 3

# --- Detection confidence thresholds ---
CONF_BALL = 0.35  # raised for cleaner ball detection
CONF_DOT = 0.25   # lowered to catch more dots
CONF_HAND = 0.30
SKIP_INITIAL_FRAMES = 1

# --- Zoom for dot detection ---
ZOOM_FACTOR = 3.0      # 3x magnification
ZOOM_PADDING = 2.2     # ball fills ~40-50% of the crop, not 90-100%: the
                       # leftover margin keeps semantic context (floor,
                       # background) the model relies on

# --- Rigid matching: gating and robustness (two-stage adaptive) ---
DOWNWARD_EPS = 0.15     # directional gating tolerance
MAX_STEP = 1.0          # max downward movement
MAX_ROTATION_DEG = 40.0 # max rotation per frame

# STRICT: first attempt, high precision
INLIER_THRESH_STRICT = 0.08
DIST_TOLERANCE_STRICT = 0.10
MIN_INLIERS_STRICT = 3

# LOOSE: fallback, more permissive
INLIER_THRESH_LOOSE = 0.10
DIST_TOLERANCE_LOOSE = 0.15
MIN_INLIERS_LOOSE = 2

# --- Model weights (not versioned: see .gitignore) ---
DEFAULT_WEIGHTS = "rf-detr/weights.pt"
DEFAULT_DEVICE = "cpu"
