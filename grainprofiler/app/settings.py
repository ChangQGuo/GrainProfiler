"""Application-wide constants matching pipeline visualization conventions."""

# Padding used when cropping subimages from YOLO boxes (matches pipeline config.yaml)
PADDING_PX = 20

# Contour / axis drawing colours (BGR, matching pipeline/utils/visualization.py)
CONTOUR_COLOR = (0, 180, 0)       # green
AXIS_COLOR = (170, 60, 190)       # purple
BOTTOM_COLOR = (0, 0, 255)        # red
TOP_COLOR = (255, 0, 255)         # magenta
CENTROID_COLOR = (0, 255, 255)    # yellow

# Qt colours for graphics items
CONTOUR_NORMAL = (0, 180, 0)
CONTOUR_HOVER = (0, 255, 0)
CONTOUR_SELECTED = (255, 255, 0)

CONTOUR_WIDTH_NORMAL = 1.0
CONTOUR_WIDTH_HOVER = 2.5
CONTOUR_WIDTH_SELECTED = 3.0

# Axis line width
AXIS_WIDTH = 1

# Endpoint marker radius (scaled by image dimension later)
ENDPOINT_RADIUS_BASE = 3

# Sibling directories to search for original photos
PHOTO_SIBLING_DIRS = [
    "test_image_min", "test_image", "test_image_best",
    "image_data", "sample_image", "image_site_data_result2",
]

# Minimum window size
WINDOW_MIN_WIDTH = 1400
WINDOW_MIN_HEIGHT = 800
