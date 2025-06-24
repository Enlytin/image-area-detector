'''
Configuration variables for image_area_detector
'''
SAMPLE_DIRECTORY = "./sample"

# Image Processing Configurations
IMAGE_THRESHOLD = 59
IMAGE_KERNEL_SIZE = (3, 3)
IMAGE_MORPH_ITERATIONS = 1
IMAGE_DILATE_ITERATIONS = 1
IMAGE_CONTOUR_COLOR = (0, 255, 0)
IMAGE_CONTOUR_THICKNESS = 3
IMAGE_MEDIAN_BLUR_SIZE = 7

# App GUI Configurations
APP_WINDOW_TITLE = "Image Area Detector"
APP_WINDOW_GEOMETRY = "600x400"
APP_FILETYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
    ("All files", "*.*")
]
