import cv2
import numpy as np
import config

def detect_objects_simple(image_path, min_area=500, show_steps=False):
    """
    Simple, effective object detection using watershed algorithm.
    This method works well for separating distinct objects.
    
    Args:
        image_path (str): Path to input image
        min_area (int): Minimum area to consider as an object
        show_steps (bool): Whether to show intermediate processing steps
    
    Returns:
        list: Detected object contours
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot load image from {image_path}")
        return []
    
    original = image.copy()
    print(f"Processing image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Noise reduction
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    if show_steps:
        cv2.imshow("1. Blurred", cv2.resize(blurred, (400, 300)))
        cv2.waitKey(1000)
    
    # Step 2: Adaptive thresholding (better than fixed threshold)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    if show_steps:
        cv2.imshow("2. Threshold", cv2.resize(thresh, (400, 300)))
        cv2.waitKey(1000)
    
    # Step 3: Morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    if show_steps:
        cv2.imshow("3. Opening", cv2.resize(opening, (400, 300)))
        cv2.waitKey(1000)
    
    # Step 4: Watershed algorithm for object separation
    # Find sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Find sure foreground using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    if show_steps:
        cv2.imshow("4. Distance Transform", cv2.resize(dist_transform, (400, 300)))
        cv2.waitKey(1000)
    
    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Extract object contours
    contours, _ = cv2.findContours((markers > 1).astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and shape quality
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Additional quality check: remove very elongated shapes
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio < 5:  # Not too elongated
                filtered_contours.append(contour)
    
    # Draw results
    result_image = original.copy()
    for i, contour in enumerate(filtered_contours):
        # Draw bounding box
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add object label
        cv2.putText(result_image, f'Object {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add area info
        area = cv2.contourArea(contour)
        cv2.putText(result_image, f'Area: {int(area)}', (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    print(f"Found {len(filtered_contours)} objects")
    
    # Show final result
    cv2.imshow("Object Detection Result", cv2.resize(result_image, (800, 600)))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return filtered_contours

def get_object_info(contours):
    """
    Extract useful information about detected objects
    """
    objects_info = []
    
    for i, contour in enumerate(contours):
        # Basic measurements
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Shape analysis
        aspect_ratio = float(w) / h
        extent = float(area) / (w * h)
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        obj_info = {
            'id': i + 1,
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'solidity': solidity,
            'center': (x + w//2, y + h//2)
        }
        
        objects_info.append(obj_info)
    
    return objects_info


