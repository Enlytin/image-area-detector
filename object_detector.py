import cv2
import numpy as np
import config

def detect_objects(image_path, method='watershed', min_area=500):
    """
    Detect distinct objects in an image using improved computer vision techniques.
    
    Args:
        image_path (str): Path to the input image
        method (str): Detection method - 'watershed', 'adaptive', or 'contour'
        min_area (int): Minimum area threshold to filter out noise
    
    Returns:
        list: List of contours representing detected objects
    """
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    print(f"Using {method} method for object detection...")
    
    if method == 'watershed':
        return _watershed_detection(image, min_area)
    elif method == 'adaptive':
        return _adaptive_detection(image, min_area)
    elif method == 'contour':
        return _improved_contour_detection(image, min_area)
    else:
        print(f"Unknown method: {method}. Using watershed instead.")
        return _watershed_detection(image, min_area)

def _watershed_detection(image, min_area):
    """
    Watershed algorithm - best for separating touching objects
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    # Extract contours
    contours, _ = cv2.findContours((markers > 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Draw results
    _draw_detection_results(original, filtered_contours, "Watershed Detection")
    
    return filtered_contours

def _adaptive_detection(image, min_area):
    """
    Adaptive thresholding with improved filtering - good for varying lighting
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 10)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Additional filtering by perimeter-to-area ratio to remove noise
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.1:  # Filter out very irregular shapes
                    filtered_contours.append(contour)
    
    # Draw results
    _draw_detection_results(original, filtered_contours, "Adaptive Detection")
    
    return filtered_contours

def _improved_contour_detection(image, min_area):
    """
    Improved version of your original contour detection
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise
    denoised = cv2.medianBlur(gray, 5)
    
    # Use Otsu's thresholding for automatic threshold selection
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Improved morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Advanced filtering
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Filter by aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Filter by solidity (area/convex_hull_area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
                
                # Keep objects with reasonable aspect ratio and solidity
                if 0.1 < aspect_ratio < 10 and solidity > 0.3:
                    filtered_contours.append(contour)
    
    # Draw results
    _draw_detection_results(original, filtered_contours, "Improved Contour Detection")
    
    return filtered_contours

def _draw_detection_results(image, contours, title):
    """
    Draw bounding boxes and labels on detected objects
    """
    for i, contour in enumerate(contours):
        # Draw bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(image, f'Object {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add area information
        area = cv2.contourArea(contour)
        cv2.putText(image, f'Area: {int(area)}', (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    print(f"{title}: Found {len(contours)} objects")
    
    # Display result
    resized = cv2.resize(image, (800, 600))
    cv2.imshow(title, resized)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

def compare_all_methods(image_path):
    """
    Compare all three improved methods
    """
    print("Comparing improved object detection methods...")
    print("=" * 60)
    
    watershed_objects = detect_objects(image_path, 'watershed', min_area=300)
    adaptive_objects = detect_objects(image_path, 'adaptive', min_area=300)
    contour_objects = detect_objects(image_path, 'contour', min_area=300)
    
    print("\n" + "=" * 60)
    print("FINAL COMPARISON:")
    print(f"Watershed method: {len(watershed_objects)} objects")
    print(f"Adaptive method: {len(adaptive_objects)} objects") 
    print(f"Improved contour method: {len(contour_objects)} objects")
    print("=" * 60)
    
    return {
        'watershed': watershed_objects,
        'adaptive': adaptive_objects,
        'contour': contour_objects
    }

if __name__ == "__main__":
    image_path = f"{config.SAMPLE_DIRECTORY}/card_blank_grey.png"
    
    # Test all methods
    results = compare_all_methods(image_path)
    
    print("\nRECOMMENDATIONS:")
    print("1. Watershed: Best for separating touching/overlapping objects")
    print("2. Adaptive: Best for images with varying lighting conditions")
    print("3. Improved Contour: Enhanced version of your original approach")
    print("\nTry adjusting min_area parameter based on your image size and object sizes.")
