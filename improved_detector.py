import cv2
import numpy as np
from sklearn.cluster import KMeans
import config

def find_objects_watershed(image_path):
    """
    Uses watershed algorithm for better object segmentation
    """
    print("Running Watershed-based object detection...")
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Use adaptive thresholding for better results
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Noise removal using morphological operations
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    
    # Mark the region of unknown with zero
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # Mark boundaries in red
    
    # Find and draw contours of detected objects
    contours, _ = cv2.findContours((markers > 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area to remove noise
    min_area = 500  # Adjust based on your image size
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Draw bounding boxes around detected objects
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(original, f'Object {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    print(f"Found {len(filtered_contours)} objects using Watershed")
    
    # Display results
    cv2.imshow("Original", cv2.resize(original, (800, 600)))
    cv2.imshow("Watershed Result", cv2.resize(image, (800, 600)))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return filtered_contours

def find_objects_kmeans_segmentation(image_path, k=3):
    """
    Uses K-means clustering for color-based segmentation
    """
    print(f"Running K-means segmentation with {k} clusters...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    original = image.copy()
    
    # Reshape image to be a list of pixels
    data = image.reshape((-1, 3))
    data = np.float32(data)
    
    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8 and reshape to original image shape
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(image.shape)
    
    # Find contours for each cluster
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    
    all_contours = []
    for i in range(k):
        # Create mask for current cluster
        mask = (labels.flatten() == i).astype(np.uint8) * 255
        mask = mask.reshape(image.shape[:2])
        
        # Find contours in this cluster
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by area
        min_area = 1000
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        all_contours.extend(filtered)
        
        # Draw contours for this cluster
        color = tuple(map(int, centers[i]))
        cv2.drawContours(original, filtered, -1, color, 2)
    
    print(f"Found {len(all_contours)} objects using K-means")
    
    # Display results
    cv2.imshow("Original with K-means", cv2.resize(original, (800, 600)))
    cv2.imshow("K-means Segmentation", cv2.resize(segmented_image, (800, 600)))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return all_contours

def find_objects_edge_detection(image_path):
    """
    Uses advanced edge detection with contour filtering
    """
    print("Running advanced edge detection...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Use Canny edge detection
    edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
    
    # Dilate edges to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            # Calculate aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Filter by reasonable aspect ratios (adjust as needed)
            if 0.2 < aspect_ratio < 5.0:
                filtered_contours.append(contour)
    
    # Draw bounding boxes
    for i, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(original, f'Obj {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    print(f"Found {len(filtered_contours)} objects using edge detection")
    
    # Display results
    cv2.imshow("Edge Detection Result", cv2.resize(original, (800, 600)))
    cv2.imshow("Edges", cv2.resize(edges, (800, 600)))
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    
    return filtered_contours

def compare_methods(image_path):
    """
    Compare all three methods on the same image
    """
    print("Comparing different object detection methods...")
    print("=" * 50)
    
    # Run all methods
    watershed_contours = find_objects_watershed(image_path)
    kmeans_contours = find_objects_kmeans_segmentation(image_path, k=4)
    edge_contours = find_objects_edge_detection(image_path)
    
    print("\n" + "=" * 50)
    print("COMPARISON RESULTS:")
    print(f"Watershed method: {len(watershed_contours) if watershed_contours else 0} objects")
    print(f"K-means method: {len(kmeans_contours) if kmeans_contours else 0} objects")
    print(f"Edge detection method: {len(edge_contours) if edge_contours else 0} objects")
    print("=" * 50)

if __name__ == "__main__":
    image_path = f"{config.SAMPLE_DIRECTORY}/card_blank_grey.png"
    
    # Compare all methods
    compare_methods(image_path)
    
    print("\nRecommendation:")
    print("- Use Watershed for objects that touch each other")
    print("- Use K-means for color-based segmentation")
    print("- Use Edge detection for well-defined object boundaries")
    print("- Adjust parameters based on your specific image characteristics")
