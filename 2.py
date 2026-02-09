import cv2
import numpy as np

# --- CONFIGURATION ---
WEBCAM_ID = "https://192.0.0.4:8080/video" # Your IP Camera
# WEBCAM_ID = 0  # USB Camera

def empty(a):
    pass

def process_monochromatic_metal():
    cap = cv2.VideoCapture(WEBCAM_ID)
    
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 400, 200)
    
    # 1. Quality: 10 = High precision corners, 50 = Loose corners
    cv2.createTrackbar("Quality (1/1000)", "Parameters", 10, 100, empty)
    # 2. Min Distance: Minimum pixels between two detected corners
    cv2.createTrackbar("Min Distance", "Parameters", 20, 100, empty)
    # 3. Block Size: Size of the area to look for a corner (3 or 5 is usually best)
    cv2.createTrackbar("Block Size", "Parameters", 3, 10, empty)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Working on a copy
        img = frame.copy()
        
        # --- STEP 1: PRE-PROCESSING FOR SAME-COLOR OBJECTS ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Bilateral Filter: Removes noise but KEEPS edges sharp (Crucial for metal)
        # Arguments: src, d, sigmaColor, sigmaSpace
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # --- STEP 2: MORPHOLOGICAL GRADIENT ---
        # This highlights TEXTURE/TOPOLOGY changes, not just color changes.
        # It detects the 'bump' of the weld or the 'crease' of the bend.
        kernel = np.ones((3,3), np.uint8)
        morph_grad = cv2.morphologyEx(filtered, cv2.MORPH_GRADIENT, kernel)

        # Enhance the gradient (make the lines brighter)
        _, grad_thresh = cv2.threshold(morph_grad, 20, 255, cv2.THRESH_BINARY)

        # --- STEP 3: CORNER DETECTION (Shi-Tomasi) ---
        # Get values from trackbars
        quality_val = cv2.getTrackbarPos("Quality (1/1000)", "Parameters")
        if quality_val < 1: quality_val = 1
        quality_level = quality_val / 1000.0
        
        min_dist = cv2.getTrackbarPos("Min Distance", "Parameters")
        if min_dist < 1: min_dist = 1
        
        block_size = cv2.getTrackbarPos("Block Size", "Parameters")
        if block_size < 3: block_size = 3
        
        # Detect Corners
        # image, maxCorners, qualityLevel, minDistance, mask, blockSize
        corners = cv2.goodFeaturesToTrack(
            filtered, 
            maxCorners=50, 
            qualityLevel=quality_level, 
            minDistance=min_dist,
            blockSize=block_size,
            useHarrisDetector=False # False = Shi-Tomasi (Better for coordinates)
        )

        # Draw the findings
        if corners is not None:
            corners = np.int0(corners)
            
            # Connect the dots visually to show the "Skeleton"
            # (Sorting helps draw lines in order, but raw points are just plotted here)
            for i in corners:
                x, y = i.ravel()
                # Draw outer circle (Green)
                cv2.circle(img, (x, y), 8, (0, 255, 0), 1) 
                # Draw center dot (Red)
                cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
                # Text
                cv2.putText(img, f"{x},{y}", (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Show views
        # Stack images to see what the computer sees vs real life
        cv2.imshow("1. Surface Topology (Gradient)", grad_thresh)
        cv2.imshow("2. Detected Corners", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_monochromatic_metal()