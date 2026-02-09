import cv2
import numpy as np
import os
import glob

# --- CONFIGURATION ---
INPUT_FOLDER = 'sample'
OUTPUT_FOLDER = 'output_lines'
EXTENSIONS = ['*.jpg', '*.jpeg', '*.png']

# --- TUNING PARAMETERS (Adjust these if lines are missing) ---
CANNY_THRESHOLD_1 = 50   # Low threshold for edge detection
CANNY_THRESHOLD_2 = 150  # High threshold (contrast)
HOUGH_THRESHOLD = 50     # Minimum votes to be considered a line (Lower = more lines)
MIN_LINE_LENGTH = 100    # Minimum length of a line in pixels (Removes small noise)
MAX_LINE_GAP = 20        # Max gap allowed to connect broken lines (e.g. over a weld)

def process_lines():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"Found {len(image_files)} images. Extracting lines...")

    for file_path in image_files:
        filename = os.path.basename(file_path)
        img = cv2.imread(file_path)
        if img is None: continue

        draw_img = img.copy()
        
        # 1. Pre-processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur is CRITICAL for metal to remove rust texture
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Edge Detection (Canny)
        edges = cv2.Canny(blurred, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

        # 3. Hough Line Transform
        # This finds mathematical lines in the edge map
        lines = cv2.HoughLinesP(
            edges, 
            1,                  # Rho: Distance resolution (1 pixel)
            np.pi/180,          # Theta: Angle resolution (1 degree)
            threshold=HOUGH_THRESHOLD,
            minLineLength=MIN_LINE_LENGTH,
            maxLineGap=MAX_LINE_GAP
        )

        line_count = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate angle (optional, to filter diagonal/straight lines)
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                
                # Draw the line
                cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Start/End Coordinates
                # Start (Blue dot)
                cv2.circle(draw_img, (x1, y1), 5, (255, 0, 0), -1)
                # End (Blue dot)
                cv2.circle(draw_img, (x2, y2), 5, (255, 0, 0), -1)
                
                # Label the coordinates (Small text)
                cv2.putText(draw_img, f"{x1},{y1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(draw_img, f"{x2},{y2}", (x2, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                line_count += 1

        # Save
        output_path = os.path.join(OUTPUT_FOLDER, "lines_" + filename)
        cv2.imwrite(output_path, draw_img)
        print(f"   Processed {filename}: Found {line_count} lines.")

    print(f"\nDone. Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    process_lines()