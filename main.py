import cv2
import numpy as np
import math

# --- CONFIGURATION ---
WEBCAM_ID = "https://192.0.0.4:8080/video"  # Use your IP camera URL here
# WEBCAM_ID = 0  # Uncomment this line if using USB webcam

# Global Variables
scale_mm_per_pixel = None
roi_box = None  # Format: (x, y, w, h)
drawing_roi = False
ix, iy = -1, -1
calibration_points = []
calibrating_mode = False

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing_roi, roi_box, calibration_points, calibrating_mode, scale_mm_per_pixel

    # --- ROI SELECTION (Click and Drag) ---
    if not calibrating_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_roi = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing_roi:
                # Visualize selection
                pass
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_roi = False
            w, h = abs(x - ix), abs(y - iy)
            # Ensure width/height are valid
            if w > 10 and h > 10:
                roi_box = (min(ix, x), min(iy, y), w, h)
                print(f"ROI Set: {roi_box}")
            else:
                roi_box = None
                print("ROI cleared")

    # --- CALIBRATION (Click 2 Points) ---
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            calibration_points.append((x, y))
            print(f"Calibration Point {len(calibration_points)}: {x,y}")
            if len(calibration_points) == 2:
                # Calculate pixel distance
                p1, p2 = calibration_points
                dist_px = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
                
                if dist_px > 0:
                    try:
                        # Ask user for real distance
                        real_dist = float(input(f"Enter real distance for {int(dist_px)} pixels (in mm): "))
                        scale_mm_per_pixel = real_dist / dist_px
                        print(f"CALIBRATED: 1 Pixel = {scale_mm_per_pixel:.4f} mm")
                    except ValueError:
                        print("Invalid number entered. Calibration failed.")
                
                calibration_points = []
                calibrating_mode = False

def empty(a):
    pass

def main():
    global calibrating_mode, roi_box

    cap = cv2.VideoCapture(WEBCAM_ID)
    
    # Create Windows
    cv2.namedWindow("Control Panel")
    cv2.resizeWindow("Control Panel", 400, 150)
    cv2.namedWindow("Main View")
    
    # Trackbars for tuning (To remove floor noise)
    cv2.createTrackbar("Threshold 1", "Control Panel", 100, 255, empty)
    cv2.createTrackbar("Threshold 2", "Control Panel", 200, 255, empty)
    
    cv2.setMouseCallback("Main View", mouse_callback)

    print("--- CONTROLS ---")
    print("[Mouse Drag] : Draw Bounding Box (ROI)")
    print("[c]          : Start Calibration (Click 2 points on tape)")
    print("[r]          : Reset ROI")
    print("[q]          : Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Copy frame for display
        display_frame = frame.copy()
        
        # Get Trackbar Values
        t1 = cv2.getTrackbarPos("Threshold 1", "Control Panel")
        t2 = cv2.getTrackbarPos("Threshold 2", "Control Panel")

        # 1. Process Only ROI if it exists
        if roi_box is not None:
            x, y, w, h = roi_box
            # Draw ROI Box
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract ROI
            roi = frame[y:y+h, x:x+w]
            
            # Preprocessing
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny Edge Detection (Using Trackbar values)
            edges = cv2.Canny(blurred, t1, t2)
            
            # Close gaps
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Show the mask to help user tune
            cv2.imshow("Black & White View (Tune Trackbars!)", mask)

            # Find Contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 500: continue  # Ignore small noise spots

                # Approx Polygons (to find bends/corners)
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                # Draw Contours relative to full image
                # We need to shift points back by (x,y) of ROI
                shifted_approx = approx + [x, y]
                cv2.drawContours(display_frame, [shifted_approx], -1, (0, 255, 255), 2)

                # Draw Points
                for point in shifted_approx:
                    px, py = point[0]
                    cv2.circle(display_frame, (px, py), 5, (0, 0, 255), -1)
                    
                    # Coordinate Text
                    if scale_mm_per_pixel:
                        # Convert to mm (relative to top-left of ROI or Image)
                        mm_x = (px - x) * scale_mm_per_pixel
                        mm_y = (py - y) * scale_mm_per_pixel
                        text = f"{int(mm_x)},{int(mm_y)}"
                    else:
                        text = f"{px},{py}"
                    
                    cv2.putText(display_frame, text, (px+10, py), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            cv2.putText(display_frame, "Draw a Box around metal part", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Calibration Visuals
        if calibrating_mode:
            cv2.putText(display_frame, f"CLICK POINT {len(calibration_points)+1} ON TAPE", 
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            for pt in calibration_points:
                cv2.circle(display_frame, pt, 5, (255, 0, 255), -1)

        cv2.imshow("Main View", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'): # Reset ROI
            roi_box = None
            cv2.destroyWindow("Black & White View (Tune Trackbars!)")
        elif key == ord('c'): # Start Calibration
            calibrating_mode = True
            calibration_points = []
            print("Calibration Mode Started. Click 2 points.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()