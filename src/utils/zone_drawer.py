import cv2
import numpy as np
import os
import sys
import logging

# Add parent directory to path to import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging_utils import setup_logger

# Initialize logger
logger = setup_logger(name="zone_drawer")

# Global list to store the coordinates you click
current_polygon = []

def mouse_callback(event, x, y, flags, param):
    global current_polygon
    frame_copy = param['frame'].copy()

    # If Left Mouse Button is clicked, record the (x,y) coordinate
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append([x, y])
        logger.info(f"Point recorded: [{x}, {y}]")

    # Draw the points and lines so you can see what you are doing
    if len(current_polygon) > 0:
        for pt in current_polygon:
            cv2.circle(frame_copy, tuple(pt), 5, (0, 255, 0), -1)

    if len(current_polygon) > 1:
        pts = np.array(current_polygon, np.int32)
        cv2.polylines(frame_copy, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    cv2.imshow("Zone Drawer", frame_copy)

def main(video_path="./samples/pexels_videos_2740 (1080p).mp4"):
    # 1. Update this path to your actual test video
    video_path = video_path
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read video frame. Check the path.")
        return

    logger.info("="*40)
    logger.info(" 🛠️ ZONE DRAWER TOOL 🛠️")
    logger.info("="*40)
    logger.info("1. Left Click to draw the corners of your zone.")
    logger.info("2. Press 'c' to clear the current drawing.")
    logger.info("3. Press 'p' to print the YAML coordinates.")
    logger.info("4. Press 'q' to quit.")
    logger.info("="*40)

    # --- THE FIX ---
    window_name = "Zone Drawer"
    
    # Create the window with the scalable flag
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Lock the viewing window to 720p so it fits your laptop
    cv2.resizeWindow(window_name, 1280, 720)

    # Attach our mouse listener to THIS window
    cv2.setMouseCallback(window_name, mouse_callback, param={'frame': frame})
    
    # Show the initial frame
    cv2.imshow(window_name, frame)
    # ---------------

    global current_polygon

    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            current_polygon = []
            cv2.imshow(window_name, frame)
            logger.info("Cleared current points.")
        elif key == ord('p'):
            logger.info("--- COPY THIS INTO YOUR zones.yaml ---")
            logger.info(f"coordinates: {current_polygon}")
            logger.info("--------------------------------------")

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    # Command line argument for video path
    import argparse
    parser = argparse.ArgumentParser(description="Draw zones on a video frame and output coordinates in YAML format.")
    parser.add_argument('--video', type=str, default="./samples/pexels_videos_2740 (1080p).mp4", help='Path to the video file')
    args = parser.parse_args()  
    
    main(video_path=args.video)