"""Function to record camera data as AVI video"""
"""Intended to be used with sensor_manager.py for live preview of camera"""

import cv2
import time
import os

def camera_capture(root_dir, duration, camera_index, output_queue):
    """
    Args:
        root_dir: Root directory where video is stored
        duration: Duration of camera data capture
        camera_index: Camera index, detected automatically in sensor_manager.py
        output_queue: Queue to store camera frames for live preview

    Returns:
        Video of captured camera frames within the specified duration
        Live preview of video (when used in conjunction with sensor_manager.py)
    """

    # Open video stream
    cap = cv2.VideoCapture(camera_index)

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Default FPS

    # Define filepath
    video_path = os.path.join(root_dir, "camera_output.avi") # AVI format

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Start camera capture
    frame_count = 0
    start = time.time()
    print("Camera capture started.")

    # Capture video within specified duration
    while time.time() - start < duration:
        ret, frame = cap.read()  # Read frame
        if not ret:
            continue # Continue even if frame not captured

        # Output frames
        output_queue.put(frame) # Put frame in queue for live preview
        out.write(frame) # Write frame to video
        frame_count += 1

    # Close camera
    cap.release() # Close live preview
    out.release() # Close frame writing
    print(f"Camera capture complete. {frame_count} frames saved to {root_dir}.")
