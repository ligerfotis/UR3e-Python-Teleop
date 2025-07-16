import cv2
import time
import os

def camera_capture(root_dir, duration, camera_index, output_queue):
    # Subdirectory for camera data
    cam_dir = os.path.join(root_dir, "camera")
    os.makedirs(cam_dir, exist_ok=True)

    # Open video stream
    cap = cv2.VideoCapture(camera_index)

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # fallback default FPS

    # Define video writer
    video_path = os.path.join(cam_dir, "camera_output.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    frame_count = 0
    start = time.time()
    print("Camera capture started.")

    while time.time() - start < duration:
        ret, frame = cap.read()  # Read frame
        if not ret:
            continue

        output_queue.put(frame)  # Put frame in queue for live preview
        out.write(frame)         # Write frame to video
        frame_count += 1

    cap.release()
    out.release()
    print(f"Camera capture complete. {frame_count} frames saved to {root_dir}.")
