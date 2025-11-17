#!/usr/bin/env python3
"""
Webcam / RealSense Hand Detection with MediaPipe
Displays camera stream with hand skeleton and 3D wrist coordinates.
Uses MediaPipe Hands only (no full-body pose).
Press 'q' or ESC to exit.
Press 'z' to set zero position (for future UR3 control).
"""

import os
import argparse
import glob
import math
import numpy as np
import cv2
import time
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import subprocess

# Setup MediaPipe components
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def setup_hand_detector(model_path=None):
    """
    Setup MediaPipe hand detector.
    If model_path is None, tries to find it in GMH-D/MP_model/ directory.
    """
    if model_path is None:
        # Try to find model in GMH-D directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "GMH-D", "MP_model", "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print(f"Warning: Hand landmarker model not found at {model_path}")
            print("Hand detection will be disabled.")
            return None
    
    print(f"Loading MediaPipe hand detector from: {model_path}")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return HandLandmarker.create_from_options(options)


def draw_hand_landmarks(image, detection_result, zero_offset=None):
    """
    Draw hand landmarks on the image and display 3D wrist coordinates.
    
    Args:
        image: BGR image to draw on
        detection_result: MediaPipe hand detection result
        zero_offset: Optional offset to subtract from coordinates (for zeroing)
    
    Returns:
        annotated_image: Image with landmarks and coordinates drawn
        wrist_xyz: Dictionary with wrist 3D coordinates (normalized) or None
    """
    if detection_result is None or not detection_result.hand_landmarks:
        return image, None
    
    annotated_image = np.copy(image)
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    MARGIN = 10
    FONT_SIZE = 0.7
    FONT_THICKNESS = 2
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
    COORD_TEXT_COLOR = (255, 255, 0)  # yellow
    ZERO_TEXT_COLOR = (0, 255, 255)  # cyan
    
    height, width, _ = annotated_image.shape
    wrist_xyz = None
    
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in hand_landmarks
        ])
        
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # Get the top left corner of the detected hand's bounding box for handedness label.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        hand_text_x = int(min(x_coordinates) * width)
        hand_text_y = int(min(y_coordinates) * height) - MARGIN
        
        # Draw handedness (left or right hand) on the image near the hand.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (hand_text_x, hand_text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        
        # Extract 3D coordinates from MediaPipe wrist landmark
        wrist_landmark = hand_landmarks[solutions.hands.HandLandmark.WRIST]
        
        # MediaPipe coordinates (normalized)
        wrist_x = wrist_landmark.x  # Normalized [0, 1]
        wrist_y = wrist_landmark.y  # Normalized [0, 1]
        wrist_z = wrist_landmark.z  # Relative depth (normalized, similar scale to x)
        
        # Calculate average z of all hand landmarks (overall hand depth)
        # Note: wrist z is ~0 since it's the reference point of the hand model.
        all_z_values = [landmark.z for landmark in hand_landmarks]
        avg_hand_z = sum(all_z_values) / len(all_z_values)
        
        # Also get palm center z (middle of palm landmarks).
        # We'll use this as the main depth value for the wrist.
        palm_landmarks = [
            solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
            solutions.hands.HandLandmark.RING_FINGER_MCP,
            solutions.hands.HandLandmark.INDEX_FINGER_MCP,
            solutions.hands.HandLandmark.PINKY_MCP,
        ]
        palm_z_values = [hand_landmarks[idx].z for idx in palm_landmarks]
        avg_palm_z = sum(palm_z_values) / len(palm_z_values) if palm_z_values else 0

        # --- Approximate palm orientation (in camera frame) ---
        # Use vector from wrist to middle finger MCP as "forward" direction of the palm.
        middle_mcp = hand_landmarks[solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        forward_vec = np.array([
            middle_mcp.x - wrist_x,
            middle_mcp.y - wrist_y,
            middle_mcp.z - wrist_z,
        ], dtype=float)

        # Normalize forward vector
        forward_norm = np.linalg.norm(forward_vec)
        if forward_norm > 1e-6:
            forward_vec /= forward_norm
        else:
            forward_vec[:] = 0.0

        # Approximate yaw/pitch of the palm in camera coordinates.
        # MediaPipe's camera coordinates: x right, y down, z into the screen.
        fx, fy, fz = forward_vec
        # Avoid division by zero
        if abs(fz) < 1e-6:
            fz = 1e-6

        # Yaw: rotation left/right around vertical axis (y)
        yaw_rad = math.atan2(fx, -fz)
        # Pitch: rotation up/down around horizontal axis (x)
        pitch_rad = math.atan2(fy, -fz)

        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)
        
        # Apply zero offset if provided
        if zero_offset is not None:
            wrist_x_zeroed = wrist_x - zero_offset['x']
            wrist_y_zeroed = wrist_y - zero_offset['y']
            # Use palm center z as the depth value used for zeroing
            depth_zeroed = avg_palm_z - zero_offset.get('z', 0)
            avg_hand_z_zeroed = avg_hand_z - zero_offset.get('z_avg', avg_hand_z)
            avg_palm_z_zeroed = avg_palm_z - zero_offset.get('z_palm', avg_palm_z)
        else:
            wrist_x_zeroed = wrist_x
            wrist_y_zeroed = wrist_y
            depth_zeroed = avg_palm_z
            avg_hand_z_zeroed = avg_hand_z
            avg_palm_z_zeroed = avg_palm_z
        
        # Store wrist coordinates.
        # IMPORTANT: we use palm center z as the main depth ('z') for control.
        wrist_xyz = {
            'x': wrist_x,
            'y': wrist_y,
            'z': avg_palm_z,        # Main depth: palm center z
            'z_wrist': wrist_z,     # Raw wrist z (close to 0)
            'z_avg': avg_hand_z,    # Average z of all landmarks
            'z_palm': avg_palm_z,   # Average z of palm landmarks (same as main depth)
            'x_zeroed': wrist_x_zeroed,
            'y_zeroed': wrist_y_zeroed,
            'z_zeroed': depth_zeroed,
            'handedness': handedness[0].category_name,
            # Palm orientation (approximate, in degrees)
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
        }
        
        # Convert to pixel coordinates for display
        wrist_pixel_x = int(wrist_x * width)
        wrist_pixel_y = int(wrist_y * height)
        
        # Draw a circle at wrist location
        cv2.circle(annotated_image, (wrist_pixel_x, wrist_pixel_y), 8, (0, 0, 255), -1)
        cv2.circle(annotated_image, (wrist_pixel_x, wrist_pixel_y), 10, (255, 255, 255), 2)
        
        # Fixed position for wrist info in top-left corner (only draw once for first hand)
        if idx == 0:  # Only draw info box for the first detected hand
            info_x = 10  # Fixed X position (left margin)
            info_y_start = 30  # Fixed Y position (top margin)
            line_height = 25
            current_y = info_y_start
            
            # Draw background rectangle for better visibility
            base_height = 260
            bg_height = base_height if zero_offset is None else (base_height + 80)
            cv2.rectangle(annotated_image, (info_x - 5, info_y_start - 20), 
                         (info_x + 250, info_y_start + bg_height), (0, 0, 0), -1)
            cv2.rectangle(annotated_image, (info_x - 5, info_y_start - 20), 
                         (info_x + 250, info_y_start + bg_height), (255, 255, 255), 2)
            
            # Header
            cv2.putText(annotated_image, "Wrist 3D (MediaPipe):",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            # Raw coordinates
            current_y += line_height
            cv2.putText(annotated_image, f"X: {wrist_x:.4f}",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            current_y += line_height
            cv2.putText(annotated_image, f"Y: {wrist_y:.4f}",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            current_y += line_height
            # Display Z coordinate based on palm center (used as wrist depth)
            cv2.putText(annotated_image, f"Z (palm center): {avg_palm_z:.4f}",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            # Show raw wrist z (for reference, usually ~0)
            current_y += line_height
            cv2.putText(annotated_image, f"Z (wrist raw): {wrist_z:.4f}",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            # Show average hand depth
            current_y += line_height
            cv2.putText(annotated_image, f"Z (hand avg): {avg_hand_z:.4f}",
                        (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, COORD_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

            # Show approximate palm orientation
            current_y += line_height
            cv2.putText(
                annotated_image,
                f"Yaw: {yaw_deg:+.1f} deg",
                (info_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COORD_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

            current_y += line_height
            cv2.putText(
                annotated_image,
                f"Pitch: {pitch_deg:+.1f} deg",
                (info_x, current_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SIZE,
                COORD_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )
            
            # Zeroed coordinates (if zero offset is set)
            if zero_offset is not None:
                current_y += line_height + 5
                cv2.putText(annotated_image, "Zeroed (relative):",
                            (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE * 0.9, ZERO_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                
                current_y += line_height
                cv2.putText(annotated_image, f"X: {wrist_x_zeroed:+.4f}",
                            (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE, ZERO_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                
                current_y += line_height
                cv2.putText(annotated_image, f"Y: {wrist_y_zeroed:+.4f}",
                            (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE, ZERO_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
                
                current_y += line_height
                cv2.putText(annotated_image, f"Z (palm depth): {avg_palm_z_zeroed:+.4f}",
                            (info_x, current_y), cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SIZE, ZERO_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    return annotated_image, wrist_xyz


def find_and_kill_camera_processes():
    """Find and kill processes using camera devices."""
    try:
        # Find all video devices
        video_devices = glob.glob('/dev/video*')
        if not video_devices:
            return
        
        # Try lsof for each device
        pids = set()
        for device in video_devices:
            try:
                result = subprocess.run(['lsof', device], 
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                                      text=True)
                if result.returncode == 0 and result.stdout:
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            parts = line.split()
                            if len(parts) > 1:
                                try:
                                    pids.add(int(parts[1]))
                                except ValueError:
                                    pass
            except Exception:
                pass
        
        # Kill all found processes
        for pid in pids:
            try:
                subprocess.run(['kill', '-9', str(pid)], check=False, 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"[DEBUG] Killed process {pid} using camera")
            except Exception:
                pass
    except Exception:
        pass
    
    # Fallback: try fuser
    try:
        for device in glob.glob('/dev/video*'):
            try:
                subprocess.run(['fuser', '-k', device], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                              check=False)
            except Exception:
                pass
    except Exception:
        pass
    
    # Also try to kill any realsense processes
    try:
        result = subprocess.run(['pgrep', '-f', 'realsense'], 
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, 
                              text=True)
        if result.returncode == 0 and result.stdout:
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    try:
                        subprocess.run(['kill', '-9', pid.strip()], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                      check=False)
                        print(f"[DEBUG] Killed realsense process {pid.strip()}")
                    except Exception:
                        pass
    except Exception:
        pass


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Hand detection with MediaPipe. Use webcam (0, 1, 2, ...) or RealSense.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python realsense_hand_detection.py 0          # Use webcam index 0
  python realsense_hand_detection.py 1          # Use webcam index 1
  python realsense_hand_detection.py realsense  # Use Intel RealSense camera
        '''
    )
    parser.add_argument('camera', type=str, nargs='?', default='1',
                        help='Camera source: webcam index (0, 1, 2, ...) or "realsense" (default: 1)')
    args = parser.parse_args()
    
    camera_source = args.camera.lower()
    use_realsense = (camera_source == 'realsense')
    
    # Initialize camera source
    cap = None
    pipeline = None
    align_to = None
    
    if use_realsense:
        try:
            import pyrealsense2 as rs
            print("[DEBUG] Initializing RealSense camera...")
            
            # Find and kill any processes using the camera
            find_and_kill_camera_processes()
            time.sleep(0.5)  # Give processes time to release
            
            # Configure RealSense pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable both streams (D435i requires both)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"[DEBUG] Starting RealSense pipeline (attempt {attempt + 1}/{max_retries})...")
                    pipeline.start(config)
                    print("[DEBUG] RealSense pipeline started successfully")
                    break
                except RuntimeError as e:
                    if "busy" in str(e).lower() or "resource" in str(e).lower():
                        print(f"[DEBUG] Camera busy, attempting to free it...")
                        find_and_kill_camera_processes()
                        time.sleep(1)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise
            
            # Wait for frames to stabilize
            print("[DEBUG] Waiting for RealSense to stabilize...")
            time.sleep(1)
            
            # Create align object to align depth frames to color frames
            align_to = rs.align(rs.stream.color)
            
            print("RealSense camera opened successfully.")
        except ImportError:
            print("Error: pyrealsense2 not installed. Install it with: pip install pyrealsense2")
            return
        except Exception as e:
            print(f"Error: Could not open RealSense camera: {e}")
            return
    else:
        # Use webcam
        try:
            camera_index = int(camera_source)
        except ValueError:
            print(f"Error: Invalid camera source '{camera_source}'. Use a number (0, 1, 2, ...) or 'realsense'")
            return
        
        print(f"Opening webcam at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam opened successfully.")
    
    # Setup MediaPipe hand detector
    print("Setting up MediaPipe hand detector...")
    detector = setup_hand_detector()
    if detector is None:
        print("Running without hand detection...")
    else:
        print("Hand detection enabled.")
    
    camera_type = "RealSense" if use_realsense else "Webcam"
    print(f"{camera_type} stream started.")
    print("Controls:")
    print("  'q' or ESC - Exit")
    print("  'z' - Set zero position for wrist coordinates")
    
    # Zero offset for wrist coordinates
    zero_offset = None

    try:
        while True:
            # Read frame from camera
            if use_realsense:
                # RealSense: wait for frames
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = align_to.process(frames)
                color_frame = aligned_frames.get_color_frame()
                
                if not color_frame:
                    print("Warning: Failed to get color frame from RealSense")
                    time.sleep(0.1)
                    continue
                
                color_image = np.asanyarray(color_frame.get_data())
            else:
                # Webcam: read frame
                ret, color_image = cap.read()
                
                if not ret or color_image is None:
                    print("Warning: Failed to read frame from webcam")
                    time.sleep(0.1)
                    continue

            # Detect hands if detector is available
            wrist_xyz = None
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Detect hand landmarks
            if detector is not None:
                detection_result = detector.detect(mp_image)
                
                # Draw hand landmarks on color image with 3D coordinates
                color_image, wrist_xyz = draw_hand_landmarks(color_image, detection_result, zero_offset)
                
                # Print wrist coordinates to console
                if wrist_xyz is not None:
                    if zero_offset is not None:
                        print(f"Wrist XYZ (zeroed): [{wrist_xyz['x_zeroed']:+.4f}, {wrist_xyz['y_zeroed']:+.4f}, {wrist_xyz['z_zeroed']:+.4f}]")
                    else:
                        print(f"Wrist XYZ: [{wrist_xyz['x']:.4f}, {wrist_xyz['y']:.4f}, {wrist_xyz['z']:.4f}]")

            # Show images
            window_title = f'{camera_type} - Hand Detection'
            cv2.imshow(window_title, color_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('z'):
                # Set zero position
                if wrist_xyz is not None:
                    zero_offset = {
                        'x': wrist_xyz['x'],
                        'y': wrist_xyz['y'],
                        'z': wrist_xyz['z'],
                        'z_avg': wrist_xyz.get('z_avg', 0),
                        'z_palm': wrist_xyz.get('z_palm', 0)
                    }
                    print(f"Zero position set:")
                    print(f"  X: {zero_offset['x']:.4f}, Y: {zero_offset['y']:.4f}, Z: {zero_offset['z']:.4f}")
                    print(f"  Z (avg): {zero_offset['z_avg']:.4f}, Z (palm): {zero_offset['z_palm']:.4f}")
                else:
                    print("No hand detected. Cannot set zero position.")
    finally:
        # Stop streaming
        camera_type = "RealSense" if use_realsense else "Webcam"
        print(f"Stopping {camera_type.lower()}...")
        if use_realsense:
            if pipeline:
                pipeline.stop()
        else:
            if cap:
                cap.release()
        cv2.destroyAllWindows()
        print(f"{camera_type} stream stopped.")


if __name__ == "__main__":
    main()

