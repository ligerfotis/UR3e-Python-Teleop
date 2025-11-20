"""Hand‑based teleoperation for UR3e using weighted DLS (no PS5 controller).

This script:
- Uses a webcam (or other OpenCV camera index) plus MediaPipe Hands to track a single hand.
- Extracts wrist / palm 3D information in the camera frame:
    * x, y: normalized image coordinates in [0, 1]
    * z: relative depth based on palm center (average MCP joints)
    * roll, pitch, yaw: approximate palm orientation in the camera frame
- Lets you set a "zero" hand pose (reference) with the 'z' key.
- Maps the *offset* from this zero pose -> desired TCP velocity (vx, vy, vz, wx, wy, wz).
- Uses the same weighted, damped least‑squares (DLS) solver as `teleop_ps5_wdls.py` to compute qdot.
- Starts in a SAFE state: no commands are sent to the robot until you explicitly enable teleop.

Controls:
- Move your hand around the camera to command TCP motion (when teleop is enabled).
- Press 'z' to set the current hand pose as zero (no motion).
- Press 't' to toggle teleop on/off (like a torque enable).
- Press 'q' or ESC to quit.

NOTE:
- The mapping from camera axes to robot base axes is approximate and may
  require flipping signs or gain tuning for your setup.
"""

import os
import math
import time
import argparse
from typing import Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


# ------------------------- Robot & control configuration -------------------------

robot_ip = "192.168.1.201"  # Set your robot IP

# Control loop period (approximate)
CYCLE_TIME = 0.02  # [s] ~50 Hz target

# Linear / angular velocity magnitudes for user commands (max)
LINEAR_SPEED_MAX = 0.08   # [m/s]
ANGULAR_SPEED_MAX = 0.5   # [rad/s]

# Mapping gains from normalized hand offsets to TCP velocities
# (These are rough initial values; tune for your workspace and camera placement.)
HAND_POS_DEADZONE = 0.01          # normalized units
HAND_ANG_DEADZONE_RAD = math.radians(3.0)  # ~3 deg

# Simple exponential smoothing for hand offsets (to reduce jitter)
# new_filtered = (1 - ALPHA) * prev_filtered + ALPHA * current
POS_FILTER_ALPHA = 0.3   # 0..1, higher = more responsive, lower = smoother
ANG_FILTER_ALPHA = 0.3   # kept for future use if you re-enable rotations

HAND_GAIN_X = LINEAR_SPEED_MAX          # [m/s] per full-scale offset in x
HAND_GAIN_Y = LINEAR_SPEED_MAX          # [m/s] per full-scale offset in y
# Make Z motion one order of magnitude more sensitive than X/Y
HAND_GAIN_Z = 10.0 * LINEAR_SPEED_MAX   # [m/s] per full-scale offset in z

HAND_GAIN_ROLL = ANGULAR_SPEED_MAX   # [rad/s] per rad roll offset
HAND_GAIN_PITCH = ANGULAR_SPEED_MAX  # [rad/s] per rad pitch offset
HAND_GAIN_YAW = ANGULAR_SPEED_MAX    # [rad/s] per rad yaw offset

# speedJ parameters
JOINT_ACCEL = 1.0  # [rad/s^2]

# Joint limits for UR3e (conservative, radians)
JOINT_LIMITS_MIN = np.deg2rad(np.array([-360, -360, -360, -360, -360, -360]))
JOINT_LIMITS_MAX = np.deg2rad(np.array([360, 360, 360, 360, 360, 360]))

# Workspace limits (simple box around the robot base) in base frame [m]
WORKSPACE_MIN = np.array([-0.60, -0.60, 0.23])
WORKSPACE_MAX = np.array([0.60, 0.60, 0.65])

# Preferred joint posture (comfortable pose), radians
Q_PREFERRED = np.deg2rad(
    np.array(
        [
            0.0,  # joint 1
            0.0,  # joint 2
            0.0,  # joint 3
            0.0,  # joint 4
            0.0,  # joint 5
            0.0,  # joint 6
        ]
    )
)

# Nullspace posture strength
NULLSPACE_GAIN = 0.05

# Damping parameters for manipulability-based DLS
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.05
MANIP_HIGH = 0.02
MANIP_LOW = 0.003

# Task weighting: same as PS5 teleop example
TASK_WEIGHTS = np.diag([1.0, 1.0, 1.0, 0.6, 0.6, 0.1])


# ------------------------- UR3e kinematics & Jacobian -------------------------

UR3E_A = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
UR3E_D = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
UR3E_ALPHA = np.array(
    [np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0]
)


def _dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH transform."""
    sa, ca = np.sin(alpha), np.cos(alpha)
    st, ct = np.sin(theta), np.cos(theta)

    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_ur3e(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward kinematics for UR3e."""
    T = np.eye(4)
    origins = [T[:3, 3].copy()]
    z_axes = [T[:3, 2].copy()]

    for i in range(6):
        Ti = _dh_transform(UR3E_A[i], UR3E_ALPHA[i], UR3E_D[i], q[i])
        T = T @ Ti
        origins.append(T[:3, 3].copy())
        z_axes.append(T[:3, 2].copy())

    return T, np.vstack(origins), np.vstack(z_axes)


def jacobian_ur3e(q: np.ndarray) -> np.ndarray:
    """Geometric Jacobian (6x6) at TCP for UR3e in base frame."""
    _, origins, z_axes = fk_ur3e(q)
    p_e = origins[-1]

    J = np.zeros((6, 6))
    for i in range(6):
        z = z_axes[i]
        p = origins[i]
        J[0:3, i] = np.cross(z, p_e - p)
        J[3:6, i] = z

    return J


def manipulability(J: np.ndarray) -> float:
    JJ = J @ J.T
    try:
        return float(np.sqrt(np.linalg.det(JJ)))
    except np.linalg.LinAlgError:
        return 0.0


def damping_from_manip(w: float) -> float:
    if w >= MANIP_HIGH:
        return LAMBDA_MIN
    if w <= MANIP_LOW:
        return LAMBDA_MAX
    t = (w - MANIP_LOW) / (MANIP_HIGH - MANIP_LOW)
    return LAMBDA_MAX + (LAMBDA_MIN - LAMBDA_MAX) * t


def weighted_dls_joint_velocity(J: np.ndarray, v_tcp: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Weighted damped least‑squares joint velocities with nullspace posture bias."""
    w = manipulability(J)
    lam = damping_from_manip(w)

    J_w = TASK_WEIGHTS @ J
    v_w = TASK_WEIGHTS @ v_tcp

    JJt_w = J_w @ J_w.T
    lam2I = (lam ** 2) * np.eye(6)
    try:
        inv_term = np.linalg.inv(JJt_w + lam2I)
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(JJt_w + lam2I)

    J_dls_w = J_w.T @ inv_term

    qdot_task = J_dls_w @ v_w

    # Nullspace term: mild bias toward preferred posture
    q_err = Q_PREFERRED - q
    qdot_null = NULLSPACE_GAIN * q_err
    N = np.eye(6) - J_dls_w @ J_w

    return qdot_task + N @ qdot_null


def enforce_joint_limits(q: np.ndarray, qdot: np.ndarray, dt: float) -> np.ndarray:
    q_next = q + qdot * dt
    scale = 1.0
    for i in range(6):
        if q_next[i] < JOINT_LIMITS_MIN[i] and qdot[i] < 0:
            max_step = JOINT_LIMITS_MIN[i] - q[i]
            if qdot[i] != 0:
                scale_i = max_step / (qdot[i] * dt)
                scale = min(scale, max(0.0, scale_i))
        elif q_next[i] > JOINT_LIMITS_MAX[i] and qdot[i] > 0:
            max_step = JOINT_LIMITS_MAX[i] - q[i]
            if qdot[i] != 0:
                scale_i = max_step / (qdot[i] * dt)
                scale = min(scale, max(0.0, scale_i))

    return qdot * scale


def enforce_workspace_limits(tcp_pose: np.ndarray, v_tcp: np.ndarray, dt: float) -> np.ndarray:
    v = v_tcp.copy()
    p = tcp_pose[:3]
    p_next = p + v[:3] * dt

    for i in range(3):
        if p_next[i] < WORKSPACE_MIN[i] and v[i] < 0.0:
            v[i] = 0.0
        elif p_next[i] > WORKSPACE_MAX[i] and v[i] > 0.0:
            v[i] = 0.0

    return v


# ------------------------- MediaPipe Hands setup -------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def setup_hand_detector(model_path: Optional[str] = None):
    """Setup MediaPipe hand detector. Uses GMH-D model if available."""
    if model_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "GMH-D", "MP_model", "hand_landmarker.task")

    if not os.path.exists(model_path):
        print(f"Warning: Hand landmarker model not found at {model_path}")
        print("Hand teleop will be disabled.")
        return None

    print(f"Loading MediaPipe hand detector from: {model_path}")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return HandLandmarker.create_from_options(options)


def extract_hand_pose(
    color_image: np.ndarray,
    detector,
    zero_offset: Optional[dict],
) -> Tuple[np.ndarray, Optional[dict]]:
    """Run hand detection and compute wrist position + palm orientation."""
    if detector is None:
        return color_image, None

    height, width, _ = color_image.shape

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    detection_result = detector.detect(mp_image)
    if detection_result is None or not detection_result.hand_landmarks:
        return color_image, None

    # Select only the right hand (ignore any detected left hands)
    target_idx = None
    for idx, handedness in enumerate(detection_result.handedness):
        if handedness and handedness[0].category_name.lower() == "right":
            target_idx = idx
            break

    if target_idx is None:
        # Right hand not found in this frame
        return color_image, None

    hand_landmarks = detection_result.hand_landmarks[target_idx]

    # Build proto and draw skeleton
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend(
        [
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ]
    )
    solutions.drawing_utils.draw_landmarks(
        color_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style(),
    )

    # Wrist and palm landmarks
    wrist = hand_landmarks[solutions.hands.HandLandmark.WRIST]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    palm_indices = [
        solutions.hands.HandLandmark.MIDDLE_FINGER_MCP,
        solutions.hands.HandLandmark.RING_FINGER_MCP,
        solutions.hands.HandLandmark.INDEX_FINGER_MCP,
        solutions.hands.HandLandmark.PINKY_MCP,
    ]
    palm_z_values = [hand_landmarks[idx].z for idx in palm_indices]
    avg_palm_z = float(sum(palm_z_values) / len(palm_z_values)) if palm_z_values else 0.0

    # Average hand depth
    all_z_values = [lm.z for lm in hand_landmarks]
    avg_hand_z = float(sum(all_z_values) / len(all_z_values))

    # Orientation: forward (wrist -> middle MCP) and side (index MCP -> pinky MCP)
    middle_mcp = hand_landmarks[solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_mcp = hand_landmarks[solutions.hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks[solutions.hands.HandLandmark.PINKY_MCP]

    forward_vec = np.array(
        [middle_mcp.x - wrist_x, middle_mcp.y - wrist_y, middle_mcp.z - wrist_z],
        dtype=float,
    )
    side_vec = np.array(
        [index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y, index_mcp.z - pinky_mcp.z],
        dtype=float,
    )

    def _safe_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-6:
            return np.zeros_like(v)
        return v / n

    forward_vec = _safe_normalize(forward_vec)
    side_vec = _safe_normalize(side_vec)

    fx, fy, fz = forward_vec
    sx, sy, sz = side_vec

    if abs(fz) < 1e-6:
        fz = 1e-6

    # Approximate yaw/pitch relative to camera
    yaw_rad = math.atan2(fx, -fz)
    pitch_rad = math.atan2(fy, -fz)

    # Approximate roll from side vector in image plane
    roll_rad = math.atan2(sy, sx)

    yaw_deg = math.degrees(yaw_rad)
    pitch_deg = math.degrees(pitch_rad)
    roll_deg = math.degrees(roll_rad)

    # Zeroing
    if zero_offset is not None:
        dx = wrist_x - zero_offset["x"]
        dy = wrist_y - zero_offset["y"]
        dz = avg_palm_z - zero_offset["z"]
        dyaw = yaw_rad - zero_offset["yaw_rad"]
        dpitch = pitch_rad - zero_offset["pitch_rad"]
        droll = roll_rad - zero_offset["roll_rad"]
    else:
        dx = wrist_x
        dy = wrist_y
        dz = avg_palm_z
        dyaw = yaw_rad
        dpitch = pitch_rad
        droll = roll_rad

    # Draw a small HUD
    info_x, info_y = 10, 30
    line_h = 22
    hud_lines = [
        f"X: {wrist_x:.3f}",
        f"Y: {wrist_y:.3f}",
        f"Z (palm): {avg_palm_z:.3f}",
        f"Roll: {roll_deg:+.1f}",
        f"Pitch: {pitch_deg:+.1f}",
        f"Yaw: {yaw_deg:+.1f}",
        f"dX: {dx:+.3f}",
        f"dY: {dy:+.3f}",
        f"dZ: {dz:+.3f}",
    ]

    cv2.rectangle(
        color_image,
        (info_x - 5, info_y - 20),
        (info_x + 260, info_y + line_h * (len(hud_lines) + 1)),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        color_image,
        (info_x - 5, info_y - 20),
        (info_x + 260, info_y + line_h * (len(hud_lines) + 1)),
        (255, 255, 255),
        1,
    )

    y = info_y
    for text in hud_lines:
        cv2.putText(
            color_image,
            text,
            (info_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += line_h

    wrist_xyz = {
        "x": wrist_x,
        "y": wrist_y,
        "z": avg_palm_z,
        "z_hand_avg": avg_hand_z,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "droll": droll,
        "dpitch": dpitch,
        "dyaw": dyaw,
    }

    return color_image, wrist_xyz


def apply_deadzone(value: float, threshold: float) -> float:
    if abs(value) < threshold:
        return 0.0
    return value


def main():
    # Parse optional camera index argument
    parser = argparse.ArgumentParser(
        description="Hand-based UR3e teleop (weighted DLS) using webcam.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "camera",
        type=int,
        nargs="?",
        default=1,
        help="Webcam index to use (default: 1).",
    )
    args = parser.parse_args()
    camera_index = args.camera

    # Robot interfaces
    rtde_control = RTDEControlInterface(robot_ip)
    rtde_receive = RTDEReceiveInterface(robot_ip)

    print(
        "Hand-based UR3e teleop (weighted DLS).\n"
        "SAFETY: teleop starts DISABLED (no robot motion).\n"
        "Controls:\n"
        "  - 'z': set current hand pose as zero.\n"
        "  - 't': toggle teleop ON/OFF.\n"
        "  - 'q' or ESC: quit.\n"
    )

    # Camera setup (webcam index can be provided via CLI)
    print(f"Opening webcam at index {camera_index} ...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {camera_index}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Hand detector
    detector = setup_hand_detector()
    if detector is None:
        print("No hand detector available. Exiting.")
        return

    zero_offset = None
    teleop_enabled = False

    # Filter state for hand offsets (dx, dy, dz)
    filtered_dx = 0.0
    filtered_dy = 0.0
    filtered_dz = 0.0
    filter_initialized = False

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: Failed to read frame from camera")
                time.sleep(0.01)
                continue

            frame, wrist_xyz = extract_hand_pose(frame, detector, zero_offset)

            # Default: no motion
            v_tcp = np.zeros(6, dtype=float)

            if wrist_xyz is not None:
                # Raw offsets from zero (normalized units)
                dx_raw = wrist_xyz["dx"]
                dy_raw = wrist_xyz["dy"]
                dz_raw = wrist_xyz["dz"]

                # Initialize filter with first measurement
                if not filter_initialized:
                    filtered_dx = dx_raw
                    filtered_dy = dy_raw
                    filtered_dz = dz_raw
                    filter_initialized = True
                else:
                    # Exponential smoothing to reduce jitter
                    filtered_dx = (1.0 - POS_FILTER_ALPHA) * filtered_dx + POS_FILTER_ALPHA * dx_raw
                    filtered_dy = (1.0 - POS_FILTER_ALPHA) * filtered_dy + POS_FILTER_ALPHA * dy_raw
                    filtered_dz = (1.0 - POS_FILTER_ALPHA) * filtered_dz + POS_FILTER_ALPHA * dz_raw

                # Apply deadzone to filtered offsets
                dx = apply_deadzone(filtered_dx, HAND_POS_DEADZONE)
                dy = apply_deadzone(filtered_dy, HAND_POS_DEADZONE)
                dz = apply_deadzone(filtered_dz, HAND_POS_DEADZONE)

                # Rotations are currently disabled: only XYZ motion is used.
                # We still compute orientation offsets for display/debugging,
                # but we do not map them into robot angular velocities.

                # Map normalized position offsets -> TCP linear velocities
                # Flip signs for X and Z as requested.
                vx = HAND_GAIN_X * dx
                vy = HAND_GAIN_Y * dy
                vz = HAND_GAIN_Z * dz

                # No rotational commands
                wx = 0.0
                wy = 0.0
                wz = 0.0

                v_tcp = np.array([vx, vy, vz, wx, wy, wz], dtype=float)

            # Keyboard controls
            cv2.imshow("Hand Teleop", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            if key == ord("z") and wrist_xyz is not None:
                zero_offset = {
                    "x": wrist_xyz["x"],
                    "y": wrist_xyz["y"],
                    "z": wrist_xyz["z"],
                    "roll_rad": wrist_xyz["roll_rad"],
                    "pitch_rad": wrist_xyz["pitch_rad"],
                    "yaw_rad": wrist_xyz["yaw_rad"],
                }
                print("Zero pose updated.")

            if key == ord("t"):
                teleop_enabled = not teleop_enabled
                state = "ENABLED" if teleop_enabled else "DISABLED"
                print(f"Teleop {state} (toggled).")

            # Send commands to robot
            if not teleop_enabled:
                # Teleop disabled: always send zero velocity for safety.
                rtde_control.speedJ([0.0] * 6, JOINT_ACCEL, CYCLE_TIME)
            else:
                # Teleop enabled
                if wrist_xyz is None or np.allclose(v_tcp, 0.0, atol=1e-4):
                    rtde_control.speedJ([0.0] * 6, JOINT_ACCEL, CYCLE_TIME)
                else:
                    q = np.array(rtde_receive.getActualQ())
                    tcp_pose = np.array(rtde_receive.getActualTCPPose())

                    v_tcp = enforce_workspace_limits(tcp_pose, v_tcp, CYCLE_TIME)

                    J = jacobian_ur3e(q)
                    qdot = weighted_dls_joint_velocity(J, v_tcp, q)
                    qdot = enforce_joint_limits(q, qdot, CYCLE_TIME)

                    rtde_control.speedJ(qdot.tolist(), JOINT_ACCEL, CYCLE_TIME)

            elapsed = time.time() - loop_start
            sleep_time = CYCLE_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        rtde_control.stopScript()
        cap.release()
        cv2.destroyAllWindows()
        print("Hand teleop stopped.")


if __name__ == "__main__":
    main()


