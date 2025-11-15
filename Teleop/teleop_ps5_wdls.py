"""PS5 teleoperation for UR3e using weighted, task‑relaxed damped least‑squares (DLS).

External user interface is unchanged:
- PS5 controller inputs map to a desired TCP velocity (vx, vy, vz, wx, wy, wz).

Internally, this script:
- Reads current joint configuration q and TCP pose from the robot.
- Computes the UR3e geometric Jacobian J(q) at the TCP.
- Uses a weighted, damped least‑squares mapping:
    * Position and key orientation axes are high‑priority (hard task).
    * Some orientation DOFs are soft (low weight) and can be sacrificed.
- Damping is based on manipulability, increasing near singularities to keep qdot bounded.
- A nullspace posture term biases joints toward a comfortable configuration.
- Joint and workspace limits are enforced.
- Only joint velocity commands (speedJ) are sent via RTDE.

Gripper control is optional; the script continues if no gripper is present.
"""

import time
from typing import Tuple

import numpy as np
import pygame
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robotiq_socket_gripper import RobotiqSocketGripper


# ------------------------- Robot & control configuration -------------------------

robot_ip = "192.168.1.201"  # Fill with your robot's IP

# Control loop period
CYCLE_TIME = 0.01  # [s] ~100 Hz

# Linear / angular velocity magnitudes for user commands
LINEAR_SPEED = 0.05  # [m/s]
ANGULAR_SPEED = 0.25  # [rad/s]

# speedJ parameters
JOINT_ACCEL = 1.0  # [rad/s^2]

# Joint limits for UR3e (conservative, radians)
JOINT_LIMITS_MIN = np.deg2rad(np.array([-360, -360, -360, -360, -360, -360]))
JOINT_LIMITS_MAX = np.deg2rad(np.array([360, 360, 360, 360, 360, 360]))

# Workspace limits (simple box around the robot base) in base frame [m]
# Slightly relaxed to allow easier left/right crossing while remaining safe.
WORKSPACE_MIN = np.array([-0.60, -0.60, 0.05])  # x, y, z
WORKSPACE_MAX = np.array([0.60, 0.60, 0.65])

# Preferred joint posture (comfortable "bent" pose), radians
Q_PREFERRED = np.deg2rad(
    np.array(
        [
            0.0,   # joint 1
            0.0, # joint 2
            0.0, # joint 3
            0.0, # joint 4
            00.0,  # joint 5
            0.0,   # joint 6
        ]
    )
)

# Nullspace posture strength (mild bias)
# Lower gain so posture bias cannot completely block large Cartesian moves.
NULLSPACE_GAIN = 0.05

# Damping parameters for manipulability-based DLS
# Slightly reduced max damping and narrower "strong damping" region so motion
# does not freeze when crossing through lower-manipulability areas.
LAMBDA_MIN = 0.01
LAMBDA_MAX = 0.05
MANIP_HIGH = 0.02   # above this, very low damping
MANIP_LOW = 0.003   # below this, stronger damping

# Task weighting:
#   High weights for position (x,y,z),
#   medium weights for two orientation axes,
#   low weight for soft orientation DOF (e.g. rotation about tool Z).
TASK_WEIGHTS = np.diag([1.0, 1.0, 1.0, 0.6, 0.6, 0.1])


# ------------------------- UR3e kinematics & Jacobian -------------------------

# UR3e modified DH parameters (commonly used set)
# a: link lengths, d: link offsets, alpha: twist angles
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
    """Forward kinematics for UR3e.

    Returns:
        T_0_6: 4x4 transform from base to TCP.
        origins: (7, 3) origins of each frame (0..6) in base frame.
        z_axes: (7, 3) z-axes of each frame (0..6) in base frame.
    """
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
        J[0:3, i] = np.cross(z, p_e - p)  # linear velocity part
        J[3:6, i] = z                     # angular velocity part

    return J


def manipulability(J: np.ndarray) -> float:
    """Yoshikawa manipulability measure."""
    JJ = J @ J.T
    try:
        return float(np.sqrt(np.linalg.det(JJ)))
    except np.linalg.LinAlgError:
        return 0.0


def damping_from_manip(w: float) -> float:
    """Choose damping factor lambda based on manipulability w."""
    if w >= MANIP_HIGH:
        return LAMBDA_MIN
    if w <= MANIP_LOW:
        return LAMBDA_MAX
    # Linear interpolation between LAMBDA_MAX (low w) and LAMBDA_MIN (high w)
    t = (w - MANIP_LOW) / (MANIP_HIGH - MANIP_LOW)
    return LAMBDA_MAX + (LAMBDA_MIN - LAMBDA_MAX) * t


def weighted_dls_joint_velocity(
    J: np.ndarray, v_tcp: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Weighted damped least‑squares joint velocities with nullspace posture bias.

    We solve:
        min || W (J qdot - v_tcp) ||^2 + lambda^2 ||qdot||^2

    using standard DLS on the weighted system J_w = W J, v_w = W v_tcp.
    """
    w = manipulability(J)
    lam = damping_from_manip(w)

    # Weighted system
    J_w = TASK_WEIGHTS @ J
    v_w = TASK_WEIGHTS @ v_tcp

    JJt_w = J_w @ J_w.T
    lam2I = (lam ** 2) * np.eye(6)
    try:
        inv_term = np.linalg.inv(JJt_w + lam2I)
    except np.linalg.LinAlgError:
        inv_term = np.linalg.pinv(JJt_w + lam2I)

    J_dls_w = J_w.T @ inv_term  # 6x6

    # Task-space joint velocity
    qdot_task = J_dls_w @ v_w

    # Nullspace term: mild bias toward preferred posture
    q_err = Q_PREFERRED - q
    qdot_null = NULLSPACE_GAIN * q_err
    N = np.eye(6) - J_dls_w @ J_w

    qdot = qdot_task + N @ qdot_null
    return qdot


def enforce_joint_limits(q: np.ndarray, qdot: np.ndarray, dt: float) -> np.ndarray:
    """Scale joint velocities to keep predicted next q within limits."""
    q_next = q + qdot * dt
    scale = 1.0
    for i in range(6):
        if q_next[i] < JOINT_LIMITS_MIN[i]:
            if qdot[i] < 0:
                max_step = JOINT_LIMITS_MIN[i] - q[i]
                if qdot[i] != 0:
                    scale_i = max_step / (qdot[i] * dt)
                    scale = min(scale, max(0.0, scale_i))
        elif q_next[i] > JOINT_LIMITS_MAX[i]:
            if qdot[i] > 0:
                max_step = JOINT_LIMITS_MAX[i] - q[i]
                if qdot[i] != 0:
                    scale_i = max_step / (qdot[i] * dt)
                    scale = min(scale, max(0.0, scale_i))

    return qdot * scale


def enforce_workspace_limits(
    tcp_pose: np.ndarray, v_tcp: np.ndarray, dt: float
) -> np.ndarray:
    """Clamp TCP velocity so next pose stays within a simple box."""
    v = v_tcp.copy()
    p = tcp_pose[:3]
    p_next = p + v[:3] * dt

    for i in range(3):
        if p_next[i] < WORKSPACE_MIN[i] and v[i] < 0.0:
            v[i] = 0.0
        elif p_next[i] > WORKSPACE_MAX[i] and v[i] > 0.0:
            v[i] = 0.0

    return v


# ------------------------- Gripper and controller setup -------------------------

rtde_control = RTDEControlInterface(robot_ip)
rtde_receive = RTDEReceiveInterface(robot_ip)

gripper = None
gripper_enabled = False
try:
    gripper = RobotiqSocketGripper(robot_ip)
    gripper.connect()
    gripper_enabled = True
except OSError as exc:
    print(f"Warning: Gripper unavailable ({exc}). Continuing without gripper control.")
    gripper = None

# Gripper parameters
GRIPPER_STEP = 1  # Increment for gripper movement
if gripper_enabled:
    gripper.set_force(20)  # Range: 0 (min) to 255 (max)
    gripper.set_speed(50)  # Range: 0 (min) to 255 (max)
    current_pos = gripper.get_pos() or 0  # Initialize gripper position
else:
    current_pos = 0


print(
    """Weighted DLS joint-space control using PS5 controller (TCP-level teleop).
Press 'Option' button (three horizontal lines, right of controller) to quit.

LEFT JOYSTICK: Translate right/left, forward/backward
L1 / L2:       Translate up/down
RIGHT JOYSTICK: Rotate about x-axis, rotate about y-axis
R1 / R2:        Rotate about z-axis (vertical axis, softer)
SQUARE: Close gripper
CIRCLE: Open gripper"""
)


# ------------------------- PS5 controller setup -------------------------

pygame.joystick.init()
pygame.display.init()
joy = pygame.joystick.Joystick(0)
joy.init()


def rescale_axis(axis_position: float, axis_ini_position: float) -> float:
    """Rescale joystick axis around its resting position."""
    low_segment_len = 1.0 + axis_ini_position
    upper_segment_len = 1.0 - axis_ini_position

    if axis_position < axis_ini_position:
        return (axis_position - axis_ini_position) / low_segment_len
    return (axis_position - axis_ini_position) / upper_segment_len


def apply_deadzone(value: float, deadzone: float = 0.1) -> float:
    """Zero small joystick values to avoid drift."""
    if abs(value) < deadzone:
        return 0.0
    return value


# Update inputs once before reading initial axes
pygame.event.pump()

# Get initial resting positions
axis0_ini = joy.get_axis(0)  # Left stick left/right
axis1_ini = joy.get_axis(1)  # Left stick up/down
axis2_ini = joy.get_axis(3)  # Right stick left/right
axis3_ini = joy.get_axis(4)  # Right stick up/down


try:
    while True:
        loop_start = time.time()

        # Quit with Options button
        if joy.get_button(9):
            break

        # Update inputs
        pygame.event.pump()

        # Rescale axes
        axis0 = rescale_axis(joy.get_axis(0), axis0_ini)
        axis1 = rescale_axis(joy.get_axis(1), axis1_ini)
        axis2 = rescale_axis(joy.get_axis(3), axis2_ini)
        axis3 = rescale_axis(joy.get_axis(4), axis3_ini)

        # Apply deadzone
        axis0 = apply_deadzone(axis0)
        axis1 = apply_deadzone(axis1)
        axis2 = apply_deadzone(axis2)
        axis3 = apply_deadzone(axis3)

        # Build desired TCP velocity from PS5 inputs
        vx = LINEAR_SPEED * axis0
        vy = LINEAR_SPEED * axis1

        if joy.get_button(6):  # L1 (up)
            vz = LINEAR_SPEED
        elif joy.get_button(4):  # L2 (down)
            vz = -LINEAR_SPEED
        else:
            vz = 0.0

        wx = -ANGULAR_SPEED * axis3  # Rotate up/down
        wy = ANGULAR_SPEED * axis2   # Rotate left/right

        if joy.get_button(7):  # R1 (Rotate left about Z-axis)
            wz = ANGULAR_SPEED
        elif joy.get_button(5):  # R2 (Rotate right about Z-axis)
            wz = -ANGULAR_SPEED
        else:
            wz = 0.0

        v_tcp = np.array([vx, vy, vz, wx, wy, wz])

        # Gripper control
        if gripper_enabled and joy.get_button(3):  # Square (Close gripper)
            current_pos = min(255, current_pos + GRIPPER_STEP)
            gripper.move(current_pos)
        elif gripper_enabled and joy.get_button(1):  # Circle (Open gripper)
            current_pos = max(0, current_pos - GRIPPER_STEP)
            gripper.move(current_pos)

        # If no motion requested, send zero velocity for stability
        if np.allclose(v_tcp, 0.0, atol=1e-4):
            rtde_control.speedJ([0.0] * 6, JOINT_ACCEL, CYCLE_TIME)
        else:
            # Read current state
            q = np.array(rtde_receive.getActualQ())
            tcp_pose = np.array(rtde_receive.getActualTCPPose())

            # Enforce workspace limits at TCP level
            v_tcp = enforce_workspace_limits(tcp_pose, v_tcp, CYCLE_TIME)

            # Compute Jacobian and weighted DLS joint velocity
            J = jacobian_ur3e(q)
            qdot = weighted_dls_joint_velocity(J, v_tcp, q)

            # Enforce joint limits
            qdot = enforce_joint_limits(q, qdot, CYCLE_TIME)

            # Send joint velocity command
            rtde_control.speedJ(qdot.tolist(), JOINT_ACCEL, CYCLE_TIME)

        # Maintain roughly constant loop period
        elapsed = time.time() - loop_start
        sleep_time = CYCLE_TIME - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    pass
finally:
    rtde_control.stopScript()
    if gripper and gripper_enabled:
        gripper.close()
    print("Weighted DLS PS5 teleop stopped.")


