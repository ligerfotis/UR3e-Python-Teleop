# ⚙️ UR3e Teleoperation
This directory contains two methods to teleoperate a UR3e cobot with a wrist-attached Robotiq 2F-140 two-finger gripper using Python: one using a keyboard (***teleop_keyboard.py***), and the other using a PlayStation 5 (DualSense) controller (***teleop_ps5.py***).
Both methods work based on the control of TCP position.

The functioning of each method is described below.

## User Instructions
### I. Requirements
**In both cases:**
- Fill the static IP address of your robot in the code in the variable '***robot_ip***' (for example, robot_ip = "192.168.1.222").
- Ensure that the robot is in 'Remote Control' mode.

**For *teleop_ps5.py*:**
- Connect the PlayStation 5 controller to your computer using Bluetooth. The controller can be put into pairing mode by simultaneously pressing the 'PS' button (bottom of the controller, above the microphone button) and the 'Share' button (three diagonal lines, on the left of the controller).

### II. Running the Codes
For keyboard-based teleoperation, run the following command:
```
cd /path/to/repository/UR3e-Python-Teleop
sudo .venv/bin/python Teleop/teleop_keyboard.py
```
This code must be run as sudo due to the use of the 'keyboard' library.

For PS5-based teleoperation, run the following command:
```
cd /path/to/repository/UR3e-Python-Teleop
.venv/bin/python Teleop/teleop_ps5.py
```

### III. Control
Both, keyboard and PlayStation 5 controller based methods of teleoperation allow for 6-DOF control: 3 axes of translation, and 3 axes of rotation.
They also enable gripper open and close.
The control keys for both methods are as follows:

| Teleoperation            | Translate Along X & Y axes | Translate Along Z-axis *(vertical)* | Rotate About X & Y axes | Rotate About Z-axis *(vertical)* | Gripper Open  | Gripper Close | Quit program  |
|:-------------------------|:--------------------------:|:-----------------------------------:|:-----------------------:|:--------------------------------:|:-------------:|:-------------:|:-------------:|
| ***teleop_keyboard.py*** |            WSAD            |                 ZX                  |          IKJL           |                NM                |  Left arrow   |  Right arrow  |       Q       |  
| ***teleop_ps5.py***      |       Left joystick        |               L1 / L2               |     Right joystick      |             R1 / R2              | Square button | Circle button | Option button |

**Note:**
The 'Option' button on the PS5 5 controller has three horizontal lines, and is found on the right of the controller.

**In both cases, other parameters that can be modified through the code include:**
- Linear speed of the robot (for translation)
- Angular speed of the robot (for rotation)
- Gripper force
- Gripper speed

### IV. Working
In both cases:
- Robot control works using RTDE (Real-Time Data Exchange) protocol.
- Gripper control is implemented using the class '*RobotiqSocketGripper*' (also in the *Teleop* directory).

## Class RobotiqSocketGripper
In both methods of teleoperation, Robotiq gripper control is achieved using the class '**RobotiqSocketGripper**'. This class works based on socket communication.

The class enables connection, activation and movement of the gripper to a specific pose. It also allows for setting the speed and force of the gripper, among other things.

Pose, speed and force are defined between values 0 and 255.

## Note:
The direction of movement of the robot when pressing the keys may vary based on the direction of viewing the robot and the direction of the TCP. 
Directions can be adjusted by the addition or removal of '-' signs for the '*speed_vector*' variable in the concerned code.

For example, in *teleop_keyboard.py*, if key 'A' moves your robot right instead of left, change the following code block:
```
        elif keyboard.is_pressed("a"): # Left
            speed_vector[0] = -linear_speed_magnitude
```
to:
```
        elif keyboard.is_pressed("a"): # Left
            speed_vector[0] = linear_speed_magnitude
```

Similar changes can be made to the '*speed_vector*' variable in the *teleop_ps5.py* code if required.