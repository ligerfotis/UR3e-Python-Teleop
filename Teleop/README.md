# ⚙️ UR3e Teleoperation
This directory contains two methods to teleoperate a UR3e cobot with a wrist-attached Robotiq 2F-140 two-finger gripper using Python: one using a keyboard, and the other using a PlayStation 5 (DualSense) controller.
Both methods work based on control of TCP position.

The functioning of each method is described below.

## Keyboard-Based Teleoperation
6-DOF UR3e keyboard-based teleoperation can be done using the file '**teleop_keyboard.py**'.

### Key features
This code allows for:

1. **3-DOF translation using the keys:**

    ***WSAD:*** Forward / backward, Left / right

    ***ZX:*** Up / down


2.  **3-DOF rotation using the keys:**

    ***IKJL:*** Rotate about x-axis, Rotate about y-axis

    ***NM:*** Rotate about z-axis (vertical axis)


3. **Robotiq 2F-140 gripper control using the keys:**

    ***Left arrow:*** Gripper close

    ***Right arrow:*** Gripper open

**Other parameters that can be modified include:**
- Linear speed of the robot (for translation)
- Angular speed of the robot (for rotation)
- Gripper force
- Gripper speed

***The program is quit by pressing the 'q' key.***

### Working
- Robot control works using RTDE (Real-Time Data Exchange) protocol.
- Gripper control is implemented using the class '*RobotiqSocketGripper*' (also in the Teleop directory).

### Requirements
- Fill the static IP address of your robot in the code. 
- This code must be run as sudo due to the use of the 'keyboard' library.
- Ensure that the robot is in 'Remote Control' mode.

## PlayStation 5 Controller-Based Teleoperation
6-DOF UR3e PlayStation 5 controller-based teleoperation can be done using the file '**teleop_ps5.py**'.

### Key features
This code allows for:

1. **3-DOF translation using:**

    ***Left joystick:*** Forward / backward, Left / right

    ***L1 / L2:*** Up / down


2.  **3-DOF rotation using:**

    ***Right joystick:*** Rotate about x-axis, Rotate about y-axis

    ***R1 / R2:*** Rotate about z-axis (vertical axis)


3. **Robotiq 2F-140 gripper control using:**

    ***Square button:*** Gripper close

    ***Circle button:*** Gripper open

**Other parameters that can be modified include:**
- Linear speed of the robot (for translation)
- Angular speed of the robot (for rotation)
- Gripper force
- Gripper speed

***The program is quit by pressing the 'Option' button*** (three horizontal lines, on the right of the controller).

### Working
- Robot control works using RTDE (Real-Time Data Exchange) protocol.
- Gripper control is implemented using the class '*RobotiqSocketGripper*' (also in the Teleop directory).

### Requirements
- Fill the static IP address of your robot in the code. 
- Connect the PlayStation 5 controller to your computer using Bluetooth. The controller can be put into pairing mode by simultaneously pressing the 'PS' button (bottom of the controller, above the microphone button) and the 'Share' button (three diagonal lines, on the left of the controller)
- Ensure that the robot is in 'Remote Control' mode.

## Class RobotiqSocketGripper
In both methods of teleoperation, Robotiq gripper control is achieved using the class '**RobotiqSocketGripper**'. This class works based on socket communication.

The class enables connection, activation and movement of the gripper to a specific pose. It also allows for setting the speed and force of the gripper, among other things.

Pose, speed and force are defined between values 0 and 255.

## Note:
The direction of movement of the keys may vary based on the direction of viewing the robot and the direction of the TCP. 
Directions can be adjusted by the addition or removal of '-' signs for the *speed_vector* variable in the code.

For example, in *teleop_keyboard.py*, if key 'a' moves your robot right instead of left, change the following code block:
```
        elif keyboard.is_pressed("a"): # Left
            speed_vector[0] = -linear_speed_magnitude
```
to:
```
        elif keyboard.is_pressed("a"): # Left
            speed_vector[0] = linear_speed_magnitude
```

Similar changes can be made to *teleop_ps5.py* also if required.