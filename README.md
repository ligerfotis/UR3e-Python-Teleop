# UR3e Python Teleop
This repository contains Python codes to control a UR3e-2 cobot with a wrist-attached Robotiq 2F-140 two-finger gripper using a keyboard and a PlayStation-5 controller.

## Experimental Setup
The setup involves:
- UR3e Robotic Arm
- 2F-140 Robotiq two-finger gripper
- PlayStation-5 (DualSense) controller

## Software Requirements
- **OS:** Ubuntu
- **Python:** 3.10 (PyCharm virtual environment)
- **Libraries:** keyboard, ur-rtde, pyserial, numpy, setuptools
- **PolyScope:** 5.15

## Setup and Installation

### Installation
1. Clone the repository.
```
git clone https://github.com/Sujatha-H/UR3e-Python-Teleop
cd UR3e-Python-Teleop
```
2. Create a Python 3.10 virtual environment.
3. Install required packages from requirements.txt
```
pip install -r requirements.txt
```
4. Copy robotiq_gripper_python.urp to the teach pendant using a USB drive.

### Setup
#### Hardware:
- Connect the Robotiq gripper to the wrist of the robot.
- Connect the PlayStation-5 controller to the computer using Bluetooth.

#### Pendant:
1. Set a static IP address for the robot through the pendant and ensure it matches the static IP address of the computer being used.
2. Install the URCap required for Robotiq grippers.
3. In Installation > General > Tool I/O, select 'Controlled by' "Robotiq_Grippers".
4. In Installation > Gripper, ensure the gripper is detected and activated.
5. In Settings > Security > Services, ensure that RTDE is detected and activated.
6. Run the UR program robotiq_gripper_python.urp.
7. Select 'Remote Control' mode before running the Python code.

#### Software:
- Run teleop_keyboard.py as sudo for keyboard control of UR3e.
- Run teleop_ps5.py for PlayStation-5 controller control of UR3e.
