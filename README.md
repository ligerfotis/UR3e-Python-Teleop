# 🤖 UR3e Python Teleop
This repository contains code to teleoperate a UR3e-2 cobot with a wrist-attached Robotiq 2F-140 two-finger gripper using Python. 
Codes for two methods of teleoperation are provided:
1. Keyboard-based, and,
2. PlayStation-5 (DualSense) controller-based.

The repository also contains code for a 'sensor manager' that can be used to start and stop recordings from multiple sensors simultaneously based on keyboard inputs.

Further instructions for each part are provided in the README.md of their respective directories.

## Experimental Setup
### Hardware Requirements
#### Teleoperation:
The setup involves:
1. UR3e Robotic Arm
2. 2F-140 Robotiq two-finger gripper
3. Keyboard / PlayStation-5 (DualSense) controller

#### Sensor control:
The 'sensor manager' is intended to be used along with a simultaneously teleoperated robot. Therefore, in addition to the above components, the sensors required include:
1. Webcam to view the entire setup
2. DIGIT sensors on the gripper fingers
3. RealSense camera mounted on the gripper
4. Microphone to record gripper audio

### Software Requirements
- **OS:** Ubuntu 22.04.5 LTS
- **Python:** 3.10 Virtual Environment
- **PolyScope:** 5.15

## Setup and Installation
### Initial Robot Setup
#### *On the robot:*
1. Connect the Robotiq 2F-140 gripper to the wrist of the robot. 
2. Connect the robot to your computer through an Ethernet cable.

#### *On the Teach Pendant:*
*I. Static IP Network Setup:*
1. Power on the robot.
2. Ensure the robot is in 'Local Control'. (Button on the top-right of the screen)
3. Go to Options (Right-most button at the top of the screen) > Settings > System > Network. Select the network method as 'Static Address'. Set an IP address for the robot and keep the default Subnet mask of *255.255.255.0*.
4. On your computer, go to Settings and detect the Ethernet cable of your robot. Click the gear icon next to the cable and go to 'IPv4'. Select the IPv4 Method as 'Manual' and set a static IP address for your computer. Keep the default Subnet mask of *255.255.255.0*.
5. Ensure the static IP addresses just set for your robot and computer match.

*II. Installation:*
6. Download the URCap required for the 2F-140 Robotiq gripper [here](https://robotiq.com/support). Copy the file to a USB drive. Insert the drive into the teach pendant.
7. Go to Options (Right-most button at the top of the screen) > Settings > System > URCaps. Select the '+' icon and open the URCap from your USB drive. Click 'Open' to add the URCap to the list of 'Active URCaps'. Click 'Restart' to allow the changes to take place.
8. Power on the robot again.
9. Go to Installation (Top-left of the screen) > General > Tool I/O. Under I/O Interface Control, select 'Controlled by' 'Robotiq_Grippers'.
10. Go to Installation > URCaps > Gripper, and scan for the gripper. Once detected, activate the gripper using the 'Activate' button.
11. Go to Options (Right-most button at the top of the screen) > Settings > Security > Services, and ensure that RTDE is enabled.
12. Save the installation using the 'Save' button on the top-right of the screen. Restart the robot if required.

*III. Program:*
13. Download the UR program 'robotiq_gripper_python.urp' from this repository. Copy the file to a USB drive. Insert the drive into the teach pendant.
14. Click the 'Open' button on the top right of the screen and open the URP from your USB drive. Save the URP on your teach pendant using the 'Save' button on the top right of the screen.
15. Go to Installation (Top left of the screen) > General > Startup. Under 'Default Program File', select the newly saved URP for 'Load default program'. This will ensure that you do not have to reload the program after every startup of the robot.

### Initial Computer Setup
1. Clone the repository.
```
git clone https://github.com/Sujatha-H/UR3e-Python-Teleop
cd UR3e-Python-Teleop
```
2. Create a Python 3.10 virtual environment.
3. Install the necessary packages from 'requirements.txt'.
```
pip install -r requirements.txt
```

### Robot Setup After Every Startup
After the initial setup, there are some steps to be followed for setup after every power up of the robot in order to run the Python programs for teleoperation. These are as follows:
1. Go to Installation > URCaps > Gripper, and ensure that the gripper is detected. Once detected, activate the gripper using the 'Activate' button.
2. Go to Run (Left-most button at the top of the screen). Ensure that the title of your default program, 'robotiq_gripper_python', is shown under 'Program'.
3. Under 'Control', click the 'play' button to start playing the program. This UR program must be running whenever the Python program is run.
4. Put the robot in 'Remote Control' using the button on the top-right of the screen.

## Known Issues
The codes teleop_keyboard.py and teleop_ps5.py do not check for singularities, and so, if a singularity is reached, the robot will enter a protective stop. The robot may require being restarted in order for the Python program to work again.