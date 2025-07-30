# 🔧 Robot Setup
This file contains the steps to be followed during initial setup of the UR3e robot and the Robotiq 2F-140 gripper.
These steps are only required to be done once.

### *On the robot:*
1. Connect the Robotiq 2F-140 gripper to the wrist of the robot. 
2. Connect the robot to your computer through an Ethernet cable.

### *On the Teach Pendant:*
#### *I. Static IP Network Setup:*
1. Power on the robot.
2. Ensure the robot is in 'Local Control' mode. (Button on the top-right of the screen)
3. Go to Options (Right-most button at the top of the screen) > Settings > System > Network. Select the network method as 'Static Address'. Set an IP address for the robot (for example, 192.168.1.222) and keep the default Subnet mask of *255.255.255.0*.
4. On your computer, go to Settings and detect the Ethernet cable of your robot. Click the gear icon next to the cable and go to 'IPv4'. Select the 'IPv4 Method' as 'Manual' and set a static IP address for your computer (for example, 192.168.1.200). Keep the default Subnet mask of *255.255.255.0*.
5. Ensure the static IP addresses just set for your robot and computer match.

#### *II. Installation:*
6. Download the URCap required for the 2F-140 Robotiq gripper [here](https://robotiq.com/support). Copy the file to a USB drive. Insert the drive into the teach pendant.
7. Go to 'Options' (Right-most button at the top of the screen) > Settings > System > URCaps. Select the '+' icon and open the URCap from your USB drive. Click 'Open' to add the URCap to the list of 'Active URCaps'. Click 'Restart' to allow the changes to take place.
8. Power on the robot again.
9. Go to Installation (Top-left of the screen) > General > Tool I/O. Under I/O Interface Control, select 'Controlled by' 'Robotiq_Grippers'.
10. Go to Installation > URCaps > Gripper, and scan for the gripper. Once detected, activate the gripper using the 'Activate' button.
11. Go to 'Options' (Right-most button at the top of the screen) > Settings > Security > Services, and ensure that RTDE is enabled.
12. Save the installation using the 'Save' button on the top-right of the screen. Restart the robot if required.

#### *III. Program:*
13. Download the UR program '**robotiq_gripper_python.urp**' from this repository. Copy the file to a USB drive. Insert the drive into the teach pendant.
14. Click the 'Open' button on the top right of the screen and open the URP from your USB drive. Save the URP on your teach pendant using the 'Save' button on the top right of the screen.
15. Go to Installation (Top left of the screen) > General > Startup. Under 'Default Program File', select the newly saved URP for 'Load default program'. This will ensure that you do not have to reload the program after every startup of the robot.
