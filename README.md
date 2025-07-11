# UR3e-Python-Teleop
This repository contains Python code to control a UR3e-2 cobot with a wrist-attached Robotiq 2F-140 two-finger gripper.

## Experimental Setup
The setup involves a:
- UR3e Robotic Arm
- 2F-140 Robotiq two-finger gripper

## Setup and Installation
Ensure static IP addresses of robot and computer match
URScript with loop of wait (0.1s) => Ensure running 
In Installations<General, under Tool I/O select 'Controlled by' "Robotiq_Grippers"
In Installations>Gripper, ensure gripper is detected and activated
Ensure in Settings>Security>Services, RTDE is enabled
Enter Remote control mode before running Python script
Run Python code as sudo
