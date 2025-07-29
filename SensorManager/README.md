# 🖥️ Sensor Manager
This directory contains codes that enable simultaneous recording from multiple sensors based on keyboard inputs for when to start and stop recording. 
These sensor recordings are intended to be done alongside robot teleoperation, which can be done using codes from the *Teleop* directory.

The sensor recordings that can be done using these codes include:
1. Webcam video to record the entire robot setup
2. Multiple DIGIT sensor videos (one sensor on each gripper finger)
3. RealSense camera colour and depth videos (camera mounted on the gripper)
4. Microphone audio to record gripper sound
5. Proprioception logging to log TCP, joint and gripper positions

Frame / sample logs are also created with timestamps for each sensor during recording periods to allow for quality checking. 

In addition, there is also a '***proprioception_replay.py***' file which could be used to replay the movements logged by *proprioception_logger.py*.

***All video recordings are set to be written at 30 FPS and the audio recording is set to be written at 48 kHz.***

## sensor_manager.py
This is the main code file that should be run in order to start the program. This code manages all the other sensors using threads.

The status of the sensors in terms of detection and connection is shown on the terminal. At the end of each recording, the FPS of the recording along with the filepath is displayed. 
If there was a significant deviation in FPS (>= 3 FPS for videos and >= 1500 samples per second for audio), it will be indicated on the terminal after recording stop.

A live preview is shown on the screen for the webcam, DIGIT sensors and RealSense colour and depth cameras as long as the program is running.

### User Instructions
- Press '1' to start recording
- Press '2' to stop recording
- Press '3' to stop the program

### Requirements
- Fill the static IP address of your robot in the code. 
- Fill the directory name to store the recorded data in.
- This code must be run as sudo due to the use of the 'keyboard' library.

### Notes:
- Check the live preview of the sensors before recordings to make sure that all of them are functioning properly.
- Ensure that the DIGIT sensors and RealSense camera are plugged in to USB 3.0 ports to prevent drops in FPS or frequent disconnection.

## camera_capture.py
This code is used to record AVI videos from a **single** webcam. The code contains two functions:
1. ***find_camera_index()***: Detects a camera that is not a DIGIT sensor or a RealSense camera. This function only works in Linux systems due to the '*pyudev*' module.
2. ***camera_capture()***: Uses the detected camera to start and stop recording videos based on keyboard inputs. Pushes frames to queue for live preview in *sensor_manager.py*. Also creates JSON frame logs during recording.

## digit_capture.py
This code is used to record AVI videos from **multiple** DIGIT sensors. The code contains two functions:
1. ***digit_capture()***: Detects all connected DIGIT sensors' serial numbers and creates a separate thread for each of them. Starts and stops the threads based on keyboard inputs.
2. ***run_digit_thread()***: Starts and stops recording videos based on keyboard inputs for a single DIGIT sensor. Pushes frames to queue for live preview in *sensor_manager.py*. Also creates JSON frame logs during recording.

## realsense_capture_color.py
This code is used to record RGB AVI videos from a **single** RealSense camera. The code contains one function:
- ***realsense_capture_color()***: Starts and stops recording videos based on keyboard inputs for an RGB RealSense camera. Pushes frames to queue for live preview in *sensor_manager.py*. Also creates JSON frame logs during recording.

## realsense_capture_depth.py
This code is used to record depth AVI videos from a **single** RealSense camera. The code contains one function:
- ***realsense_capture_depth()***: Starts and stops recording videos based on keyboard inputs for a depth RealSense camera. Pushes frames to queue for live preview in *sensor_manager.py*. Also creates JSON frame logs during recording.

## audio_capture.py
This code is used to record WAV audio from a **single** microphone. The code contains two functions:
1. ***find_microphone_index()***: Detects a microphone based on its name. 
2. ***audio_capture()***: Uses the detected camera to start and stop recording audio based on keyboard inputs. Also creates JSON sample logs during recording.

### Note:
The automatic microphone detection by *find_microphone_index* is done based on the name of the microphone to prevent use of the webcam microphones. 
Fill in the name for your microphone accordingly. 

## proprioception_logger.py
This code is used to log the TCP positions, joint angles and gripper positions of the UR robot with their respective timestamps.
The code contains one function:
- ***proprioception_logger()***: Creates a 30Hz JSON log of the following parameters:
    1. "*elapsed_time*": Time elapsed from recording start
    2. "*tcp_pose_mm*": Cartesian pose of TCP in mm
    3. "*joint_positions_rad*": Joint angles in rad
    4. "*gripper_position_0_255*": Gripper position in range 0-255
    5. "*gripper_position_mm*": Gripper position in mm

The proprioception logging does not require teleoperation using the *Teleop* directory codes to work. The logging works even when moving the robot in freedrive mode.

## proprioception_replay.py
This code can be used to replay the movements recorded 