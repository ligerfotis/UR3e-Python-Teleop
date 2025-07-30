# 🖥️ Sensor Manager
This directory contains codes that enable simultaneous recording from multiple sensors based on keyboard inputs for when to start and stop recording. 
These sensor recordings are intended to be done alongside robot teleoperation, which can be done using the codes from the *Teleop* directory.

The sensor recordings that can be done using these codes include:
1. Webcam video to record the entire robot setup
2. Multiple DIGIT sensor videos (one sensor on each gripper finger)
3. RealSense camera RGB and depth videos (camera mounted on the gripper)
4. Microphone audio to record gripper sound
5. Proprioception logging to log TCP, joint and gripper positions

Frame / sample logs are also created with timestamps for each sensor during recording periods to allow for quality checking. 

In addition, there is also a '***proprioception_replay.py***' file which can be used to replay the movements logged by *proprioception_logger.py*.

***All video recordings are set to be written at 30 FPS and the audio recording is set to be written at 48 kHz.***

## Sensor Capture (Record)
'***sensor_manager.py***' contains the code that should be run in order to start the sensor capture. This code manages all the other sensors using threads.

### I. Features
- The status of the sensors in terms of detection and connection is shown on the terminal. 
- A live preview is shown on the screen for the webcam, DIGIT sensors and RealSense colour and depth cameras as long as the program is running.
- For each sensor, apart from the sensor data, a frame / sample log with corresponding timestamps (in terms of elapsed time from recording start) is generated to allow for quality control. 
- At the end of each recording, the FPS of the recording along with the filepath is displayed. 
- If there was a significant deviation in FPS (>= 3 FPS for videos and >= 1500 samples per second for audio), it will be indicated on the terminal after recording stop. 

### II. Requirements
1. Ensure that all your sensors are connected to your computer.
2. Fill in '***robot_ip***' in *sensor_manager.py* with your robot's static IP address 
   (for example, robot_ip = "192.168.1.222")
3. Fill in '***root_dir***' in *sensor_manager.py* with the path to the directory where you want your sensor data to be saved (for example, root_dir = "/home/user/SensorCapture")
4. Fill in Line 17 (***if " " in name:***) in *find_microphone_index()* in *audio_capture.py* with the name of your audio device (for example, if "pnp audio" in name:). 
   - In order to find the name of your device, open a Python script and run the following command:
   ```
   import sounddevice as sd
   print(sd.query_devices())
   ```
   - Identify your microphone and modify the string with its name.

### III. Running the Code
Run the following command to begin sensor capture:
```
sudo PYTHONPATH=$(pwd) .venv/bin/python SensorManager/sensor_manager.py
```
This code is run as sudo due to the use of the 'keyboard' library for the recording start and stop inputs.

### IV. User Instructions
After the code is run, pressing the following keyboard keys controls recording timing:

| Key |     Function     |
|:---:|:----------------:|
|  1  | Start recording  |
|  2  |  Stop recording  |
|  3  | Stop the program |

### V. Notes
1. Check the live preview of the sensors before recordings to make sure that all of them are functioning properly.
2. Ensure that the DIGIT sensors and RealSense camera are plugged into USB 3.0 ports without shared hubs to prevent drops in FPS or frequent disconnection.
3. **Since the program is to be run in sudo, the generated sensor data may only have root access. To give folder permission to the user, run the following command in the terminal:**
```
sudo chown -R user:user [folder_path]
```
This command will ensure that all files within the specified folder are now owned by the specified user.

#### 4. Proprioception Logging
'***proprioception_logger()***' in '*proprioception_logger.py*' creates a 30Hz JSON log of the following parameters:
  1. "***elapsed_time***": Time elapsed from recording start
  2. "***tcp_pose_mm***": Cartesian pose of TCP in mm
  3. "***joint_positions_rad***": Joint angles in rad
  4. "***gripper_position_0_255***": Gripper position in range 0-255
  5. "***gripper_position_mm***": Gripper position in mm

The proprioception logging does not require teleoperation using the '*Teleop*' directory codes to work. The logging works even when manually moving the robot in freedrive mode.

## Proprioception Replay
'***proprioception_replay.py***' contains the code can be used to replay the movements recorded by *proprioception_logger()*. 

### I. Requirements
While this code is intended to replay movements logged by '*proprioception_logger()*', any JSON log file could be used as long as it has **either one** of the following sets of logged entries:
1. "*elapsed_time*", "*tcp_pose_mm*" and "*gripper_position_0_255*", or,
2. "*elapsed_time*", "*joint_positions_rad*" and "*gripper_position_0_255*".

Therefore, the proprioception replay can be done using either TCP positions or joint angles.

**Before running the code:**
1. Fill in the static IP address of your robot in the variable '***robot_ip***' (for example, robot_ip = "192.168.1.222").
2. Fill in the path of the JSON file from which the movements are to be replayed in the variable '***json_file***' (for example, json_file = "/home/user/SensorCapture/proprioception_log.json").
3. Active the gripper.
4. Ensure the robot is in 'Remote Control' mode.

### II. Running the Code
Run the following command to begin proprioception replay:
```
PYTHONPATH=$(pwd) .venv/bin/python SensorManager/proprioception_replay.py
```