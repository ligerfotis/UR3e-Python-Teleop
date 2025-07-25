"""Function to control sensor_manager.py using keyboard inputs"""
"""Enables synchronous recording of all sensors used"""
"""Requires sudo permission"""

import threading
import keyboard
import time


def sensor_control_keyboard(recording_event: threading.Event, stop_event: threading.Event):
    """
    Pressing '1' starts recording, pressing '2' stops recording and pressing '3' stops the program.
    """
    print("Once all sensors are active, press '1' to start recording, '2' to stop recording, '3' to stop the program.")
    while not stop_event.is_set():
        if keyboard.is_pressed('1') and not recording_event.is_set():
            recording_event.set()
            print("Recording started.")
            time.sleep(0.5) # Debounce
        elif keyboard.is_pressed('2') and recording_event.is_set():
            recording_event.clear()
            print("Recording stopped.")
            time.sleep(0.5) # Debounce
        elif keyboard.is_pressed('3'):
            stop_event.set()
            print("Program stopped.")
            time.sleep(0.5) # Debounce