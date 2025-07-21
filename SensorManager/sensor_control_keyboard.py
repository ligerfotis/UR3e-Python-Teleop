"""Function to control sensor_manager.py using keyboard inputs"""
"""Enables synchronous activation and deactivation of all sensors used"""
"""Requires sudo permission"""

import threading
import keyboard
import time


def sensor_control_keyboard(start_event: threading.Event, stop_event: threading.Event):
    """
    Pressing 'r' starts recording and pressing 's' stops the program and recording
    """
    print("Press 'r' to start recording and 's' to stop the program.")
    while not stop_event.is_set():
        if keyboard.is_pressed('r') and not start_event.is_set():
            start_event.set()
            print("Recording started.")
            time.sleep(0.5) # Debounce
        elif keyboard.is_pressed('s'):
            stop_event.set()
            print("Recording stopped.")
            time.sleep(0.5) # Debounce