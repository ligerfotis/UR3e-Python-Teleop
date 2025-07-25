"""Function to record audio from a microphone based on keyboard inputs"""
"""Uses function from get_unique_filename.py"""
"""Intended to be used with sensor_manager.py for coordinated recording"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import json
from time import perf_counter as now

from get_unique_filename import get_unique_filename

# Search for microphone name to detect index automatically
def find_microphone_index():
    for device in sd.query_devices():
        name = device['name'].lower()
        if " " in name: # Add name according to requirement
            return device['index']
    return None


# Main function to capture audio data
def audio_capture(root_dir: str, recording_event, stop_event):
    """
    Records multiple audio files in one program session based on keyboard inputs
    """

    # Detect audio device
    device_index = find_microphone_index()
    if device_index is None:
        print("Microphone not found.")
        return

    samplerate = 48000 # 48 kHz
    channels = 1 # Mono audio

    # print(f"Audio device index: {device_index} ")

    # Main loop
    try:
        print(f"- Microphone {device_index} thread ready.")
        while not stop_event.is_set(): # Before program stop
            # Start recording when triggered
            if recording_event.is_set():
                start_time = now()  # Start timer
                buffer = [] # Reset buffer to store audio chunks
                audio_log = [] # Reset log
                sample_count = 0
                print(f"+ Microphone {device_index} recording started.")

                # Callback function to handle incoming audio data
                def callback(indata, samples, time_info, status):
                    """
                    Args:
                        indata: Numpy array of audio data
                        samples: Number of audio samples per channel
                        time_info: Metadata about timing of audio
                        status: Reports errors and warnings
                    """
                    nonlocal sample_count

                    if stop_event.is_set(): # Checks if program is stopped
                        raise sd.CallbackAbort
                    buffer.append(indata.copy()) # Append incoming audio to buffer
                    sample_count += samples
                    elapsed_time = now() - start_time
                    audio_log.append({
                        "elapsed_time": elapsed_time,
                        "frame_count": sample_count,
                    })

                try:
                    # Open input audio stream
                    with sd.InputStream(samplerate=samplerate, channels=channels, device=device_index, callback=callback):
                        #While recording
                        while recording_event.is_set() and not stop_event.is_set():
                            sd.sleep(100)
                except Exception as e:
                    print(f"Microphone stream failed: {e}")
                    continue

                # Save audio data to WAV file
                if buffer:
                    # Concatenate all buffer data
                    audio_data = np.concatenate(buffer)
                    # Create unique filename to prevent overwriting
                    filepath = get_unique_filename("microphone_audio", ".wav", root_dir)
                    # Write audio to filepath
                    sf.write(filepath, audio_data, samplerate)
                    print(f"! Microphone recording stopped.")

                    # Save log
                    log_path = get_unique_filename("microphone_log", ".json", root_dir)
                    with open(log_path, "w") as f:
                        json.dump(audio_log, f, indent=4)

                    # Calculate actual sample rate
                    duration = now() - start_time
                    actual_samplerate = sample_count / duration if duration > 0 else samplerate

                    # Catch FPS deviations greater than 3
                    if abs(actual_samplerate - samplerate) > 1500:
                        print(f"[WARNING] Microphone sample rate deviated significantly: {actual_samplerate:.2f}")

                    print(f"Microphone audio saved with {actual_samplerate:.2f} FPS to {filepath}")

                else:
                    print("No microphone data captured.")

                # Reset recording_event to allow next recording
                recording_event.clear()

            sd.sleep(100) # Idle time to wait for next recording

    finally:
        print("Microphone capture stopped.")