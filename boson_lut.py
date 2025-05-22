# %%
import cv2
import numpy as np
import argparse
import datetime
import tkinter as tk
from tkinter import Button, Label, Frame, OptionMenu, StringVar, filedialog, messagebox
from PIL import Image, ImageTk
import os
import pyudev
import threading
import time
import ast
import queue  # Import queue for safe Tkinter updates
import json # Import json for configuration saving/loading
import traceback # For detailed error logging
# Add traceback for debugging errors within Tkinter callbacks
import traceback

DEBUG_MODE = False


# Constants

LUTS = {
    'BONE': cv2.COLORMAP_BONE,
    'BLACKHOT': cv2.COLORMAP_JET,
    'WHITEHOT': cv2.COLORMAP_HOT,
    'RAINBOW': cv2.COLORMAP_RAINBOW,
    'OCEAN': cv2.COLORMAP_OCEAN,
    'LAVA': cv2.COLORMAP_PINK,
    'ARCTIC': cv2.COLORMAP_WINTER,
    'GLOBOW': cv2.COLORMAP_PARULA,
    'GRADEDFIRE': cv2.COLORMAP_AUTUMN,
    'INSTALERT': cv2.COLORMAP_SUMMER,
    'SPRING': cv2.COLORMAP_SPRING,
    'SUMMER': cv2.COLORMAP_SUMMER,
    'COOL': cv2.COLORMAP_COOL,
    'HSV': cv2.COLORMAP_HSV,
    'PINK': cv2.COLORMAP_PINK,
    'HOT': cv2.COLORMAP_HOT,
    'MAGMA': cv2.COLORMAP_MAGMA,
    'INFERNO': cv2.COLORMAP_INFERNO,
    'PLASMA': cv2.COLORMAP_PLASMA,
    'VIRIDIS': cv2.COLORMAP_VIRIDIS,
    'CIVIDIS': cv2.COLORMAP_CIVIDIS,
    # Using lambda functions to defer creation until needed
    'ISOTHERM_RED': lambda: create_custom_lut('red', 32),
    'ISOTHERM_GREEN': lambda: create_custom_lut('green', 32),
    'ISOTHERM_BLUE': lambda: create_custom_lut('blue', 32)
}

CAMERA_RESOLUTIONS = {
    'BOSON': (640, 512),
    'LEPTON3': (160, 120),
    'LEPTON2': (80, 60),
    'AMPBANK': (256, 192)
}

TARGET_PROCESSING_FPS = {
    'LEPTON3': 15,    # Lepton (~9Hz source, ~15Hz processing) -> 10 FPS display (100ms delay) is fine.
    'LEPTON2': 15,    # Lepton (~9Hz source, ~15Hz processing) -> 10 FPS display (100ms delay) is fine.
    'BOSON': 60,      # Boson (30Hz or 60Hz processing) -> 30 FPS display (33ms delay) offers good smoothness.
    'AMPBANK': 25,    # Ampbank (25Hz source/processing) -> 25 FPS display (40ms delay).
    'Unknown': 15     # Fallback display rate for unknown camera types.
}

# separate definition for display FPS vs processing FPS (ffmpeg output)
TARGET_DISPLAY_FPS = {
    'LEPTON3': 10,  # Lepton (~9Hz source, ~15Hz processing) -> 10 FPS display (100ms delay) is fine.
    'LEPTON2': 10,  # Lepton (~9Hz source, ~15Hz processing) -> 10 FPS display (100ms delay) is fine.
    'BOSON': 60,    # Boson (30Hz or 60Hz processing) -> 30 FPS display (33ms delay) offers good smoothness.
    'AMPBANK': 25,  # Ampbank (25Hz source/processing) -> 25 FPS display (40ms delay).
    'Unknown': 10   # Fallback display rate for unknown camera types.
}

FLIR_VENDOR_ID = '09cb'     # FLIR vendor ID
LEPTON_PRODUCT_ID = '0100'  # FLIR Lepton (Cubeternet WebCam) product ID
AMPBANK_VENDOR_ID = '3474'  # Ampbank vendor ID
AMPBANK_PRODUCT_ID = '4321' # Ampbank product ID

CONFIG_FILE = 'thermal_config.json'

# Thread Locks
cap_lock = threading.Lock()
out_lock = threading.Lock()
recording_lock = threading.RLock()
screenshot_lock = threading.Lock() # Lock for screenshot_requested

update_display_call_count = 0

# Threading Events
exit_event = threading.Event() # Event to signal main program exit
capture_stop_event = threading.Event() # Event to signal video capture thread to stop
camera_disconnected_event = threading.Event() # Event to signal camera disconnection

# Shared Variables
screenshot_requested = False
frame_queue = queue.Queue(maxsize=1) # Queue to safely pass frames to the main thread

current_camera_type = None # Variable to store the current camera type string (e.g., 'BOSON')
current_display_delay_ms = 100 # Default to 100ms (10 FPS) initially

# Helper Functions

def load_custom_luts():
    """Scans the current working directory for .lut files and loads them into the LUTS dictionary."""
    for filename in os.listdir('.'):
        if filename.endswith('.lut'):
            try:
                with open(filename, 'r') as f:
                    lut_data = f.read().strip()
                    try:
                        if lut_data.startswith('[') and lut_data.endswith(']'):
                            lut_values = ast.literal_eval(lut_data)
                            if isinstance(lut_values, list) and all(isinstance(color, tuple) and len(color) == 3 for color in lut_values):
                                lut_name = os.path.splitext(filename)[0]
                                lut_array = np.array(lut_values, dtype=np.uint8).reshape((256, 1, 3))
                                LUTS[lut_name] = lut_array
                                print(f"Loaded custom LUT: {lut_name}")
                            else:
                                print(f"Invalid format in LUT file: {filename}. Expected a list of 3-tuple colors.")
                        else:
                            print(f"Invalid format in LUT file: {filename}. File must contain a list of color tuples.")
                    except (SyntaxError, ValueError) as e:
                         print(f"Error parsing LUT data in {filename}: {e}")
                    except Exception as e:
                        print(f"Unexpected error validating LUT data in {filename}: {e}")

            except FileNotFoundError:
                print(f"Error: LUT file not found {filename}")
            except IOError as e:
                print(f"Error reading LUT file {filename}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {filename}: {e}")


def get_video_devices_for_flir():
    """Detects available FLIR cameras using pyudev."""
    flir_devices = []
    if DEBUG_MODE: print("DEBUG: Entering get_video_devices_for_flir...")
    try:
        if DEBUG_MODE: print("DEBUG: Attempting to create pyudev.Context()...")
        context = pyudev.Context()
        if DEBUG_MODE: print("DEBUG: pyudev.Context() created successfully.")
        
        if DEBUG_MODE: print("DEBUG: Attempting to list video4linux devices...")
        # Create an iterator for devices
        devices_to_process_iterator = context.list_devices(subsystem='video4linux')
        
        # Check if the iterator is empty without consuming it all at once
        try:
            first_device_check = next(iter(context.list_devices(subsystem='video4linux'))) # Test iteration
            if DEBUG_MODE: print(f"DEBUG: Found at least one video4linux device ({first_device_check.device_node}). Proceeding to iterate.")
            # Re-create the iterator for the actual loop as `next` consumes one item
            devices_to_process = context.list_devices(subsystem='video4linux')
        except StopIteration:
            if DEBUG_MODE: print("DEBUG: No video4linux devices found by context.list_devices iterator.")
            devices_to_process = [] # Ensure an empty iterable if no devices

        processed_any_device_in_loop = False
        for device in devices_to_process:
            processed_any_device_in_loop = True
            if DEBUG_MODE: print(f"DEBUG: Processing V4L device: {device.device_node}")
            parent = device.find_parent(subsystem='usb', device_type='usb_device')
            if parent is None:
                if DEBUG_MODE: print(f"DEBUG:   Device {device.device_node} has no USB parent. Skipping.")
                continue
            
            vendor_id = parent.properties.get('ID_VENDOR_ID')
            product_id = parent.properties.get('ID_MODEL_ID')
            model = parent.properties.get('ID_MODEL', 'Unknown Model') # Default if not present
            device_node = device.device_node 

            camera_type_str = 'Unknown'
            # logic to determine camera_type_str based on vendor_id, product_id, model
            if vendor_id == FLIR_VENDOR_ID:
                if 'Boson' in model: camera_type_str = 'BOSON'
                # Add other FLIR models if needed
            elif vendor_id == '1e4e' and product_id == LEPTON_PRODUCT_ID: # GroupGets PureThermal, common for Lepton
                camera_type_str = 'LEPTON3' 
            elif vendor_id == AMPBANK_VENDOR_ID and product_id == AMPBANK_PRODUCT_ID:
                camera_type_str = 'AMPBANK'

            if DEBUG_MODE: print(f"DEBUG:   Potential thermal camera: Model: '{model}', Device: {device_node}, VendorID: {vendor_id}, ProductID: {product_id}, Identified Type: {camera_type_str}")
            flir_devices.append({'device_node': device_node, 'camera_type': camera_type_str, 'model': model})
        
        if not processed_any_device_in_loop and not devices_to_process: # Check if loop ran
             if DEBUG_MODE: print("DEBUG: Loop for processing V4L devices did not iterate (no devices passed to it).")
        elif not processed_any_device_in_loop and devices_to_process:
             if DEBUG_MODE: print("DEBUG: Loop for processing V4L devices was given devices but did not seem to process any (check loop logic).")


    except pyudev.PermissionsError as e:
        print(f"CRITICAL PYUDEV PermissionsError in get_video_devices_for_flir: {e}")
        if DEBUG_MODE: print("DEBUG: This often means the user doesn't have permission to access udev. Try checking udev rules or running with sudo (if appropriate and safe).")
        traceback.print_exc()
    except pyudev.EnvironmentError as e: # For other udev environment issues like libudev not found
        print(f"CRITICAL PYUDEV EnvironmentError in get_video_devices_for_flir: {e}")
        if DEBUG_MODE: print("DEBUG: This might be due to libudev.so not being found or other system udev issues.")
        traceback.print_exc()
    except Exception as e:
        print(f"CRITICAL UNEXPECTED ERROR in get_video_devices_for_flir: {e}")
        traceback.print_exc()

    if not flir_devices:
        if DEBUG_MODE: print("DEBUG: No thermal cameras were identified after processing all devices.")
    else:
        if DEBUG_MODE: print(f"DEBUG: Identified {len(flir_devices)} thermal camera(s) to be returned.")
    if DEBUG_MODE: print("DEBUG: Exiting get_video_devices_for_flir.")
    return flir_devices


def create_custom_lut(color, color_gradient_step):
    """Creates a custom LUT using predefined color data."""
    if color_gradient_step <= 0:
        raise ValueError("color_gradient_step must be greater than 0.")

    if color == 'red':
        gradient_colors = ((0, 0, 64), (0, 0, 255), color_gradient_step)
    elif color == 'green':
        gradient_colors = ((0, 64, 0), (0, 255, 0), color_gradient_step)
    elif color == 'blue':
        gradient_colors = ((64, 0, 0), (255, 0, 0), color_gradient_step)
    else:
        raise ValueError("Unsupported color for LUT creation. Supported colors are 'red', 'green', and 'blue'.")

    BLACK_TO_WHITE_STEP = 256 - color_gradient_step
    # Ensure linspace arguments are correct
    if BLACK_TO_WHITE_STEP > 0:
        black_to_white = np.linspace((0, 0, 0), (255, 255, 255), BLACK_TO_WHITE_STEP).astype(np.uint8)
    else:
        black_to_white = np.array([], dtype=np.uint8).reshape(0, 3) # Handle case where step is 256

    color_gradient = np.linspace(*gradient_colors).astype(np.uint8)
    custom_colors = np.concatenate((black_to_white, color_gradient))

    # Ensure we have exactly 256 colors by interpolating if necessary
    if len(custom_colors) != 256:
        custom_colors = np.linspace(custom_colors[0], custom_colors[-1], 256, dtype=np.uint8)

    # Create a custom LUT with 256 entries
    custom_lut = custom_colors.reshape((256, 1, 3))
    return custom_lut


def apply_lut(frame, lut_name): # frame is expected to be 8-bit single-channel (CV_8UC1)
    """Applies the selected LUT to a video frame using cv2.applyColorMap."""
    if not (len(frame.shape) == 2 and frame.dtype == np.uint8):
        print(f"ERROR apply_lut: Input frame is not 8-bit single-channel! Shape: {frame.shape}, dtype: {frame.dtype}.")
        # Fallback: convert to BGR so it can be displayed, or handle error appropriately
        return cv2.cvtColor(frame.astype(np.uint8) if frame.dtype != np.uint8 else frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame.astype(np.uint8)

    if lut_name not in LUTS:
        print(f"Warning: Unsupported LUT: {lut_name}. Using WHITEHOT instead.")
        lut_name = 'WHITEHOT' 

    colormap_source = LUTS[lut_name] # This can be an ID, a numpy array, or a callable
    
    actual_colormap_for_apply = None
    if callable(colormap_source): 
        actual_colormap_for_apply = colormap_source() # Returns (256,1,3) np.uint8 array
        # print(f"DEBUG apply_lut: Using custom generated LUT array for '{lut_name}' - shape: {actual_colormap_for_apply.shape}, dtype: {actual_colormap_for_apply.dtype}")
    elif isinstance(colormap_source, np.ndarray): 
        actual_colormap_for_apply = colormap_source # Is a (256,1,3) np.uint8 array
        # print(f"DEBUG apply_lut: Using custom file LUT array for '{lut_name}' - shape: {actual_colormap_for_apply.shape}, dtype: {actual_colormap_for_apply.dtype}")
    else: 
        actual_colormap_for_apply = colormap_source # Is an integer ID for built-in OpenCV colormaps
        # print(f"DEBUG apply_lut: Using built-in OpenCV colormap ID for '{lut_name}': {actual_colormap_for_apply}")

    try:
        # cv2.applyColorMap handles integer IDs and custom LUT arrays (e.g., (256,1,3) uint8)
        colored_frame = cv2.applyColorMap(frame, actual_colormap_for_apply)
    except cv2.error as e:
        print(f"ERROR during cv2.applyColorMap for LUT '{lut_name}': {e}")
        print(f"DEBUG apply_lut: Frame details - shape: {frame.shape}, dtype: {frame.dtype}")
        if isinstance(actual_colormap_for_apply, np.ndarray):
            print(f"DEBUG apply_lut: LUT array details - shape: {actual_colormap_for_apply.shape}, dtype: {actual_colormap_for_apply.dtype}, continuous: {actual_colormap_for_apply.flags.c_contiguous}")
        else:
            print(f"DEBUG apply_lut: LUT ID was: {actual_colormap_for_apply}")
        # Fallback: convert original single-channel frame to BGR to display something
        colored_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) 
        
    return colored_frame


def capture_and_process_video(cap, lut_var, recording_var, out_var, frame_width, frame_height, 
                              flip_var, rotate_var, 
                              desired_capture_fps, # FPS target for this camera
                              capture_stop_event, frame_queue, camera_disconnected_event):
    """
    Captures video frames, performs full processing (reshape, grayscale, flip/rotate, LUT),
    and queues RGB frames for display. Uses desired_capture_fps for loop timing.
    """
    global screenshot_requested

    capture_loop_count = 0 # Local counter for periodic logging
    thread_name = threading.current_thread().name 

    if camera_disconnected_event: # Make sure it's a valid event object
        camera_disconnected_event.clear()

    if DEBUG_MODE: print(f"DEBUG CAPTURE ({thread_name}): Video capture thread started. Target processing FPS: {desired_capture_fps}")

    while not capture_stop_event.is_set():
        capture_loop_count += 1
        start_time = time.time()

        raw_frame_from_cam = None
        ret = False
        try:
            with cap_lock: # Ensure cap_lock is correctly defined and passed
                if cap and cap.isOpened():
                    ret, raw_frame_from_cam = cap.read()
                else:
                    if capture_loop_count % 60 == 1: # Log less frequently
                         if DEBUG_MODE: print("DEBUG CAPTURE: Warning - Camera object not valid or not open in capture loop.")
                    ret = False 
        except Exception as e_capread:
            print(f"ERROR CAPTURE: Exception during cap.read(): {e_capread}")
            traceback.print_exc()
            ret = False
        
        if not ret or raw_frame_from_cam is None:
            if capture_loop_count % 20 == 1: # Log first time and then periodically
                if DEBUG_MODE: print("DEBUG CAPTURE: Warning - Could not read frame. Camera may be disconnected.")
            if camera_disconnected_event:
                camera_disconnected_event.set()
            time.sleep(0.1) # Wait a bit before retrying if camera read fails
            continue

        # For periodic detailed logging of raw frame data
        if capture_loop_count % 100 == 1: 
            if DEBUG_MODE: print(f"DEBUG CAPTURE (Loop {capture_loop_count}): Raw frame from cap.read() - shape: {raw_frame_from_cam.shape}, dtype: {raw_frame_from_cam.dtype}, size: {raw_frame_from_cam.size}")

        h = frame_height # Expected height (e.g., 120 for Lepton, 512 for Boson)
        w = frame_width  # Expected width (e.g., 160 for Lepton, 640 for Boson)
        
        processed_gray_frame = None # This is what we aim to get: (h, w) uint8
        
        current_frame_data = raw_frame_from_cam.copy() # Work on a copy

        # 1. Interpret and reshape raw frame to get an 8-bit Grayscale frame (h, w)
        if current_frame_data.dtype == np.uint8:
            if current_frame_data.shape == (h, w, 3):  # Correct BGR (e.g., Boson)
                if capture_loop_count % 100 == 1 and DEBUG_MODE:
                    print(f"DEBUG CAPTURE: Raw frame is uint8, shaped as BGR ({h},{w},3). Converting to Grayscale.")
                processed_gray_frame = cv2.cvtColor(current_frame_data, cv2.COLOR_BGR2GRAY)

            elif current_frame_data.shape == (h, w):  # Correct Grayscale
                if capture_loop_count % 100 == 1 and DEBUG_MODE:
                    print(f"DEBUG CAPTURE: Raw frame is uint8, shaped as Grayscale ({h},{w}).")
                processed_gray_frame = current_frame_data

            elif current_frame_data.size == h * w * 3:  # Flat BGR data (e.g., Lepton)
                if capture_loop_count % 100 == 1 and DEBUG_MODE:
                    print(f"DEBUG CAPTURE: Raw frame (shape {current_frame_data.shape}, uint8, size {current_frame_data.size}) looks like flat BGR. Reshaping to ({h},{w},3) then to Grayscale.")
                try:
                    bgr_reshaped = current_frame_data.reshape((h, w, 3))
                    processed_gray_frame = cv2.cvtColor(bgr_reshaped, cv2.COLOR_BGR2GRAY)
                except ValueError as e_reshape_bgr:
                    print(f"ERROR CAPTURE: Failed to reshape flat uint8 BGR data: {e_reshape_bgr}")

            elif current_frame_data.size == h * w:  # Flat Grayscale data
                if capture_loop_count % 100 == 1 and DEBUG_MODE:
                    print(f"DEBUG CAPTURE: Raw frame (shape {current_frame_data.shape}, uint8, size {current_frame_data.size}) looks like flat Grayscale. Reshaping to ({h},{w}).")
                try:
                    processed_gray_frame = current_frame_data.reshape((h, w))
                except ValueError as e_reshape_gray:
                    print(f"ERROR CAPTURE: Failed to reshape flat uint8 Grayscale data: {e_reshape_gray}")

            elif capture_loop_count % 100 == 1:
                if DEBUG_MODE:
                    print(f"DEBUG CAPTURE: Raw frame uint8 but unhandled shape/size: {current_frame_data.shape}, size: {current_frame_data.size}. Expected size {h*w} (Gray) or {h*w*3} (BGR).")


        elif current_frame_data.dtype == np.uint16: # Potential 16-bit Grayscale (Y16)
            if current_frame_data.shape == (h, w) or \
               (current_frame_data.ndim == 1 and current_frame_data.size == h * w) or \
               (current_frame_data.ndim == 2 and current_frame_data.size == h * w and (current_frame_data.shape[0] == 1 or current_frame_data.shape[1] == 1)): # Check for flat or (1,N)/(N,1)
                if capture_loop_count % 100 == 1 and DEBUG_MODE: print(f"DEBUG CAPTURE: Raw frame is uint16 (shape {current_frame_data.shape}, size {current_frame_data.size}). Assuming Y16. Reshaping to ({h},{w}) then normalizing.")
                try:
                    y16_shaped = current_frame_data.reshape((h,w)) # Reshape if flat
                    processed_gray_frame = cv2.normalize(y16_shaped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                except ValueError as e_reshape_y16:
                     print(f"ERROR CAPTURE: Failed to reshape or process uint16 Y16 data: {e_reshape_y16}")
            elif capture_loop_count % 100 == 1: # uint16 but unhandled shape/size
                if DEBUG_MODE: print(f"DEBUG CAPTURE: Raw frame uint16 but unhandled shape/size: {current_frame_data.shape}, size: {current_frame_data.size}. Expected size {h*w}.")
        
        elif capture_loop_count % 100 == 1: # Unhandled raw frame type
            if DEBUG_MODE: print(f"DEBUG CAPTURE: Raw frame has unhandled dtype/shape for processing: {current_frame_data.dtype} / {current_frame_data.shape}.")


        # If processed_gray_frame is still None or invalid shape after all attempts, handle error
        if processed_gray_frame is None or \
           not (processed_gray_frame.shape == (h, w) and processed_gray_frame.dtype == np.uint8):
            if capture_loop_count % 10 == 1: # Log more frequently on this error path
                print(f"ERROR CAPTURE: Could not derive a valid ({h},{w}) uint8 grayscale frame. " +
                      f"Final processed_gray_frame state: shape {processed_gray_frame.shape if processed_gray_frame is not None else 'None'}, " +
                      f"dtype {processed_gray_frame.dtype if processed_gray_frame is not None else 'None'}. Sending dummy RGB frame.")
            rgb_to_queue = np.zeros((h, w, 3), dtype=np.uint8) # Black dummy frame
            # Queue the dummy frame and skip further processing for this iteration
            try:
                frame_queue.put_nowait(rgb_to_queue)
            except queue.Full: pass
            time.sleep(max(0, (1.0 / desired_capture_fps) - (time.time() - start_time)))
            continue # Go to next loop iteration

        # 2. Apply Flip
        current_flip = flip_var.get()
        if current_flip == "Horizontal": processed_gray_frame = cv2.flip(processed_gray_frame, 1)
        elif current_flip == "Vertical": processed_gray_frame = cv2.flip(processed_gray_frame, 0)
        elif current_flip == "Both": processed_gray_frame = cv2.flip(processed_gray_frame, -1)

        # 3. Apply Rotation
        current_rotation = rotate_var.get()
        if current_rotation == "90": processed_gray_frame = cv2.rotate(processed_gray_frame, cv2.ROTATE_90_CLOCKWISE)
        elif current_rotation == "180": processed_gray_frame = cv2.rotate(processed_gray_frame, cv2.ROTATE_180)
        elif current_rotation == "270": processed_gray_frame = cv2.rotate(processed_gray_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        # 4. Apply LUT
        current_lut_name = lut_var.get()
        # apply_lut expects CV_8UC1 (grayscale) and returns BGR
        colored_bgr_frame = apply_lut(processed_gray_frame, current_lut_name) 

        # Sanity check output of apply_lut
        if colored_bgr_frame is None or not (len(colored_bgr_frame.shape) == 3 and colored_bgr_frame.shape[2] == 3 and colored_bgr_frame.dtype == np.uint8):
            if capture_loop_count % 10 == 1:
                 print(f"ERROR CAPTURE: apply_lut for '{current_lut_name}' returned invalid frame. Using grayscale as fallback.")
            # Fallback: convert the (valid) processed_gray_frame to BGR if LUT fails
            colored_bgr_frame = cv2.cvtColor(processed_gray_frame, cv2.COLOR_GRAY2BGR)
        
        # 5. Convert final BGR to RGB for the display queue
        rgb_to_queue = cv2.cvtColor(colored_bgr_frame, cv2.COLOR_BGR2RGB)

        # Recording and Screenshot (use colored_bgr_frame for consistency)
        with recording_lock: 
            if recording_var[0] and out_var[0] is not None:
                frame_to_write = colored_bgr_frame
                h_rot, w_rot = colored_bgr_frame.shape[:2]

                # frame_width and frame_height are the dimensions VideoWriter expects.
                # If rotated frame dimensions (w_rot, h_rot) differ from VideoWriter's (frame_width, frame_height),
                # we need to place the rotated frame onto a canvas of the correct size.
                if (w_rot != frame_width or h_rot != frame_height):
                    if DEBUG_MODE:
                        print(f"DEBUG CAPTURE: Rotated frame dim ({w_rot}x{h_rot}) differs from VideoWriter dim ({frame_width}x{frame_height}). Creating canvas.")
                    
                    # Create a black canvas of the VideoWriter's expected dimensions
                    # Note: VideoWriter uses (width, height), OpenCV uses (height, width, channels) for np.zeros
                    canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

                    # Calculate new dimensions for the rotated frame to fit onto the canvas while maintaining aspect ratio
                    aspect_ratio_rotated = w_rot / h_rot
                    target_w = frame_width
                    target_h = int(target_w / aspect_ratio_rotated)

                    if target_h > frame_height:
                        target_h = frame_height
                        target_w = int(target_h * aspect_ratio_rotated)
                    
                    resized_rotated_frame = cv2.resize(colored_bgr_frame, (target_w, target_h))

                    # Calculate top-left position to center the resized frame on the canvas
                    x_offset = (frame_width - target_w) // 2
                    y_offset = (frame_height - target_h) // 2

                    # Place the resized frame onto the canvas
                    canvas[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = resized_rotated_frame
                    frame_to_write = canvas
                
                try:
                    out_var[0].write(frame_to_write) # Write the (potentially adjusted) frame
                except Exception as e_write:
                    print(f"Error writing frame to video: {e_write}")
                    print("Stopping recording due to writing error.")
                    # This is where toggle_recording is called from the capture thread.
                    # It needs to match the new signature of toggle_recording.
                    # We pass None for GUI-related elements as this call isn't meant to update the GUI button directly.
                    # desired_capture_fps, frame_width, and frame_height are already parameters 
                    # of this capture_and_process_video function.
                    
                    # Call toggle_recording to stop the recording state
                    recording_var[0], out_var[0] = toggle_recording(
                        recording_var,          # The recording state list
                        out_var,                # The video writer list
                        None,                   # record_button_widget (no direct GUI update from here)
                        None,                   # original_button_bg (not needed if widget is None)
                        frame_width,            # Current frame_width for this camera session
                        frame_height,           # Current frame_height for this camera session
                        desired_capture_fps,    # Pass the FPS for this thread
                        None                    # root (no direct GUI update from here)
                    )
        
        # Ensure screenshot_requested is handled safely (e.g. declared global if modified)
        # For this function, if it's read-only, it's fine. If modified, it needs 'global screenshot_requested'
        # The original code showed 'global screenshot_requested' at the start of the function
        with screenshot_lock: 
            if screenshot_requested: # This implies screenshot_requested is a global boolean
                screenshot_requested = False # Reset flag immediately after checking
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                screenshot_path = f"screenshot_{timestamp}.png"
                try:
                    cv2.imwrite(screenshot_path, colored_bgr_frame) 
                    print(f"Screenshot saved to: {screenshot_path}")
                except Exception as e_ss:
                    print(f"Error saving screenshot: {e_ss}")

        # Queue the final RGB frame for display
        try:
            frame_queue.put_nowait(rgb_to_queue)
        except queue.Full:
            if capture_loop_count % 60 == 1 and desired_capture_fps > 15 : # Log if queue is often full for faster cameras
                if DEBUG_MODE: print("DEBUG CAPTURE: Frame queue was full. Frame dropped.")
        except Exception as e_queue:
            print(f"ERROR CAPTURE: Failed to queue frame: {e_queue}")
            traceback.print_exc()
        
        # Maintain target frame rate for the capture loop
        processing_time = time.time() - start_time
        sleep_duration = max(0, (1.0 / desired_capture_fps) - processing_time)
        time.sleep(sleep_duration)

    if DEBUG_MODE: print(f"DEBUG CAPTURE ({thread_name}): Video capture thread (Target FPS: {desired_capture_fps}) finished.")

def take_screenshot():
    """Handles screenshot requests."""
    global screenshot_requested
    with screenshot_lock: # Use the screenshot lock
        screenshot_requested = True


# Modify the toggle_recording function definition
def toggle_recording(recording_var, out_var, record_button_widget, original_button_bg, 
                     frame_width, frame_height, 
                     current_desired_capture_fps, # <<< New parameter
                     root):
    """Starts or stops recording and changes the record button color."""
    with recording_lock: # Assuming recording_lock is a global threading.Lock
        if not recording_var[0]:
            # Start recording
            now = datetime.datetime.now()
            output_file = f"flir-{now.strftime('%M%H%S-%d%m%y')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            try:
                if frame_width <= 0 or frame_height <= 0:
                    print(f"Error: Invalid frame dimensions for recording: {frame_width}x{frame_height}")
                    return recording_var[0], out_var[0]
                
                # Use current_desired_capture_fps for the VideoWriter
                # Ensure it's a float and provide a sensible fallback if it's zero or invalid
                fps_for_writer = float(current_desired_capture_fps) if current_desired_capture_fps > 0 else 20.0 
                print(f"DEBUG TOGGLE_RECORDING: Starting video writer for {output_file} with FPS: {fps_for_writer}, Dimensions: {frame_width}x{frame_height}")

                out_var[0] = cv2.VideoWriter(output_file, fourcc, fps_for_writer, (frame_width, frame_height))
                if not out_var[0].isOpened():
                    print(f"Error: Could not open video writer for {output_file}. Check codecs and parameters.")
                    out_var[0] = None
                    return recording_var[0], out_var[0]

            except Exception as e:
                print(f"Error starting recording: {e}")
                traceback.print_exc() # Print full traceback for debugging
                out_var[0] = None 
                return recording_var[0], out_var[0]
            
            recording_var[0] = True
            print(f"Started recording to {output_file}")
            if root and record_button_widget:
                root.after(0, lambda: record_button_widget.config(bg="red")) # Changed from 17 to 0 for immediate feedback
        else:
            # Stop recording (logic remains largely the same)
            recording_var[0] = False
            with out_lock: 
                if out_var[0] is not None: 
                    try:
                        out_var[0].release()
                        if DEBUG_MODE: print("DEBUG TOGGLE_RECORDING: Video writer released.")
                    except Exception as e:
                        print(f"Error releasing video writer: {e}")
                    out_var[0] = None 
            print("Stopped recording")
            if root and record_button_widget: 
                root.after(0, lambda: record_button_widget.config(bg=original_button_bg))
    return recording_var[0], out_var[0]


def exit_program(cap, out_var):
    """Exits the program, releasing resources."""
    capture_stop_event.set() # Signal the capture thread to stop
    # Give the thread a moment to finish
    time.sleep(0.5) # Adjust sleep time as needed

    with cap_lock:
        if cap: # Check if cap is valid before releasing
            try:
                cap.release()
            except Exception as e:
                 print(f"Error releasing camera on exit: {e}")
            cap = None

    with out_lock:
        if out_var[0] is not None: # Check if out_var[0] is not None before releasing
            try:
                out_var[0].release()
            except Exception as e:
                 print(f"Error releasing video writer on exit: {e}")
            out_var[0] = None

    exit_event.set()
    if root:
        root.quit() # Use root.quit() to exit the Tkinter main loop gracefully


def update_display(video_label, status_label, camera_disconnected_event):
    global update_display_call_count # Make sure this global counter is still active
    global current_display_delay_ms # Access the global delay variable

    update_display_call_count += 1
    # Print less frequently to avoid too much spam, e.g., every 20 calls
    if update_display_call_count % 20 == 0: # Reduced from 10 to 20 for less spam
        if DEBUG_MODE: print(f"DEBUG UPDATE_DISPLAY: Call number {update_display_call_count}, Approx Q-size: {frame_queue.qsize()}")

    try:
        frame_rgb = frame_queue.get_nowait() # Expecting RGB NumPy array

        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.dtype != np.uint8 or frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            if update_display_call_count % 20 == 0: # Log error less frequently
                print(f"ERROR UPDATE_DISPLAY: Received invalid frame from queue. Type: {type(frame_rgb)}, Dtype: {frame_rgb.dtype if isinstance(frame_rgb, np.ndarray) else 'N/A'}, Shape: {frame_rgb.shape if isinstance(frame_rgb, np.ndarray) else 'N/A'}")
            if not exit_event.is_set():
                 root.after(100, update_display, video_label, status_label, camera_disconnected_event)
            return

        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        # Explicitly delete the PIL image object after PhotoImage is created
        # This helps ensure its resources are released if PhotoImage makes its own copy
        # or if the reference counting is tricky.
        del img_pil 

        video_label.configure(image=imgtk)
        video_label.imgtk = imgtk 

        status_label.config(text="Status: Connected", fg="green")
        camera_disconnected_event.clear()

    except queue.Empty:
        pass 
    except Exception as e:
        print(f"ERROR in update_display (Call {update_display_call_count}): {e}")
        if update_display_call_count % 1 == 0 or True: # Ensure traceback prints
            traceback.print_exc() 
        status_label.config(text="Status: Display Error", fg="orange")

    if camera_disconnected_event.is_set():
        status_label.config(text="Status: Camera Disconnected", fg="red")
        video_label.configure(image='') 
        video_label.imgtk = None 

    if not exit_event.is_set():
        root.after(current_display_delay_ms, 
            update_display, video_label, status_label, camera_disconnected_event)
    else:
        if DEBUG_MODE: print(f"DEBUG UPDATE_DISPLAY (Call {update_display_call_count}): Exit event set, not rescheduling.")

def load_config():
    """Loads configuration from the JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                print(f"Loaded configuration from {CONFIG_FILE}")
                return config
        except Exception as e:
            print(f"Error loading configuration from {CONFIG_FILE}: {e}")
            return {} # Return empty config on error
    else:
        print(f"Configuration file {CONFIG_FILE} not found. Using default settings.")
        return {} # Return empty config if file doesn't exist

def save_config(lut_name, camera_device, flip_state, rotate_state):
    """Saves current configuration to the JSON file."""
    config = {
        'lut_name': lut_name,
        'camera_device': camera_device,
        'flip_state': flip_state,
        'rotate_state': rotate_state
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
            print(f"Saved configuration to {CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving configuration to {CONFIG_FILE}: {e}")


def get_camera_type_from_device(device_node):
    """Attempts to determine camera type based on device node."""
    context = pyudev.Context()
    try:
        device = pyudev.Device.from_device_node(context, device_node)
        parent = device.find_parent(subsystem='usb', device_type='usb_device')
        if parent:
            vendor_id = parent.properties.get('ID_VENDOR_ID')
            product_id = parent.properties.get('ID_MODEL_ID')
            model = parent.properties.get('ID_MODEL', '')

            if vendor_id == FLIR_VENDOR_ID:
                 if 'Boson' in model: return 'BOSON'
                 # Add other FLIR models if needed
            elif vendor_id == '1e4e' and product_id == LEPTON_PRODUCT_ID:
                return 'LEPTON3' # Assuming Cubeternet is Lepton3
            elif vendor_id == AMPBANK_VENDOR_ID and product_id == AMPBANK_PRODUCT_ID:
                 return 'AMPBANK'
    except Exception as e:
        print(f"Error determining camera type for {device_node}: {e}")

    return 'Unknown' # Default if type cannot be determined


def open_camera(device_node, camera_type):
    """Opens the camera and sets resolution based on detected type."""
    cap = cv2.VideoCapture(device_node, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Error: Could not open video device {device_node}")
        return None, 0, 0

    # Set frame properties based on the camera type
    # Use hardcoded resolutions based on the determined camera_type string
    frame_width, frame_height = CAMERA_RESOLUTIONS.get(camera_type, CAMERA_RESOLUTIONS.get('BOSON')) # Default to BOSON if type is unknown

    print(f"Attempting to set resolution to {frame_width}x{frame_height} for camera type {camera_type}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Verify if the resolution was set correctly (optional but recommended)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != frame_width or actual_height != frame_height:
        print(f"Warning: Could not set resolution to {frame_width}x{frame_height}. Using camera's default: {actual_width}x{actual_height}")
        frame_width = actual_width
        frame_height = actual_height

    return cap, frame_width, frame_height


def main(initial_lut_name, initial_camera_type_hint='BOSON'):
    global root # For Tkinter root, current VideoCapture, and current capture thread
    global current_camera_type, current_display_delay_ms # Module-level globals
    
    # These are event/queue objects, typically defined globally and used by threads
    global exit_event, capture_stop_event, camera_disconnected_event, frame_queue 
    
    # These are locks, defined globally
    global cap_lock 
    
    # Initialize variables that 'switch_camera' will modify via 'nonlocal'
    # These MUST be defined BEFORE 'def switch_camera(...):'
    cap = None
    current_camera_index_node = None # You likely already initialize this or similar
    frame_width = 0                  # You likely already initialize this
    frame_height = 0                 # You likely already initialize this
    capture_thread = None            # << Initialize here
    desired_capture_fps = 15         # Default, will be updated based on camera

    # Variables local to main, potentially modified by nested switch_camera via nonlocal
    # or passed to lambdas that capture their current state.
    # Note: current_camera_index_node, frame_width, frame_height, desired_capture_fps
    # are now primarily managed *within* main and passed around or updated by switch_camera.
    # For switch_camera to update them if they are local to main, it needs 'nonlocal'.

    if DEBUG_MODE: print("DEBUG: Entering main function...")

    if DEBUG_MODE: print("DEBUG: Loading configuration...")
    config = load_config()
    initial_preferred_lut = config.get('lut_name', initial_lut_name)
    initial_camera_device_node_from_config = config.get('camera_device', None)
    initial_flip_state = config.get('flip_state', 'None')
    initial_rotate_state = config.get('rotate_state', '0')
    if DEBUG_MODE: print(f"DEBUG: Config loaded. Preferred LUT: {initial_preferred_lut}, Camera Node from Config: {initial_camera_device_node_from_config}")

    if DEBUG_MODE: print("DEBUG: Getting available video devices...")
    available_camera_info_list = get_video_devices_for_flir()
    if DEBUG_MODE: print(f"DEBUG: Available camera info list: {available_camera_info_list}")

    if not available_camera_info_list:
        print("ERROR: No thermal cameras detected. Exiting application.")
        messagebox.showerror("Camera Error", "No thermal cameras detected. The application will now exit.")
        return

    # Camera Selection Logic (prioritizing Boson for testing as per last logs)
    default_camera_info = None
    if DEBUG_MODE: print(f"DEBUG MAIN: Starting camera selection. Configured device node: {initial_camera_device_node_from_config}")
    if DEBUG_MODE: print(f"DEBUG MAIN: CAMERA_RESOLUTIONS dictionary keys available: {list(CAMERA_RESOLUTIONS.keys())}")

    # Test prioritization (e.g., for Boson) - can be removed for normal operation
    if DEBUG_MODE: print("DEBUG MAIN: Attempting to prioritize BOSON for initial startup...")
    for cam_info_iter in available_camera_info_list:
        if cam_info_iter.get('camera_type') == 'BOSON':
            default_camera_info = cam_info_iter
            if DEBUG_MODE: print(f"DEBUG MAIN: Prioritized and selected BOSON: {default_camera_info}")
            break
    
    if default_camera_info:
        if DEBUG_MODE: print(f"DEBUG: SUCCESS (Camera prioritized) - default_camera_info is: {default_camera_info}")
    else:
        if DEBUG_MODE: print("DEBUG MAIN: Prioritized camera not found or logic skipped, proceeding with normal selection.")
        if initial_camera_device_node_from_config: # Stage 1: find by configured device node
            for cam_info_cfg in available_camera_info_list:
                if cam_info_cfg.get('device_node') == initial_camera_device_node_from_config:
                    default_camera_info = cam_info_cfg
                    if DEBUG_MODE: print(f"DEBUG: Stage 1 - Selected camera from config: {default_camera_info}")
                    break
            if not default_camera_info and DEBUG_MODE:
                 print(f"DEBUG: Stage 1 - Configured camera {initial_camera_device_node_from_config} not found.")

        if not default_camera_info: # Stage 2: find by known thermal type
            if DEBUG_MODE: print("DEBUG: Stage 2 - Config camera not used/found. Searching for known thermal types...")
            for cam_info_type_sel in available_camera_info_list:
                camera_type_from_device = cam_info_type_sel.get('camera_type', '').strip()
                if DEBUG_MODE: print(f"DEBUG: Stage 2 - Checking device: {cam_info_type_sel.get('device_node')}, type: '{camera_type_from_device}'")
                if camera_type_from_device in CAMERA_RESOLUTIONS:
                    default_camera_info = cam_info_type_sel
                    if DEBUG_MODE: print(f"DEBUG: Stage 2 - Selected first suitable thermal camera (type '{camera_type_from_device}'): {default_camera_info}")
                    break
            if not default_camera_info and DEBUG_MODE: print("DEBUG: Stage 2 - No camera found with a type listed in CAMERA_RESOLUTIONS.")
                 
        if not default_camera_info and available_camera_info_list: # Stage 3: fallback
            if DEBUG_MODE: print("DEBUG: Stage 3 - Falling back to the first available video device.")
            default_camera_info = available_camera_info_list[0]
            if DEBUG_MODE: print(f"DEBUG: Stage 3 - Selected first available V4L device as fallback: {default_camera_info}")

    if not default_camera_info:
        print("ERROR: No suitable thermal camera could be determined. Exiting application.")
        messagebox.showerror("Camera Error", "No suitable thermal camera could be selected. The application will now exit.")
        return
    
    if DEBUG_MODE: print(f"DEBUG: SUCCESS (final selection) - default_camera_info has been set to: {default_camera_info}")

    # Initialize main's state variables for the selected camera
    current_camera_index_node = default_camera_info['device_node']
    current_camera_type = default_camera_info.get('camera_type', 'Unknown')

    desired_capture_fps = TARGET_PROCESSING_FPS.get(current_camera_type, 15)
    if DEBUG_MODE: print(f"DEBUG MAIN: Setting target capture processing FPS to {desired_capture_fps} for camera type '{current_camera_type}'.")

    display_fps_target = TARGET_DISPLAY_FPS.get(current_camera_type, 10)
    current_display_delay_ms = int(1000.0 / display_fps_target)
    if DEBUG_MODE: print(f"DEBUG MAIN: Initial display delay set to {current_display_delay_ms}ms for {display_fps_target} FPS display.")

    if DEBUG_MODE: print(f"DEBUG: Attempting to open camera device: {current_camera_index_node} with identified type: {current_camera_type}")
    cap, frame_width, frame_height = open_camera(current_camera_index_node, current_camera_type)

    if cap is None:
        print(f"ERROR: Failed to open camera {current_camera_index_node}. Exiting application.")
        messagebox.showerror("Camera Error", f"Failed to open camera: {current_camera_index_node}.\nThe application will now exit.")
        return
    if DEBUG_MODE: print(f"DEBUG: Camera {current_camera_index_node} opened. Resolution: {frame_width}x{frame_height}")

    out_var = [None]
    recording_var = [False]

    # Define switch_camera (nested function)
    def switch_camera(*args_trace):
        nonlocal cap, current_camera_index_node, frame_width, frame_height, capture_thread, desired_capture_fps
        global current_camera_type, capture_stop_event, camera_disconnected_event, frame_queue, cap_lock, current_display_delay_ms

        selected_camera_display_str = camera_var.get()
        current_active_thread_name = capture_thread.name if capture_thread and hasattr(capture_thread, 'name') else "NoCurrentCaptureThread"
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Camera switch triggered for: {selected_camera_display_str}. Current active thread: {current_active_thread_name}")
        
        try:
            selected_device_node = selected_camera_display_str.split(' ')[0]
        except IndexError:
            if DEBUG_MODE: print(f"ERROR SWITCH_CAMERA: Could not parse device node from '{selected_camera_display_str}'")
            if status_label: status_label.config(text="Error: Invalid camera selection.", fg="red")
            return

        if selected_device_node == current_camera_index_node:
            if DEBUG_MODE: print("DEBUG SWITCH_CAMERA: Selected camera is already active. No change.")
            return

        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Attempting to switch from '{current_camera_index_node}' to '{selected_device_node}'")
        if status_label: status_label.config(text=f"Switching to {selected_device_node}...", fg="orange")

        old_thread_to_join = capture_thread 
        old_thread_name = old_thread_to_join.name if old_thread_to_join and hasattr(old_thread_to_join, 'name') else "UnknownOldThread"
        old_camera_node_being_stopped = current_camera_index_node 
        old_camera_type_being_stopped = current_camera_type 

        if old_thread_to_join and old_thread_to_join.is_alive():
            if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Signaling old capture thread ({old_thread_name}) for camera {old_camera_node_being_stopped} to stop.")
            capture_stop_event.set()
            with cap_lock:
                if cap and cap.isOpened(): 
                    if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Releasing VideoCapture for old camera ({old_camera_node_being_stopped}).")
                    try: cap.release()
                    except Exception as e: print(f"ERROR SWITCH_CAMERA: Exception while releasing old cap: {e}")
            
            if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Joining old capture thread ({old_thread_name}). Timeout: 5.0s")
            old_thread_to_join.join(timeout=5.0) 
            if old_thread_to_join.is_alive():
                if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Warning - Old capture thread ({old_thread_name}) for {old_camera_node_being_stopped} did not join in time! It is a daemon.")
            else:
                if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Old capture thread ({old_thread_name}) for {old_camera_node_being_stopped} successfully joined.")
        
        capture_stop_event.clear() 
        if camera_disconnected_event: camera_disconnected_event.clear()
        cap = None 
        
        pause_duration = 2.0 
        if "LEPTON" in old_camera_type_being_stopped.upper():
            if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Switched away from a Lepton ({old_camera_type_being_stopped}). Pausing longer (5.0s).")
            pause_duration = 5.0
        else:
            if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Pausing for {pause_duration:.1f}s after stopping {old_camera_node_being_stopped}.")
        time.sleep(pause_duration)

        new_camera_details = None
        for cam_info_iter in available_camera_info_list: 
            if cam_info_iter.get('device_node') == selected_device_node:
                new_camera_details = cam_info_iter
                break
        if not new_camera_details:
            print(f"ERROR SWITCH_CAMERA: Details not found for {selected_device_node}")
            if status_label: status_label.config(text=f"Error finding {selected_device_node}", fg="red")
            return

        new_cam_type_str = new_camera_details.get('camera_type', 'Unknown')
        new_cam_model_str = new_camera_details.get('model', 'Unknown Model')
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: New camera details: Node={selected_device_node}, Type='{new_cam_type_str}', Model='{new_cam_model_str}'")
        
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Opening new camera '{selected_device_node}' (type: '{new_cam_type_str}')")
        new_cap_obj, new_frame_width, new_frame_height = open_camera(selected_device_node, new_cam_type_str)

        if new_cap_obj is None or not new_cap_obj.isOpened():
            print(f"ERROR SWITCH_CAMERA: Failed to open new camera '{selected_device_node}'")
            if status_label: status_label.config(text=f"Failed to open {selected_device_node}", fg="red")
            if new_cap_obj: new_cap_obj.release() 
            return 
        
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: New camera '{selected_device_node}' opened. Resolution: {new_frame_width}x{new_frame_height}")

        cap = new_cap_obj
        current_camera_index_node = selected_device_node 
        current_camera_type = new_cam_type_str
        frame_width = new_frame_width   
        frame_height = new_frame_height 
        
        new_desired_capture_fps_for_thread = TARGET_PROCESSING_FPS.get(current_camera_type, 15)
        desired_capture_fps = new_desired_capture_fps_for_thread # Update main's 'desired_capture_fps'
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Set main's desired_capture_fps & thread target to {desired_capture_fps} for '{current_camera_type}'.")

        new_display_fps_target = TARGET_DISPLAY_FPS.get(current_camera_type, 10)
        current_display_delay_ms = int(1000.0 / new_display_fps_target)
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: Updated display delay to {current_display_delay_ms}ms for {new_display_fps_target} FPS display for '{current_camera_type}'.")
        
        thread_args_tuple = (
            cap, lut_var, recording_var, out_var, frame_width, frame_height, 
            flip_var, rotate_var, new_desired_capture_fps_for_thread, # Use the specific FPS for this thread
            capture_stop_event, frame_queue, camera_disconnected_event
        )
        
        if DEBUG_MODE: print("DEBUG SWITCH_CAMERA: Starting new capture thread...")
        capture_thread = threading.Thread(
            target=capture_and_process_video, 
            args=thread_args_tuple,
            name=f"CaptureThread-{new_cam_type_str}"
        )
        capture_thread.daemon = True
        capture_thread.start()
        if DEBUG_MODE: print(f"DEBUG SWITCH_CAMERA: New capture thread ({capture_thread.name}) started for {selected_device_node}")
        if status_label: status_label.config(text=f"Status: Switched to {selected_device_node} ({new_cam_model_str})", fg="blue")

    if DEBUG_MODE: print("DEBUG: Initializing Tkinter root window...")
    root = tk.Tk()
    root.title("Thermal Camera Control")
    if DEBUG_MODE: print("DEBUG: Tkinter root window created.")

    control_frame = Frame(root)
    control_frame.pack(pady=10, padx=10, fill=tk.X)

    camera_var = StringVar(root)
    camera_display_strings_list = []
    if available_camera_info_list:
        for cam_info in available_camera_info_list:
            camera_display_strings_list.append(
                f"{cam_info.get('device_node', 'N/A')} ({cam_info.get('model', 'Unknown Model')}) - {cam_info.get('camera_type', 'Unknown Type')}"
            )
        unique_sorted_camera_options = sorted(list(set(camera_display_strings_list)))
    else: 
        unique_sorted_camera_options = ["No Cameras Detected"]
        print("ERROR MAIN: No available cameras to populate dropdown.")

    initial_display_string_for_var = "Error: No Camera" 
    if default_camera_info:
        initial_display_string_for_var = f"{default_camera_info.get('device_node', 'N/A')} ({default_camera_info.get('model', 'Unknown Model')}) - {default_camera_info.get('camera_type', 'Unknown Type')}"
    
    if initial_display_string_for_var in unique_sorted_camera_options:
        camera_var.set(initial_display_string_for_var)
    elif unique_sorted_camera_options: 
        print(f"Warning: Initial camera string '{initial_display_string_for_var}' not in unique options. Defaulting to '{unique_sorted_camera_options[0]}'.")
        camera_var.set(unique_sorted_camera_options[0])
    else: 
        camera_var.set("No Cameras Available")

    Label(control_frame, text="Camera:").pack(side="left", padx=(0,5))
    initial_menu_value = camera_var.get()
    options_for_menu = unique_sorted_camera_options
    if not unique_sorted_camera_options or (len(unique_sorted_camera_options)==1 and unique_sorted_camera_options[0]=="No Cameras Detected"):
        options_for_menu = [initial_menu_value] if initial_menu_value else ["N/A"]
        if initial_menu_value not in options_for_menu : initial_menu_value = options_for_menu[0]
        camera_var.set(initial_menu_value)

    if initial_menu_value not in options_for_menu and options_for_menu:
        print(f"Warning: camera_var value '{initial_menu_value}' not in final options list. Setting to '{options_for_menu[0]}'.")
        camera_var.set(options_for_menu[0])
        initial_menu_value = camera_var.get()
    elif not options_for_menu:
        options_for_menu = ["N/A"]
        camera_var.set("N/A")
        initial_menu_value = "N/A"

    camera_menu = OptionMenu(control_frame, camera_var, initial_menu_value, *options_for_menu)
    camera_menu.pack(side="left", padx=5)
    camera_var.trace_add("write", switch_camera) 
    if DEBUG_MODE: print("DEBUG: Camera selection trace added to call switch_camera with 'write' mode.")
    
    # Record Button
    record_button = Button(control_frame, text="Record")
    try:
        original_button_bg = record_button.cget('background') 
    except tk.TclError: 
        original_button_bg = root.cget('background') 
        if original_button_bg == record_button.winfo_class(): 
             original_button_bg = "SystemButtonFace" 
    record_button.config(command=lambda: toggle_recording(
        recording_var, out_var, record_button, original_button_bg,
        frame_width, frame_height, desired_capture_fps, # Pass current values from main's scope
        root
    ))
    record_button.pack(side="left", padx=5)

    # Screenshot, Flip, Rotate, LUT controls
    Button(control_frame, text="Screenshot", command=take_screenshot).pack(side="left", padx=5)

    flip_options = ["None", "Horizontal", "Vertical", "Both"]
    flip_var = StringVar(root)
    flip_var.set(initial_flip_state)
    Label(control_frame, text="Flip:").pack(side="left", padx=(10,0))
    flip_menu = OptionMenu(control_frame, flip_var, *flip_options)
    flip_menu.pack(side="left", padx=5)

    rotate_options = ["0", "90", "180", "270"]
    rotate_var = StringVar(root)
    rotate_var.set(initial_rotate_state)
    Label(control_frame, text="Rotation:").pack(side="left", padx=(10,0))
    rotate_menu = OptionMenu(control_frame, rotate_var, *rotate_options)
    rotate_menu.pack(side="left", padx=5)

    lut_var = StringVar(root)
    lut_keys_sorted = sorted(LUTS.keys())
    if not lut_keys_sorted: lut_keys_sorted = ["N/A"]
    
    if initial_preferred_lut in lut_keys_sorted:
        lut_var.set(initial_preferred_lut)
    elif 'WHITEHOT' in lut_keys_sorted:
        lut_var.set('WHITEHOT')
    elif lut_keys_sorted[0] != "N/A":
        lut_var.set(lut_keys_sorted[0])
    else: # Only "N/A" is an option
        lut_var.set("N/A")

    Label(control_frame, text="LUT:").pack(side="left", padx=(10,0))
    lut_menu = OptionMenu(control_frame, lut_var, lut_var.get(), *lut_keys_sorted)
    lut_menu.config(width=15) 
    lut_menu.pack(side="left", padx=5)
    
    # Action Buttons Frame
    action_button_frame = Frame(root)
    action_button_frame.pack(pady=5, padx=10, fill=tk.X)
    Button(action_button_frame, text="Save Config", command=lambda: save_config(
        lut_var.get(), current_camera_index_node, flip_var.get(), rotate_var.get()
    )).pack(side="left", padx=5)
    Button(action_button_frame, text="Exit Program", command=lambda: exit_program(cap, out_var)).pack(side="left", padx=5)

    # Status and Video Labels
    status_label = Label(root, text="Status: Initializing...", fg="black", anchor="w")
    status_label.pack(pady=5, padx=10, fill=tk.X)
    video_label = Label(root)
    video_label.pack(padx=10, pady=10)

    if DEBUG_MODE: print("DEBUG: GUI elements configured.")
    
    initial_thread_name = f"CaptureThread-{current_camera_type}"
    if DEBUG_MODE: print(f"DEBUG MAIN: Preparing to start initial capture thread named: {initial_thread_name}")
    if DEBUG_MODE: print("DEBUG: Starting video capture thread...")
    
    capture_thread = threading.Thread(target=capture_and_process_video, args=(
        cap, lut_var, recording_var, out_var, frame_width, frame_height, 
        flip_var, rotate_var, desired_capture_fps, 
        capture_stop_event, frame_queue, camera_disconnected_event),
        name=initial_thread_name)
    capture_thread.daemon = True
    capture_thread.start()
    if DEBUG_MODE: print(f"DEBUG MAIN: Initial capture thread ({capture_thread.name}) has been told to start.")

    if DEBUG_MODE: print("DEBUG: Starting display update schedule...")
    update_display(video_label, status_label, camera_disconnected_event) # Ensure status_label is defined
    if DEBUG_MODE: print("DEBUG: Display update scheduled.")

    root.protocol("WM_DELETE_WINDOW", lambda: exit_program(cap, out_var))

    if DEBUG_MODE: print("DEBUG: Starting Tkinter mainloop...")
    root.mainloop()
    if DEBUG_MODE: print("DEBUG: Tkinter mainloop exited.")

    # Final cleanup
    if capture_thread and capture_thread.is_alive():
        if DEBUG_MODE: print("DEBUG MAIN: Mainloop exited, ensuring capture_stop_event is set for thread:", capture_thread.name)
        capture_stop_event.set() # Signal before attempting to release cap
        # Release camera and writer directly here if exit_program might not have run or fully cleaned cap
        # However, exit_program should handle this. This is more of a fallback.
        with cap_lock:
            if cap and cap.isOpened():
                if DEBUG_MODE: print(f"DEBUG MAIN: Releasing camera ({current_camera_index_node}) in main thread final cleanup.")
                try: cap.release()
                except Exception as e: print(f"Error releasing cap in main final cleanup: {e}")
                cap = None # Clear main's reference

        if DEBUG_MODE: print(f"DEBUG MAIN: Waiting for capture thread ({capture_thread.name}) to join...")
        capture_thread.join(timeout=3.0) # Increased timeout slightly
        if capture_thread.is_alive():
            if DEBUG_MODE: print(f"DEBUG MAIN: Capture thread ({capture_thread.name}) did not join in time.")
        else:
            if DEBUG_MODE: print(f"DEBUG MAIN: Capture thread ({capture_thread.name}) joined successfully.")
    elif capture_thread:
         if DEBUG_MODE: print(f"DEBUG MAIN: Capture thread ({capture_thread.name}) was already finished after mainloop exit.")
    else: # Should not happen if thread was started
        if DEBUG_MODE: print("DEBUG MAIN: No active capture thread object at final cleanup.")

    # out_var is released by exit_program and toggle_recording
    if DEBUG_MODE: print("DEBUG: Main function cleanup complete.")

# Argparse Setup
if __name__ == "__main__":
    if DEBUG_MODE: print("DEBUG: Inside if __name__ == '__main__' block.")
    # Load custom LUTs here so they are available for argparse choices
    if DEBUG_MODE: print("DEBUG: About to call load_custom_luts() from __main__.")
    load_custom_luts()
    if DEBUG_MODE: print("DEBUG: Returned from load_custom_luts() in __main__.")
    parser = argparse.ArgumentParser(description="Apply a false-color LUT to a video stream from a thermal camera.")
    # lut_name argument still exists but will be overridden by config if available
    parser.add_argument(
        '--lut',
        # choices=LUTS.keys(), # This might fail if LUTS isn't populated yet or if custom LUTs add too many
        default='WHITEHOT',
        help='Name of the initial LUT to apply.'
    )
    # Add any other arguments your script expects

    if DEBUG_MODE: print("DEBUG: About to parse arguments.")
    try:
        args = parser.parse_args()
        if DEBUG_MODE: print(f"DEBUG: Arguments parsed: {args}")
    except Exception as e:
        if DEBUG_MODE: print(f"DEBUG: Error parsing arguments: {e}")
        import sys
        sys.exit(1) # Exit if argument parsing fails

    if DEBUG_MODE: print("DEBUG: About to call main() function.")
    try:
        # Ensure the argument name here matches what your main() function expects.
        # The original main() was: main(initial_lut_name, initial_camera_type_hint='BOSON')
        # So, you'd pass args.lut to initial_lut_name.
        main(initial_lut_name=args.lut) 
        if DEBUG_MODE: print("DEBUG: Returned from main() function normally.")
    except Exception as e:
        if DEBUG_MODE: print(f"DEBUG: An error occurred during main() execution: {e}")
        import traceback
        traceback.print_exc() # Print full traceback if main crashes

    if DEBUG_MODE: print("DEBUG: End of if __name__ == '__main__' block.")
