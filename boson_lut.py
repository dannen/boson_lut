import cv2
import numpy as np
import argparse
import datetime
import tkinter as tk
from tkinter import Button, Label, Frame, OptionMenu, StringVar
from PIL import Image, ImageTk
import os
import pyudev
import threading
import time
import ast

# --- Constants ---

LUTS = {
    'WHITEHOT': cv2.COLORMAP_BONE,
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

FLIR_VENDOR_ID = '09cb'  # FLIR vendor ID
LEPTON_PRODUCT_ID = '0100'  # FLIR Lepton (Cubeternet WebCam) product ID
AMPBANK_VENDOR_ID = '3474'  # Ampbank vendor ID
AMPBANK_PRODUCT_ID = '4321'  # Ampbank product ID

# --- Thread Locks ---
cap_lock = threading.Lock()
out_lock = threading.Lock()
recording_lock = threading.Lock()

# --- Helper Functions ---

def load_custom_luts():
    """Scans the current working directory for .lut files and loads them into the LUTS dictionary."""
    for filename in os.listdir('.'): 
        if filename.endswith('.lut'):
            try:
                with open(filename, 'r') as f:
                    # Expect the file to contain only a list of tuples
                    lut_data = f.read().strip()
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
                print(f"Error loading LUT file {filename}: {e}")
            except Exception as e:
                print(f"Unexpected error loading LUT file {filename}: {e}")


def get_video_devices_for_flir():
    """Detects available FLIR cameras using pyudev."""
    context = pyudev.Context()
    flir_devices = []
    for device in context.list_devices(subsystem='video4linux'):
        parent = device.find_parent(subsystem='usb', device_type='usb_device')
        if parent is None:
            continue
        vendor_id = parent.properties.get('ID_VENDOR_ID')
        product_id = parent.properties.get('ID_MODEL_ID')
        model = parent.properties.get('ID_MODEL', '')

        if vendor_id == FLIR_VENDOR_ID:
            print(f"Found FLIR camera: {model}, Device: {device.device_node}")
            flir_devices.append(device.device_node)
        elif vendor_id == '1e4e' and product_id == LEPTON_PRODUCT_ID:
            print(
                f"Found FLIR Lepton camera (Cubeternet WebCam): {model}, Device: {device.device_node}")
            flir_devices.append(device.device_node)
        elif vendor_id == AMPBANK_VENDOR_ID and product_id == AMPBANK_PRODUCT_ID:
            print(f"Found Ampbank camera: {model}, Device: {device.device_node}")
            flir_devices.append(device.device_node)

    if not flir_devices:
        print("FLIR camera not found")
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
    black_to_white = np.linspace((0, 0, 0), (255, 255, 255), BLACK_TO_WHITE_STEP).astype(np.uint8)
    color_gradient = np.linspace(*gradient_colors).astype(np.uint8)
    custom_colors = np.concatenate((black_to_white, color_gradient))

    # Ensure we have exactly 256 colors by interpolating if necessary
    if len(custom_colors) != 256:
        custom_colors = np.linspace(custom_colors[0], custom_colors[-1], 256, dtype=np.uint8)

    # Create a custom LUT with 256 entries
    custom_lut = custom_colors.reshape((256, 1, 3))
    return custom_lut


def apply_lut(frame, lut_name):
    """Applies the selected LUT to a video frame."""
    if lut_name not in LUTS:
        raise ValueError(f"Unsupported LUT: {lut_name}")

    colormap = LUTS[lut_name]
    if callable(colormap):
        colormap = colormap()
    if isinstance(colormap, np.ndarray):
        # Apply custom LUT using LUT transformation
        colored_frame = cv2.LUT(frame, colormap)
    else:
        # Apply built-in OpenCV colormap
        colored_frame = cv2.applyColorMap(frame, colormap)
    return colored_frame


def capture_and_process_video(cap, lut_var, recording_var, out_var, frame_width, frame_height, video_label, record_button, flip_var, exit_event, rotate_var):
    """Captures video frames, applies the LUT, and handles recording and screenshots."""
    global screenshot_requested
    last_frame = None
    imgtk = ImageTk.PhotoImage(image=Image.new('RGB', (frame_width, frame_height)))  # Create PhotoImage outside the loop
    video_label.configure(image=imgtk)
    video_label.imgtk = imgtk

    while not exit_event.is_set():
        start_time = time.time()

        with cap_lock:
            ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from video stream")
            break

        current_lut = lut_var.get()
        try:
            # --- Apply Flip ---
            flip_value = flip_var.get()
            if flip_value == "Horizontal":
                frame = cv2.flip(frame, 1)
            elif flip_value == "Vertical":
                frame = cv2.flip(frame, 0)
            elif flip_value == "Both":
                frame = cv2.flip(frame, -1)  # Flip both horizontally and vertically

            # --- Rotate image ---
            rotation_value = rotate_var.get()
            if rotation_value == "90":
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_value == "180":
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation_value == "270":
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # --- End rotation ---

            colored_frame = apply_lut(frame, current_lut)
        except ValueError as e:
            print(f"Error applying LUT: {e}")
            continue  # Skip to the next frame

        with recording_lock:
            if recording_var[0] and out_var[0]:
                with out_lock:
                    try:
                        out_var[0].write(colored_frame)
                    except Exception as e:
                        print(f"Error writing frame to video: {e}")
                        recording_var[0], out_var[0] = toggle_recording(
                            recording_var, out_var, record_button, frame_width, frame_height)  # Stop recording

        # Convert the frame to an image that Tkinter can use
        if last_frame is None or not np.array_equal(last_frame, colored_frame):
            try:
                img = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)  # Create a new PhotoImage
                video_label.configure(image=imgtk)  # Update the label's image
                video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
                last_frame = colored_frame.copy()
            except Exception as e:
                print(f"Error displaying frame: {e}")

        if screenshot_requested:
            screenshot_requested = False
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            screenshot_path = f"screenshot_{timestamp}.png"
            try:
                cv2.imwrite(screenshot_path, colored_frame)
                print(f"Screenshot saved to: {screenshot_path}")
            except Exception as e:
                print(f"Error saving screenshot: {e}")

        # Maintain a steady frame rate
        time.sleep(max(0, 1/30 - (time.time() - start_time)))  # 30 FPS limit


def take_screenshot():
    """Handles screenshot requests."""
    global screenshot_requested
    screenshot_requested = True


def toggle_recording(recording_var, out_var, record_button, frame_width, frame_height):
    """Starts or stops recording and changes the record button color."""
    with recording_lock:
        if not recording_var[0]:
            # Start recording
            now = datetime.datetime.now()
            output_file = f"flir-{now.strftime('%M%H%S-%d%m%y')}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            try:
                out_var[0] = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
            except Exception as e:
                print(f"Error starting recording: {e}")
                return recording_var[0], out_var[0]
            recording_var[0] = True
            print(f"Started recording to {output_file}")
            record_button.config(bg="red")  # Turn button red
        else:
            # Stop recording
            recording_var[0] = False
            if out_var[0]:
                with out_lock:
                    out_var[0].release()
                    out_var[0] = None  # Set out to None when stopping recording
            print("Stopped recording")
            record_button.config(bg=root.cget('bg'))  # Reset button color
    return recording_var[0], out_var[0]


def exit_program(cap, out_var):
    """Exits the program, releasing resources."""
    with cap_lock:
        cap.release()
    with out_lock:
        if out_var[0]:
            out_var[0].release()
    exit_event.set()
    root.destroy()

# --- Main Function ---

def main(lut_name, camera_type='BOSON'):
    """Main function that sets up the application."""
    global exit_event
    exit_event = threading.Event()
    global screenshot_requested
    screenshot_requested = False

    # Load custom LUTs from the current directory
    load_custom_luts()

    available_cameras = get_video_devices_for_flir()
    if not available_cameras:
        print("No FLIR cameras detected.")
        return

    # Prioritize Boson camera if available, otherwise default to the first camera
    default_camera = available_cameras[0] if 'BOSON' in available_cameras else available_cameras[0]

    if not default_camera:
        print("No FLIR cameras detected.")
        return

    current_camera_index = default_camera
    cap = cv2.VideoCapture(current_camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Error: Could not open video device {current_camera_index}")
        return

    # Set frame properties based on the camera type
    if camera_type in CAMERA_RESOLUTIONS:
        frame_width, frame_height = CAMERA_RESOLUTIONS[camera_type]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    else:
        print(f"Warning: Unknown camera type '{camera_type}'. Using default resolution.")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize recording variables
    out_var = [None]
    recording_var = [False]

    # Create the Tkinter window
    global root
    root = tk.Tk()
    root.title("Thermal Camera Control")

    # Control frame for buttons and LUT selection
    control_frame = Frame(root)
    control_frame.pack(pady=5)

                # Camera selection dropdown
    camera_var = StringVar(root)
    camera_var.set(f"Camera {current_camera_index}")
    camera_menu = OptionMenu(control_frame, camera_var, *[f"Camera {device}" for device in available_cameras])
    camera_menu.pack(side="left", padx=5)

    # The fix for this issue is to use the "command" option to trigger the switch_camera function
    def switch_camera(*args):
        nonlocal cap, current_camera_index, out_var
        selected_camera = camera_var.get().split()[-1]
        if selected_camera != current_camera_index:
            with cap_lock:
                cap.release()
                current_camera_index = selected_camera
                cap = cv2.VideoCapture(current_camera_index, cv2.CAP_V4L2)
                if not cap.isOpened():
                    print(f"Error: Could not open video device {current_camera_index}")
                    return
                # Set frame properties again after switching
                if camera_type in CAMERA_RESOLUTIONS:
                    frame_width, frame_height = CAMERA_RESOLUTIONS[camera_type]
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
            update_video_frame()  # Restart the video capture thread

    camera_var.trace('w', switch_camera)

    # Record button
    record_button = Button(control_frame, text="Start/Stop Recording", command=lambda: toggle_recording(recording_var, out_var, record_button, frame_width, frame_height))
    record_button.pack(side="left", padx=5)

    # Screenshot button
    Button(control_frame, text="Take Screenshot", command=take_screenshot).pack(side="left", padx=5)

    # --- Flip Controls ---
    flip_options = ["None", "Horizontal", "Vertical", "Both"]
    flip_var = StringVar(root)
    flip_var.set("None")  # Default no flip

    Label(control_frame, text="Flip:").pack(side="left", padx=5)
    flip_menu = OptionMenu(control_frame, flip_var, *flip_options)
    flip_menu.pack(side="left", padx=5)

    # --- Rotation controls ---
    rotate_var = StringVar(root)
    rotate_var.set("0")  # Default no rotation
    rotate_options = ["0", "90", "180", "270"]
    Label(control_frame, text="Rotation:").pack(side="left", padx=5) 
    rotate_menu = OptionMenu(control_frame, rotate_var, *rotate_options)
    rotate_menu.pack(side="left", padx=5)

    # LUT selection dropdown
    lut_var = StringVar(root)
    lut_var.set(lut_name)
    if lut_name not in LUTS:
        print(f"Error: Unsupported LUT '{lut_name}'. Defaulting to 'WHITEHOT'.")
        lut_var.set('WHITEHOT')
    lut_menu = OptionMenu(control_frame, lut_var, *sorted(LUTS.keys()))
    lut_menu.pack(side="left", padx=5)

    # Exit button
    Button(control_frame, text="Exit Program", command=lambda: exit_program(cap, out_var)).pack(side="left", padx=5)


    # Video frame label
    video_label = Label(root)
    video_label.pack()

    # Start the video capture and processing in a separate thread
    def update_video_frame():
        global exit_event

        capture_thread = threading.Thread(target=capture_and_process_video, args=(
            cap, lut_var, recording_var, out_var, frame_width, frame_height, video_label, record_button, flip_var, exit_event, rotate_var))
        capture_thread.daemon = True
        capture_thread.start()

    update_video_frame()

    # Start the Tkinter main loop (MUST be after setting up UI elements)
    root.mainloop()

    # Release resources
    with cap_lock:
        cap.release()
    with out_lock:
        if out_var[0]:
            out_var[0].release()
    cv2.destroyAllWindows()

# --- Argparse Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a false-color LUT to a video stream from a thermal camera.")
    parser.add_argument('lut_name', type=str, nargs='?', default='WHITEHOT',
                        choices=sorted(LUTS.keys()),  # Access LUTS here
                        help="Name of the LUT to apply. Available options: " +
                             ", ".join(sorted(LUTS.keys())) + ". Default is 'WHITEHOT'.")
    parser.add_argument('--camera_type', type=str, default='BOSON',
                        choices=['BOSON', 'LEPTON3', 'LEPTON2', 'AMPBANK'],
                        help="Type of camera being used ('BOSON', 'LEPTON3', 'LEPTON2', 'AMPBANK'). Default is 'BOSON'.")

    args = parser.parse_args()
    main(args.lut_name, args.camera_type)

