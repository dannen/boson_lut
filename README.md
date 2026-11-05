# Boson LUT Script

## Overview
This script is used to apply false-color Look-Up Tables (LUTs) to a video stream from a FLIR thermal camera (such as FLIR Boson or FLIR Lepton). The script also provides a GUI using Tkinter for user interaction, where users can select different LUTs and control video recording.

The primary features of the script include:
- Detection of available FLIR cameras.
- Application of various predefined or custom LUTs to thermal video streams.
- Video recording capability.
- A user-friendly GUI for camera and LUT selection.

## Required Python Modules
The following Python modules are required to run this script:
- `opencv-python`
- `numpy`
- `Pillow`
- `pyudev`
- `tkinter`
- `argparse`

To install the required modules, run the following command:
```sh
pip install opencv-python numpy Pillow pyudev
```

Note: `tkinter` is usually included with the standard Python installation.

## Usage
The script can be run from the command line and provides various command line options to customize the behavior. Below are the available options.

### Command Line Options
- `LUT` (optional positional argument, or `--lut LUT`): The LUT to start with. Use one or the other, not both.
  - Built-in options: `ARCTIC`, `BLACKHOT`, `BONE`, `CIVIDIS`, `COOL`, `GLOBOW`, `GRADEDFIRE`, `HOT`, `HSV`, `INFERNO`, `INSTALERT`, `ISOTHERM_BLUE`, `ISOTHERM_GREEN`, `ISOTHERM_RED`, `LAVA`, `MAGMA`, `OCEAN`, `PINK`, `PLASMA`, `RAINBOW`, `SPRING`, `SUMMER`, `VIRIDIS`, `WHITEHOT`.
  - Any `.lut` file in the current directory is also available, under its file name without the extension. `--help` lists everything it found.
  - Default: the LUT saved with **Save Config**, otherwise `WHITEHOT`. A LUT named on the command line wins over the saved one.
- `--camera_type` (optional argument): The type of camera being used: `BOSON`, `LEPTON3`, `LEPTON2` or `AMPBANK`.
  - Default: detected from the camera's USB IDs. Those can't tell a Lepton 2 from a Lepton 3, so a Lepton is treated as a `LEPTON3` unless you pass `--camera_type LEPTON2`.
  - If several cameras are attached, the one of this type is opened first.

### Example Usage
```sh
python boson_lut.py ironblack
```
This command starts with the `ironblack` LUT and detects the camera type.

```sh
python boson_lut.py --lut RAINBOW --camera_type LEPTON2
```
This command starts with the `RAINBOW` LUT and treats the Lepton as a Lepton 2.

## Features
- **Thermal Camera Detection**: Automatically detects connected FLIR Boson, FLIR Lepton and Ampbank cameras using `pyudev`. Other video devices, such as webcams, are not listed.
- **GUI Controls**: A GUI interface using Tkinter allows users to:
  - Select different LUTs.
  - Switch between detected cameras.
  - Flip and rotate the picture.
  - Start or stop video recording, and take screenshots. Both are saved in the current directory.
  - Calibrate a Boson on demand.
  - Save the current LUT, camera, flip and rotation to `thermal_config.json` in the current directory.
  - Exit the application.
- **Recording**: A recording keeps the size the picture had when it started. If the rotation or camera changes during a recording, the picture is scaled to fit that size, with black bars.
- **Boson Calibration**: By default a Boson recalibrates itself periodically (flat field correction, or FFC), which freezes the picture for about 0.4 seconds and shifts its brightness, including in recordings. While this program runs, it turns that off, calibrates once at startup, and leaves further calibrations to the **Calibrate** button. A message appears when the camera recommends calibrating, which it does once its temperature or time threshold is reached. The picture can degrade if you don't. On exit, the camera goes back to the mode it was in, and the setting is never saved to the camera's flash. Calibrating during a recording shows up in the video.
- **Custom LUTs**: Besides the isotherm LUTs (`ISOTHERM_RED`, `ISOTHERM_GREEN` and `ISOTHERM_BLUE`), the script loads `.lut` files from the current directory. Each file holds a list of 256 `(B, G, R)` tuples.

## Set USB Permissions
We faced an issue with the application not having permission to access the FLIR Boson USB device. To resolve this, we updated udev rules:

1. **Create a Udev Rule for the FLIR Device**:
   ```bash
   sudo nano /etc/udev/rules.d/99-flir-boson.rules
   ```

2. **Add the Following Line** (to grant access to the USB device):
   ```bash
   SUBSYSTEM=="usb", ATTRS{idVendor}=="09cb", ATTRS{idProduct}=="4007", MODE="0666"
   ```

   **Example `99-flir-boson.rules` file**:
   ```
   # Udev rule to grant permissions for FLIR Boson USB device
   SUBSYSTEM=="usb", ATTRS{idVendor}=="09cb", ATTRS{idProduct}=="4007", MODE="0666"
   ```

3. **Reload Udev Rules**:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

   **Alternative Command** (using device paths):
   ```bash
   sudo udevadm trigger /dev/bus/usb/*
   ```

## Serial Port Permissions
Calibration control uses the Boson's USB serial port (usually `/dev/ttyACM0`), which normally requires membership of the `dialout` group:
```bash
sudo usermod -aG dialout $USER
```
Log in again afterwards. Without access, the program still runs, and the camera keeps calibrating periodically.

## Notes
- Ensure that your system has access to the appropriate video devices, and that you have permissions to access them.
- The script uses OpenCV for video capture and processing, and Tkinter for GUI management.
- The FLIR cameras must be connected to the system before running the script.

## License
This project is licensed under the MIT License.

