#!/usr/bin/env python3
# Copyright (C) 2026. All rights reserved.

import sys
import os
import cv2
import time
from datetime import datetime
import queue
import traceback
import importlib.util
import re
import logging
import gc
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QLabel, QPushButton,
                             QFileDialog, QDoubleSpinBox, QSpinBox, QCheckBox,
                             QMessageBox, QGroupBox, QTabWidget, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, QTimer, QProcess, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor


class VideoRecordThread(QThread):
    """
    A dedicated QThread for video encoding and disk I/O.
    Using QThread allows us to emit signals to the main GUI thread safely.
    """
    # Signal emitted when all frames are written and the file is closed safely
    recording_finished = pyqtSignal(str)

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        # Reduced maxsize to prevent massive RAM usage (60 frames ~= 2 secs of buffer)
        self.frame_queue = queue.Queue(maxsize=60)
        self.is_running = True
        self.writer = None

    def run(self):
        """Initializes the VideoWriter and continuously consumes frames from the queue."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        # Keep running until explicitly stopped AND the queue is completely emptied
        while self.is_running or not self.frame_queue.empty():
            try:
                # Wait for up to 0.1s for a new frame to arrive
                frame = self.frame_queue.get(timeout=0.1)

                # Ensure the writer is properly opened before writing
                if frame is not None and self.writer and self.writer.isOpened():
                    self.writer.write(frame)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VideoRecordThread] Error writing frame: {e}")

        # Gracefully release the resource to ensure the MP4 header is finalized
        if self.writer:
            self.writer.release()

        # Notify the main thread that the file is ready
        self.recording_finished.emit(self.output_path)

    def stop(self):
        """Signals the thread to stop processing new incoming frames."""
        self.is_running = False


# ---------------------------------------------------------
# Stream Redirector for Capturing Console Logs (Thread-safe via queue)
# ---------------------------------------------------------
class StreamRedirector:
    """Redirects stdout and stderr to a thread-safe Python queue."""
    def __init__(self, log_queue, is_error=False):
        self.log_queue = log_queue
        self.is_error = is_error

    def write(self, text):
        if text:
            # Put a tuple of (is_error, text) into the queue
            self.log_queue.put((self.is_error, text))

    def flush(self):
        pass


# ---------------------------------------------------------
# QThread for Running Inference
# ---------------------------------------------------------
class InferenceThread(QThread):
    """
    Runs the inference script and pushes data to queues in a background thread.
    Uses 'Monkey Patching' to hijack cv2.imshow and route frames to the GUI.
    """
    def __init__(self, script_path, cmd_args, log_queue, frame_queue):
        super().__init__()
        self.script_path = script_path
        self.cmd_args = cmd_args
        self.log_queue = log_queue
        self.frame_queue = frame_queue
        self._is_running = True

    def run(self):
        # 1. Redirect standard output and error to our queues
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamRedirector(self.log_queue, is_error=False)
        sys.stderr = StreamRedirector(self.log_queue, is_error=True)

        # Force logging module to output INFO logs to stdout instead of stderr
        # This prevents normal progress logs from being painted red as errors.
        # logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s', force=True)
        # %(levelname)s:%(name)s:%(message)s
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

        # 2. Mock sys.argv to simulate command line execution
        old_argv = sys.argv
        sys.argv = [self.script_path] + self.cmd_args

        # ----- PATH AND CACHE MANAGEMENT ----
        # Insert target script directory to sys.path so local imports work correctly
        script_dir = os.path.dirname(os.path.abspath(self.script_path))
        sys.path.insert(0, script_dir)

        # Define common local module names that conflict across different models
        conflict_modules = ["factory", "config"]
        for mod in conflict_modules:
            if mod in sys.modules:
                del sys.modules[mod]
        # ---------------------------------------

        # 3. Hijack (Monkey Patch) OpenCV functions
        old_imshow = cv2.imshow
        old_waitKey = cv2.waitKey
        old_namedWindow = getattr(cv2, 'namedWindow', None)
        old_resizeWindow = getattr(cv2, 'resizeWindow', None)
        old_destroyAllWindows = getattr(cv2, 'destroyAllWindows', None)
        old_getWindowProperty = getattr(cv2, 'getWindowProperty', None)

        def mock_imshow(winname, mat):
            if self._is_running:
                try:
                    # Implement Frame Dropping: discard the oldest frame if the queue is full
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    # Push a deep copy of the frame to isolate memory from the C++ thread
                    # self.frame_queue.put_nowait(mat.copy())
                    # Push frame directly (Zero-copy approach assumes inference script allocates new buffer per frame)
                    self.frame_queue.put_nowait(mat)
                except Exception:
                    pass

        def mock_waitKey(delay=0):
            if not self._is_running:
                return ord('q')
            if delay > 0:
                time.sleep(delay / 1000.0)
            return -1

        def mock_namedWindow(winname, flags=None):
            pass

        def mock_resizeWindow(winname, width, height):
            pass

        def mock_destroyAllWindows():
            pass

        def mock_getWindowProperty(winname, prop_id):
            return 1.0 if self._is_running else -1.0

        cv2.imshow = mock_imshow
        cv2.waitKey = mock_waitKey
        cv2.namedWindow = mock_namedWindow
        cv2.resizeWindow = mock_resizeWindow
        cv2.destroyAllWindows = mock_destroyAllWindows
        cv2.getWindowProperty = mock_getWindowProperty

        try:
            # 4. Dynamically load and execute the selected python script
            spec = importlib.util.spec_from_file_location("__main__", self.script_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["__main__"] = module
            spec.loader.exec_module(module)
        except SystemExit:
            pass # Catch sys.exit() so it doesn't kill our GUI
        except Exception as e:
            self.log_queue.put((True, traceback.format_exc()))
        finally:
            # 5. Restore original system state
            if sys.path and sys.path[0] == script_dir:
                sys.path.pop(0)

            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.argv = old_argv
            cv2.imshow = old_imshow
            cv2.waitKey = old_waitKey

            if old_namedWindow: cv2.namedWindow = old_namedWindow
            if old_resizeWindow: cv2.resizeWindow = old_resizeWindow
            if old_destroyAllWindows: cv2.destroyAllWindows = old_destroyAllWindows
            if old_getWindowProperty: cv2.getWindowProperty = old_getWindowProperty

            # === FIX FREEZE & COLLISION: FORCE GARBAGE COLLECTION ===
            if "__main__" in sys.modules:
                del sys.modules["__main__"]
            # Purge the local modules again upon exit
            for mod in conflict_modules:
                if mod in sys.modules:
                    del sys.modules[mod]
            gc.collect()

            # Send a sentinel value to notify the GUI that the thread has finished
            self.log_queue.put((False, "___THREAD_FINISHED___"))

    def stop(self):
        self._is_running = False


# ---------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------
class InferenceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.inference_thread = None

        # Flag to distinguish between natural finish and manual stop
        self._manual_stop = False

        # Generate a timestamped log filename for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"dx_studio_{timestamp}.log"

        # Initialize thread-safe communication queues
        self.log_queue = queue.Queue()
        # Restrict frame buffer to 3 frames to prevent memory explosion
        self.frame_queue = queue.Queue(maxsize=3)

        # Setup Regex for stripping ANSI codes before saving to file
        self.ansi_escape = re.compile(r'\x1b\[([0-9;]*)m')
        self.ansi_stripper = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]')

        # Start a QTimer to poll the queues at ~30Hz (33ms)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queues)
        self.poll_timer.start(33)

        # Initialize vatiable to store dxtop output
        self.dxtop_text = ""

        # Setup QProces to run dxtop asynchronously in the background
        self.dxtop_process = QProcess(self)

        # Connect rhe native readyReadStandardOutput signal to our update function
        self.dxtop_process.readyReadStandardOutput.connect(self.update_dxtop_info)

        # Start the continuous terminal process
        self.dxtop_process.start("dxtop")

        self.show_res_flag = False
        self.resize_timer = QTimer(self)
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.hide_resolution_osd)

        self._is_programmatic_resize = False
        self.video_index = -1

        self.retiring_recorders = []

        self.init_ui()

    def hide_resolution_osd(self):
        """Hides the resolution text after the user stops dragging the window."""
        # self.show_res_flag = False

    def resizeEvent(self, event):
        """Overrides the default resize event to trigger the resolution OSD."""
        super().resizeEvent(event)

        self.show_res_flag = True
        # Restart the timer to keep text visible while dragging.
        # It will hide 1.5 seconds after dragging stops.
        self.resize_timer.start(10000)

        if getattr(self, '_is_programmatic_resize', False):
            self._is_programmatic_resize = False
        else:
            # Reset the combo box to "Free Resize" if the user manually drags the window
            if hasattr(self, 'resolution_combo'):
                self.resolution_combo.blockSignals(True)
                self.resolution_combo.setCurrentIndex(0)
                self.resolution_combo.blockSignals(False)

    def update_dxtop_info(self):
        """
        Slot triggered by QProcess when new stdout data is available.
        Uses a 2D grid approach with a complete VT command set to perfectly emulate dxtop.
        """
        raw_data = self.dxtop_process.readAllStandardOutput().data()

        try:
            # Initialize a persistent buffer to handle chunked output from OS pipes
            if not hasattr(self, 'dxtop_buffer'):
                self.dxtop_buffer = ""

            self.dxtop_buffer += raw_data.decode('utf-8', errors='ignore')

            # 1. True Frame Delimiter (Home + Clear Screen sequences)
            if '\x1b[H\x1b[2J' in self.dxtop_buffer:
                frames = self.dxtop_buffer.split('\x1b[H\x1b[2J')
                self.dxtop_buffer = frames[-1]

            # Return early if buffer size suggests an incomplete frame
            if len(self.dxtop_buffer) < 50:
                return

            text = self.dxtop_buffer

            # 2. Precision fix for the "v2.5" split issue
            # Intercepts and swaps "v2^[[6;1H.5" into "v2.5^[[6;1H" before VT grid processing
            text = re.sub(
                r'(\d)\x1b\[(\d+;\d+)[Hf]\.5',
                lambda m: f"{m.group(1)}.5\x1b[{m.group(2)}H",
                text
            )

            # 3. Pre-processing: Clean up colors and unnecessary terminal commands
            text = re.sub(r'\x1b\].*?(?:\x07|\x1b\\)', '', text)  # Strip OSC Titles
            text = re.sub(r'\x1b\[[0-9;]*m', '', text)           # Strip ANSI Colors
            text = re.sub(r'\x1b\([a-zA-Z]', '', text)           # Strip Charsets (fixes ^[(B)
            text = re.sub(r'\x1b\[\?\d+[hl]', '', text)          # Strip Hide cursor & Modes
            text = re.sub(r'\x1b[=>]', '', text)                 # Strip Keypad modes

            # Convert degree symbol to a dot to avoid fallback destruction below
            text = text.replace('°', '.')

            # Fallback for OpenCV: Convert unsupported Unicode blocks to ASCII pipes
            text = re.sub(r'[^\x00-\x7F]', '|', text)

            # Convert 'X' commands to literal spaces (e.g., ^[[16X -> 16 spaces)
            text = re.sub(r'\x1b\[(\d+)X', lambda m: ' ' * int(m.group(1)), text)

            # Convert 'b' commands to repeat characters (restores UI horizontal dividers)
            text = re.sub(r'([^\x1b\n])\x1b\[(\d+)b', lambda m: m.group(1) * (int(m.group(2)) + 1), text)

            # 4. Virtual Terminal Layout Engine
            # Create a 40x120 2D array representing character cells
            grid = [[' ' for _ in range(120)] for _ in range(40)]
            cursor_y, cursor_x = 0, 0

            # Tokenize the stream into text chunks and cursor commands
            tokens = re.split(r'(\x1b\[[0-9;]*[a-zA-Z]|\r|\n)', text)

            for token in tokens:
                if not token:
                    continue

                if token == '\r':
                    cursor_x = 0
                elif token == '\n':
                    cursor_y += 1
                    cursor_x = 0
                elif token.startswith('\x1b['):
                    cmd_type = token[-1]
                    params = token[2:-1].split(';')

                    try:
                        # Emulate core Virtual Terminal (VT) positioning behaviors
                        if cmd_type in ('H', 'f'):
                            # Absolute positioning: ESC [ Y ; X H
                            cursor_y = max(0, int(params[0]) - 1) if params[0] else 0
                            cursor_x = max(0, int(params[1]) - 1) if len(params) > 1 and params[1] else 0
                        elif cmd_type == 'd':
                            # Absolute row assignment
                            cursor_y = max(0, int(params[0]) - 1) if params[0] else 0
                        elif cmd_type == 'G':
                            # Absolute column assignment
                            cursor_x = max(0, int(params[0]) - 1) if params[0] else 0
                        elif cmd_type == 'A':
                            # Move up
                            cursor_y = max(0, cursor_y - (int(params[0]) if params[0] else 1))
                        elif cmd_type == 'B':
                            # Move down
                            cursor_y += (int(params[0]) if params[0] else 1)
                        elif cmd_type == 'C':
                            # Move right
                            cursor_x += (int(params[0]) if params[0] else 1)
                        elif cmd_type == 'D':
                            # Move left
                            cursor_x = max(0, cursor_x - (int(params[0]) if params[0] else 1))
                        elif cmd_type == 'J':
                            # Clear display based on parameter
                            if params[0] == '2':
                                grid = [[' ' for _ in range(120)] for _ in range(40)]
                        elif cmd_type == 'K':
                            # Clear line from cursor rightwards
                            if cursor_y < 40:
                                for i in range(cursor_x, 120):
                                    grid[cursor_y][i] = ' '
                    except ValueError:
                        pass
                else:
                    # Write regular text tokens directly to the mapped grid
                    for char in token:
                        if cursor_y < 40 and cursor_x < 120:
                            grid[cursor_y][cursor_x] = char
                        cursor_x += 1

            # 5. Render the Grid to a string representation
            rendered_lines = []
            for row in grid:
                # Strip trailing spaces to keep alignment clean
                line = "".join(row).rstrip()
                # Omit completely blank lines
                if line:
                    rendered_lines.append(line)

            final_text = '\n'.join(rendered_lines)

            # Restrict excessively long horizontal lines for UI cleanliness
            final_text = re.sub(r'-{30,}', '-' * 50, final_text)

            if final_text:
                self.dxtop_text = final_text

        except Exception as e:
            print(f"Error parsing dxtop: {e}")

    def init_ui(self):
        self.setWindowTitle('DX Studio')
        self.resize(1024, 768)

        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.tab_display = QWidget()
        self.tab_settings = QWidget()
        self.tab_console = QWidget()

        self.tabs.addTab(self.tab_display, "Display")
        self.tabs.addTab(self.tab_settings, "Settings")
        self.tabs.addTab(self.tab_console, "Console")

        self.setup_display_tab()
        self.setup_settings_tab()
        self.setup_console_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def setup_display_tab(self):
        layout = QVBoxLayout()
        self.display_label = QLabel("Waiting for video stream...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #111; color: #888; font-size: 24px;")

        # Override the minimum size hint to prevent the window from being locked
        # when a large QPixmap is set. This allows the user to shrink the window.
        self.display_label.setMinimumSize(1, 1)

        layout.addWidget(self.display_label)
        self.tab_display.setLayout(layout)

    def setup_settings_tab(self):
        layout = QVBoxLayout()

        file_group = QGroupBox("File Paths")
        file_layout = QFormLayout()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.normpath(os.path.join(base_dir, "../../../../workspace/res/models"))
        video_dir = os.path.normpath(os.path.join(base_dir, "../../../../workspace/res/videos/sample_videos_v3.1.0"))

        def populate_combo(combo_box, target_dir, valid_extensions):
            combo_box.setEditable(True)
            if os.path.exists(target_dir):
                # Sort files alphabetically for a cleaner dropdown list
                for file_name in sorted(os.listdir(target_dir)):
                    if file_name.endswith(valid_extensions):
                        full_path = os.path.normpath(os.path.join(target_dir, file_name))
                        combo_box.addItem(full_path)

        # 1. Setup Python Script Input
        self.script_input = QComboBox()
        self.script_input.setEditable(True)
        # default_script = os.path.normpath(os.path.join(base_dir, "object_detection/yolov5s/yolov5s_async.py"))
        # default_script = os.path.normpath(os.path.join(base_dir, "object_detection/yolo26x/yolo26x_async.py"))
        default_script = os.path.normpath(os.path.join(base_dir, "classification/yolo26x_cls/yolo26x_cls_async.py"))
        self.script_input.addItem(default_script)

        self.script_btn = QPushButton("Browse")
        self.script_btn.setFixedWidth(100)  # Enforce a fixed width for uniform appearance
        self.script_btn.clicked.connect(self.browse_script)
        script_layout = QHBoxLayout()
        script_layout.addWidget(self.script_input)
        script_layout.addWidget(self.script_btn)
        file_layout.addRow("Python Script:", script_layout)

        # 2. Setup DXNN Model Input (Auto-scan)
        self.model_input = QComboBox()
        populate_combo(self.model_input, model_dir, ('.dxnn',))
        # Default to YOLOv5S if available
        for i in range(self.model_input.count()):
            # if "YoloV5S.dxnn" in self.model_input.itemText(i):
            # if "yolo26x.dxnn" in self.model_input.itemText(i):
            if "yolo26x-cls.dxnn" in self.model_input.itemText(i):
                self.model_input.setCurrentIndex(i)
                break

        self.model_btn = QPushButton("Browse")
        self.model_btn.setFixedWidth(100)
        self.model_btn.clicked.connect(self.browse_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_input)
        model_layout.addWidget(self.model_btn)
        file_layout.addRow("DXNN Model:", model_layout)

        # 3. Setup Video File Input (Auto-scan)
        self.video_input = QComboBox()
        populate_combo(self.video_input, video_dir, ('.mp4', '.mov', '.avi'))

        self.video_btn = QPushButton("Browse")
        self.video_btn.setFixedWidth(100)
        self.video_btn.clicked.connect(self.browse_video)
        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_input)
        video_layout.addWidget(self.video_btn)
        file_layout.addRow("Video File:", video_layout)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        param_group = QGroupBox("Inference Parameters")
        param_layout = QFormLayout()

        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setRange(0.0, 144.0)
        self.fps_spinbox.setSingleStep(1.0)
        self.fps_spinbox.setValue(30.0)
        self.fps_spinbox.setEnabled(False)

        # Create Auto FPS checkbox to match native video frame rate
        self.auto_fps_checkbox = QCheckBox("Auto FPS")
        self.auto_fps_checkbox.setChecked(True)
        self.auto_fps_checkbox.stateChanged.connect(self._toggle_fps_mode)

        # Group the spinbox and checkbox horizontally
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(self.fps_spinbox)
        fps_layout.addWidget(self.auto_fps_checkbox)

        # Add the combined layout to the form
        param_layout.addRow("Target FPS (0 for unlimited):", fps_layout)

        self.loop_spinbox = QSpinBox()
        self.loop_spinbox.setRange(-1, 10000)
        self.loop_spinbox.setValue(1)
        param_layout.addRow("Loop Count:", self.loop_spinbox)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        flag_group = QGroupBox("Execution Flags")
        flag_layout = QHBoxLayout()
        self.save_checkbox = QCheckBox("Save Output (--save)")
        self.show_log_checkbox = QCheckBox("Show Log (--show-log)")
        self.auto_loop_checkbox = QCheckBox("Auto Loop Playlist")
        self.auto_loop_checkbox.setChecked(1)
        self.show_dxtop_checkbox = QCheckBox("Show DXTOP (OSD)")
        self.show_dxtop_checkbox.setChecked(False)

        self.record_video_checkbox = QCheckBox("Record OSD Video (MP4)")
        self.record_video_checkbox.setChecked(False)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Free Resize", "640x480 (4:3)", "800x600 (4:3)",
            "1024x768 (4:3)", "1280x720 (16:9)", "1920x1080 (16:9)",
            "2160x1440 (3:2)", "3840x2160 (16:9)"
        ])
        self.resolution_combo.currentTextChanged.connect(self.change_window_resolution)

        # Create a ComboBox for OSD text size selection
        self.osd_size_combo = QComboBox()
        self.osd_size_combo.addItems(["Small", "Medium", "Large"])
        self.osd_size_combo.setCurrentIndex(1)  # Set default to 'Medium'
        self.osd_size_combo.setFixedWidth(90)   # Keep it compact

        flag_layout.addWidget(self.save_checkbox)
        flag_layout.addWidget(self.show_log_checkbox)
        flag_layout.addWidget(self.auto_loop_checkbox)
        flag_layout.addWidget(self.show_dxtop_checkbox)
        flag_layout.addWidget(self.record_video_checkbox)
        flag_layout.addWidget(self.osd_size_combo)
        flag_layout.addWidget(self.resolution_combo)
        flag_group.setLayout(flag_layout)
        layout.addWidget(flag_group)

        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Inference")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px;")
        self.run_btn.clicked.connect(self.execute_command)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; font-size: 14px;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_command)

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()
        self.tab_settings.setLayout(layout)

    def _toggle_fps_mode(self, state):
        """
        Disables the FPS spinbox when Auto mode is enabled.
        Prevents user confusion by locking manual input.
        """
        if state == Qt.Checked:
            self.fps_spinbox.setEnabled(False)
        else:
            self.fps_spinbox.setEnabled(True)

    def change_window_resolution(self, text):
        """Resizes the main window to ensure the display_label matches the target resolution EXACTLY."""
        if text == "Free Resize":
            return

        try:
            # 1. Parse the target dimensions from the combo box string (e.g., "1280x720 (16:9)")
            dim_part = text.split()[0]  # Extracts "1280x720"
            target_w_str, target_h_str = dim_part.split('x')
            target_w = int(target_w_str)
            target_h = int(target_h_str)

            # 2. Calculate the UI overhead (everything EXCEPT the display_label)
            # This accounts for margins, toolbars, splitters, and padding.
            ui_overhead_width = self.width() - self.display_label.width()
            ui_overhead_height = self.height() - self.display_label.height()

            # 3. Calculate the new total window size required
            new_window_width = target_w + ui_overhead_width
            new_window_height = target_h + ui_overhead_height

            self._is_programmatic_resize = True

            # 4. Apply the exact resize to the main GUI window
            self.resize(new_window_width, new_window_height)

        except (ValueError, IndexError):
            pass

    def setup_console_tab(self):
        layout = QVBoxLayout()
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("font-family: Consolas, monospace;")

        clear_btn = QPushButton("Clear Console")
        clear_btn.clicked.connect(self.console_output.clear)

        layout.addWidget(self.console_output)
        layout.addWidget(clear_btn)
        self.tab_console.setLayout(layout)

    # --- UI Interactions ---
    def _update_combo_from_browse(self, combo_box, filename):
        """Helper to add browsed file to combobox if it doesn't exist, and select it."""
        if filename:
            # Check if the file is already in the list
            index = combo_box.findText(filename)
            if index == -1:
                # Not found, add it and set index to the newly added item
                combo_box.addItem(filename)
                combo_box.setCurrentIndex(combo_box.count() - 1)
            else:
                # Found, just select it
                combo_box.setCurrentIndex(index)

    def browse_script(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Python Script", "", "Python Files (*.py);;All Files (*)")
        self._update_combo_from_browse(self.script_input, filename)

    def browse_model(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select DXNN Model", "", "DXNN Models (*.dxnn);;All Files (*)")
        self._update_combo_from_browse(self.model_input, filename)

    def browse_video(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        self._update_combo_from_browse(self.video_input, filename)

    def write_to_log_file(self, text):
        """Strip ANSI colors and append perfectly clean text to dx_studio.log synchronously"""
        clean_text = self.ansi_escape.sub('', text)
        try:
            with open(self.log_filename, "a", encoding="utf-8") as f:
                f.write(clean_text)
        except Exception:
            pass

    def on_inference_finished(self):
        """
        Slot called when the native QThread finishes.
        Guaranteed to execute even if the target script crashes unexpectedly.
        """
        if getattr(self, 'inference_thread', None) is not None:
            self.inference_thread.deleteLater()
            self.inference_thread = None

        # 1. Drain the queues one last time to catch any remaining logs or frames
        #    before the UI resets, ensuring no resources are left behind.
        self.poll_queues()

        # 2. Safely shut down the background video recording thread
        if getattr(self, 'video_recorder', None) is not None:
            # Transfer ownership to a local variable
            recorder = self.video_recorder

            # Immediately free up the main reference for the next auto-loop cycle
            self.video_recorder = None

            # Protect the thread from Python's Garbage Collection by keeping a reference
            self.retiring_recorders.append(recorder)

            # Disconnect custom signals to prevent delayed popups/actions in the UI
            # while the next video is already playing
            try:
                recorder.recording_finished.disconnect()
            except TypeError:
                pass

            # Define an asynchronous callback for when the thread actually exits
            def cleanup_thread():
                # Remove the strong reference so Python GC can collect it
                if recorder in self.retiring_recorders:
                    self.retiring_recorders.remove(recorder)

                # Safely instruct Qt to delete the underlying C++ object
                recorder.deleteLater()
                print("[SYSTEM] Background recording thread fully cleaned up.")

            # Connect QThread's native 'finished' signal to our cleanup callback
            recorder.finished.connect(cleanup_thread)

            # Finally, signal the thread's run() loop to exit gracefully
            recorder.stop()

        # 3. Handle Auto-Loop transition safely
        if not self._manual_stop:
            loop_val = self.loop_spinbox.value()

            # Record the starting video index on the very first natural finish
            if self.video_index == -1:
                self.video_index = self.video_input.currentIndex()

            current_idx = self.video_input.currentIndex()

            # Determine the index of the NEXT video to be played
            if getattr(self, 'auto_loop_checkbox', None) and self.auto_loop_checkbox.isChecked():
                next_idx = (current_idx + 1) % self.video_input.count()
            else:
                next_idx = current_idx  # Single video loop stays on the same index

            # A full cycle is complete if the next video is our starting video
            if next_idx == self.video_index:
                if loop_val > 0:
                    loop_val -= 1
                    self.loop_spinbox.setValue(loop_val)

            # If currently -1 (infinite loop) or still greater than 0 after decrement, continue playing
            if loop_val == -1 or loop_val > 0:

                # Determine whether to switch to the next video based on the checkbox
                if getattr(self, 'auto_loop_checkbox', None) and self.auto_loop_checkbox.isChecked():
                    msg = "[SYSTEM] Auto-looping to next video...\n"
                    # Apply the calculated next index
                    if hasattr(self, 'video_input') and self.video_input.count() > 0:
                        self.video_input.setCurrentIndex(next_idx)
                else:
                    # Single video loop
                    msg = "[SYSTEM] Looping current video...\n"

                self.write_to_log_file(msg)
                self.console_output.insertPlainText(msg)
                self.console_output.ensureCursorVisible()

                # Use QTimer to yield control back to the event loop before restarting
                QTimer.singleShot(100, self.execute_command)
                return  # Continue playback, return directly to avoid cleanup

        # Reset the tracker for the next manual run
        self.video_index = -1

        # Normal completion or manually stopped
        if hasattr(self, 'run_btn'):
            self.run_btn.setEnabled(True)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.setEnabled(False)

        msg = "[SYSTEM] Process finished and resources cleaned up.\n"
        self.write_to_log_file(msg)
        self.console_output.insertPlainText(msg)
        self.console_output.ensureCursorVisible()

    def on_recording_saved(self, saved_path: str):
            """
            Slot triggered entirely asynchronously when the VideoRecordThread completes its task.
            Safe to perform GUI operations here.
            """
            # QMessageBox.information(
            #     self,
            #     "Recording Saved",
            #     f"OSD Video successfully saved to:\n{saved_path}"
            # )

            print(f"OSD Video successfully saved to:\n{saved_path}")

            # Clean up the thread resource properly
            if getattr(self, 'video_recorder', None):
                self.video_recorder.deleteLater()
                self.video_recorder = None

            print("Inference process finished and resources cleaned up.")

    def execute_command(self):
        self._manual_stop = False

        if hasattr(self, 'record_video_checkbox') and self.record_video_checkbox.isChecked():
            model_name = self.model_input.currentText()
            model_name = os.path.basename(model_name)
            model_name, _ = os.path.splitext(model_name)

            video_name = self.video_input.currentText()
            video_name = os.path.basename(video_name)
            video_name, _ = os.path.splitext(video_name)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.output_video_path = f"dx_studio_record_{model_name}_{video_name}_{timestamp}.mp4"

        script = self.script_input.currentText().strip()
        model = self.model_input.currentText().strip()
        video = self.video_input.currentText().strip()

        if not script or not model:
            QMessageBox.warning(self, "Warning", "Python Script and DXNN Model are required fields.")
            return

        # Prepare arguments
        cmd_args = ["--model", model]
        if video: cmd_args.extend(["--video", video])
        if self.auto_fps_checkbox.isChecked():
            cmd_args.extend(["--fps", "-1"])
        else:
            cmd_args.extend(["--fps", str(self.fps_spinbox.value())])

        # loop_val = self.loop_spinbox.value()
        # if loop_val > 1: cmd_args.extend(["--loop_val", str(loop_val)])
        self.loop_val = self.loop_spinbox.value()
        if self.save_checkbox.isChecked(): cmd_args.append("--save")
        if self.show_log_checkbox.isChecked(): cmd_args.append("--show-log")

        # Automatically switch to the Display tab so the user sees the video
        self.tabs.setCurrentWidget(self.tab_display)

        msg = f"[SYSTEM] Starting thread...\nArgs: {' '.join(cmd_args)}\n"
        self.write_to_log_file(msg)
        self.console_output.insertPlainText(msg)
        self.console_output.ensureCursorVisible()

        # Clear any residual data in the queues from previous runs
        while not self.log_queue.empty(): self.log_queue.get_nowait()
        while not self.frame_queue.empty(): self.frame_queue.get_nowait()

        # Create and start the inference thread
        self.inference_thread = InferenceThread(script, cmd_args, self.log_queue, self.frame_queue)

        # Connect the native finished signal to our cleanup slot
        self.inference_thread.finished.connect(self.on_inference_finished)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.inference_thread.start()

    def stop_command(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self._manual_stop = True
            msg = "[SYSTEM] Stopping process gracefully...\n"
            self.write_to_log_file(msg)
            self.console_output.insertPlainText(msg)
            self.console_output.ensureCursorVisible()
            self.inference_thread.stop()

    def insert_ansi_text(self, text, default_color):
        """Translate ANSI control codes to native QTextEdit UI styling"""
        parts = self.ansi_escape.split(text)

        for i, part in enumerate(parts):
            if i % 2 == 1:
                codes = part.split(';')
                for code in codes:
                    if code in ('0', ''):
                        self.console_output.setTextColor(default_color)
                        font = self.console_output.currentFont()
                        font.setBold(False)
                        self.console_output.setCurrentFont(font)
                    elif code == '1':
                        font = self.console_output.currentFont()
                        font.setBold(True)
                        self.console_output.setCurrentFont(font)
                    elif code in ('31', '91'): self.console_output.setTextColor(QColor("#EF4444"))
                    elif code in ('32', '92'): self.console_output.setTextColor(QColor("#10B981"))
                    elif code in ('33', '93'): self.console_output.setTextColor(QColor("#F59E0B"))
                    elif code in ('34', '94'): self.console_output.setTextColor(QColor("#3B82F6"))
                    elif code in ('35', '95'): self.console_output.setTextColor(QColor("#8B5CF6"))
                    elif code in ('36', '96'): self.console_output.setTextColor(QColor("#06B6D4"))
            else:
                if part:
                    self.console_output.insertPlainText(part)

    def poll_queues(self):
        """Periodically checks the queues and safely updates the UI in the main thread."""

        # Fetch the current OS default text color from the active application palette
        default_text_color = QApplication.palette().text().color()

        # Process Logs (drain the queue)
        while not self.log_queue.empty():
            try:
                is_error, text = self.log_queue.get_nowait()
                self.write_to_log_file(text)
                if is_error:
                    self.insert_ansi_text(text, QColor("red"))
                else:
                    self.insert_ansi_text(text, default_text_color)
                self.console_output.setTextColor(default_text_color)
                self.console_output.ensureCursorVisible()
            except queue.Empty:
                break

        # Process Frames (Frame Dropping: keep only the newest frame)
        latest_frame = None
        while not self.frame_queue.empty():
            try:
                latest_frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        # Push the latest frame to the custom widget for painting
        if latest_frame is not None:
            self.render_frame(latest_frame)

    def render_frame(self, frame):
        """Converts OpenCV BGR image to QPixmap and scales it to the label."""

        # Calculate dynamic resolution ratio based on 1080p
        # This ensures OSD elements scale proportionately across different input video sizes
        frame_h, frame_w = frame.shape[:2]
        ratio = frame_h / 1080.0

        # =========================================================
        # 1. Draw Model Info & Video Source (Top-Left)
        # =========================================================
        # try:
        #     model_name = self.model_input.itemText(self.model_input.currentIndex())
        #     model_name = os.path.basename(model_name)
        #     video_name = self.video_input.itemText(self.video_input.currentIndex())
        #     video_name = os.path.basename(video_name)

        #     info_texts = [f"Source: {video_name}", f"Model: {model_name}"]

        #     # Dynamic text scaling for top-left OSD
        #     info_font_scale = 1.0 * ratio
        #     info_thick = max(1, int(2 * ratio))

        #     # Find the maximum width among the info text lines for bounding box calculation
        #     max_text_w = 0
        #     for text in info_texts:
        #         (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, info_font_scale, info_thick)
        #         if w > max_text_w:
        #             max_text_w = w

        #     # Position at top-left with safe padding boundaries
        #     margin = max(10, int(20 * ratio))
        #     padding = max(5, int(10 * ratio))
        #     line_spacing = max(20, int(35 * ratio))

        #     box_x1 = margin
        #     box_y1 = margin
        #     box_x2 = box_x1 + max_text_w + padding * 2
        #     box_h = padding * 2 + (len(info_texts) - 1) * line_spacing + h
        #     box_y2 = box_y1 + box_h

        #     # Draw semi-transparent background overlay
        #     overlay = frame.copy()
        #     cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        #     cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        #     # Draw standard white text line by line
        #     text_x = box_x1 + padding
        #     text_y = box_y1 + padding + h
        #     for text in info_texts:
        #         cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        #                     info_font_scale, (255, 255, 255), info_thick, cv2.LINE_AA)
        #         text_y += line_spacing

        # except Exception:
        #     pass

        # =========================================================
        # 2. Draw DXTOP Monitor (Top-Right)
        # =========================================================
        if hasattr(self, 'show_dxtop_checkbox') and self.show_dxtop_checkbox.isChecked() and self.dxtop_text:

            # Default to Medium size if combo box is not yet initialized
            size_mode = "Medium"
            if hasattr(self, 'osd_size_combo'):
                size_mode = self.osd_size_combo.currentText()

            # Assign rendering parameters based on the selected size
            if size_mode == "Small":
                # Small
                base_start_y = 25
                base_line_height = 20
                base_font_scale = 0.5
                base_outline_thick = 2
                base_inner_thick = 1
            elif size_mode == "Medium":
                # Medium
                base_start_y = 28
                base_line_height = 25
                base_font_scale = 0.65
                base_outline_thick = 2
                base_inner_thick = 1
            else:
                # Large
                base_start_y = 30
                base_line_height = 30
                base_font_scale = 0.8
                base_outline_thick = 4
                base_inner_thick = 2

            # Apply dynamic ratio to base values
            start_y = max(10, int(base_start_y * ratio))
            line_height = int(base_line_height * ratio)
            font_scale = base_font_scale * ratio
            outline_thick = max(1, int(base_outline_thick * ratio))
            inner_thick = max(1, int(base_inner_thick * ratio))

            lines = self.dxtop_text.split('\n')

            # Calculate the maximum width of the DXTOP block to align it to the right edge
            max_dxtop_w = 0
            for line in lines:
                (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, inner_thick)
                if w > max_dxtop_w:
                    max_dxtop_w = w

            # Position at top-right by subtracting width from total frame width
            dxtop_x = frame_w - max_dxtop_w - max(10, int(20 * ratio))

            for i, line in enumerate(lines):
                y_pos = start_y + (i * line_height)

                # Draw text outline for visibility against bright backgrounds
                cv2.putText(frame, line, (dxtop_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), outline_thick, cv2.LINE_AA)

                # Draw the actual inner green text
                cv2.putText(frame, line, (dxtop_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 0), inner_thick, cv2.LINE_AA)

        # =========================================================
        # 3. Draw Dynamic Resolution OSD (Bottom-Left)
        # =========================================================
        if getattr(self, 'show_res_flag', False):
            current_w = self.display_label.width()
            current_h = self.display_label.height()
            res_text = f"Resolution: {current_w} x {current_h}"

            # Scale properties for resolution prompt
            res_font_scale = 1.0 * ratio
            res_inner_thick = max(1, int(2 * ratio))

            (text_w, text_h), baseline = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, res_font_scale, res_inner_thick)

            margin = max(10, int(20 * ratio))
            padding = max(5, int(10 * ratio))

            box_x1 = margin
            box_y2 = frame_h - margin
            box_x2 = box_x1 + text_w + padding * 2
            box_y1 = box_y2 - (text_h + baseline + padding * 2)

            # Draw semi-transparent black background behind the yellow text
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            text_x = box_x1 + padding
            text_y = box_y2 - padding - baseline

            # Render vibrant yellow text
            cv2.putText(frame, res_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        res_font_scale, (0, 255, 255), res_inner_thick, cv2.LINE_AA)

        # =========================================================
        # 4. Record the final composited frame (Asynchronous Queueing)
        # =========================================================
        if hasattr(self, 'record_video_checkbox') and self.record_video_checkbox.isChecked():
            # Initialize the background recording thread on the first frame
            if getattr(self, 'video_recorder', None) is None:

                target_fps = 30.0  # Safe fallback

                # Extract the exact native FPS directly from the selected video source
                if hasattr(self, 'auto_fps_checkbox') and self.auto_fps_checkbox.isChecked():
                    video_path = self.video_input.currentText().strip()
                    if video_path:
                        # Handle both camera index (int) and video file path (string)
                        source = int(video_path) if video_path.isdigit() else video_path

                        # Briefly open the video source to parse its header properties
                        cap = cv2.VideoCapture(source)
                        if cap.isOpened():
                            native_fps = cap.get(cv2.CAP_PROP_FPS)
                            # Verify if the returned FPS is a valid, positive number
                            if native_fps > 0:
                                target_fps = native_fps
                        cap.release()
                else:
                    # Use manually assigned FPS from the spinbox
                    if hasattr(self, 'fps_spinbox') and self.fps_spinbox.value() > 0:
                        target_fps = float(self.fps_spinbox.value())

                print(f"target_fps: {target_fps}")

                output_path = getattr(self, 'output_video_path', 'output.mp4')

                # Spawn and start the worker thread
                self.video_recorder = VideoRecordThread(output_path, target_fps, frame_w, frame_h)

                # Connect the thread's finished signal to our UI callback
                self.video_recorder.recording_finished.connect(self.on_recording_saved)
                self.video_recorder.start()

            # Push the frame into the queue.
            # We use .copy() to prevent the main thread from mutating the image while the writer is saving it.
            if self.video_recorder.isRunning():
                try:
                    self.video_recorder.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    # Drop frame automatically if the disk is too slow, protecting UI from OOM crash
                    print("[Warning] Video encoding queue is full. Dropping frame.")

        # Convert BGR (OpenCV format) to RGB
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w

        # Create QImage from numpy array
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap and scale down to fit the window if necessary
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = InferenceGUI()
    gui.show()
    sys.exit(app.exec_())
