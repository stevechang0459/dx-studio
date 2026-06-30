#!/usr/bin/env python3
# Copyright (C) 2026. All rights reserved.

import sys
import os
import cv2
import time
import queue
import traceback
import importlib.util
import logging
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QFormLayout, QLabel, QPushButton,
                             QFileDialog, QDoubleSpinBox, QSpinBox, QCheckBox,
                             QMessageBox, QGroupBox, QTabWidget, QTextEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor

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
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)

        # 2. Mock sys.argv to simulate command line execution
        old_argv = sys.argv
        sys.argv = [self.script_path] + self.cmd_args

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
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.argv = old_argv
            cv2.imshow = old_imshow
            cv2.waitKey = old_waitKey

            if old_namedWindow: cv2.namedWindow = old_namedWindow
            if old_resizeWindow: cv2.resizeWindow = old_resizeWindow
            if old_destroyAllWindows: cv2.destroyAllWindows = old_destroyAllWindows
            if old_getWindowProperty: cv2.getWindowProperty = old_getWindowProperty

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

        # Initialize thread-safe communication queues
        self.log_queue = queue.Queue()
        # Restrict frame buffer to 3 frames to prevent memory explosion
        self.frame_queue = queue.Queue(maxsize=3)

        # Start a QTimer to poll the queues at ~30Hz (33ms)
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self.poll_queues)
        self.poll_timer.start(33)

        self.init_ui()

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
        default_script = os.path.normpath(os.path.join(base_dir, "object_detection/yolov5s/yolov5s_async.py"))
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
            if "YoloV5S.dxnn" in self.model_input.itemText(i):
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
        param_layout.addRow("Target FPS (0 for unlimited):", self.fps_spinbox)

        self.loop_spinbox = QSpinBox()
        self.loop_spinbox.setRange(1, 10000)
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

        flag_layout.addWidget(self.save_checkbox)
        flag_layout.addWidget(self.show_log_checkbox)
        flag_layout.addWidget(self.auto_loop_checkbox)
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

    # --- Execution Logic ---
    def execute_command(self):
        self._manual_stop = False

        script = self.script_input.currentText().strip()
        model = self.model_input.currentText().strip()
        video = self.video_input.currentText().strip()

        if not script or not model:
            QMessageBox.warning(self, "Warning", "Python Script and DXNN Model are required fields.")
            return

        # Prepare arguments (excluding the script name itself, as it's passed separately)
        cmd_args = ["--model", model]
        if video: cmd_args.extend(["--video", video])
        cmd_args.extend(["--fps", str(self.fps_spinbox.value())])

        loop_val = self.loop_spinbox.value()
        if loop_val > 1: cmd_args.extend(["--loop", str(loop_val)])
        if self.save_checkbox.isChecked(): cmd_args.append("--save")
        if self.show_log_checkbox.isChecked(): cmd_args.append("--show-log")

        # Automatically switch to the Display tab so the user sees the video
        self.tabs.setCurrentWidget(self.tab_display)

        self.console_output.insertPlainText(f"[SYSTEM] Starting thread...\nArgs: {' '.join(cmd_args)}\n")
        self.console_output.ensureCursorVisible()

        # Clear any residual data in the queues from previous runs
        while not self.log_queue.empty(): self.log_queue.get_nowait()
        while not self.frame_queue.empty(): self.frame_queue.get_nowait()

        # Create and start the inference thread
        self.inference_thread = InferenceThread(script, cmd_args, self.log_queue, self.frame_queue)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.inference_thread.start()

    def stop_command(self):
        if self.inference_thread and self.inference_thread.isRunning():
            self._manual_stop = True
            self.console_output.insertPlainText("[SYSTEM] Stopping process gracefully...\n")
            self.console_output.ensureCursorVisible()
            self.inference_thread.stop()

    def poll_queues(self):
        """Periodically checks the queues and safely updates the UI in the main thread."""

        # Fetch the current OS default text color from the active application palette
        default_text_color = QApplication.palette().text().color()

        # 1. Process Logs (drain the queue)
        while not self.log_queue.empty():
            try:
                is_error, text = self.log_queue.get_nowait()
                if text == "___THREAD_FINISHED___":
                    if self.inference_thread is not None:
                        # Wait up to 1 second for thread to terminate naturally
                        if not self.inference_thread.wait(1000):
                            self.console_output.setTextColor(QColor("red"))
                            self.console_output.insertPlainText("[WARNING] Thread hung! Force killing...\n")
                            self.console_output.setTextColor(default_text_color)

                            self.inference_thread.terminate()
                            self.inference_thread.wait()
                        self.inference_thread = None

                    # --- Auto-Loop Logic ---
                    if self.auto_loop_checkbox.isChecked() and not self._manual_stop:
                        self.console_output.insertPlainText("[SYSTEM] Auto-looping to next video...\n")
                        self.console_output.ensureCursorVisible()

                        # Increment the video combo box index, wrapping around
                        if self.video_input.count() > 0:
                            current_idx = self.video_input.currentIndex()
                            next_idx = (current_idx + 1) % self.video_input.count()
                            self.video_input.setCurrentIndex(next_idx)

                        self.execute_command()
                    else:
                        # Normal finish or manually stopped
                        self.run_btn.setEnabled(True)
                        self.stop_btn.setEnabled(False)
                        self.console_output.insertPlainText("[SYSTEM] Process finished.\n")
                        self.console_output.ensureCursorVisible()

                elif is_error:
                    # Switch pen to red for error tracebacks
                    self.console_output.setTextColor(QColor("red"))
                    self.console_output.insertPlainText(text)
                    # Revert pen to default theme color
                    self.console_output.setTextColor(default_text_color)
                    self.console_output.ensureCursorVisible()
                else:
                    self.console_output.insertPlainText(text)
                    self.console_output.ensureCursorVisible()
            except queue.Empty:
                break

        # 2. Process Frames (Frame Dropping: keep only the newest frame)
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
