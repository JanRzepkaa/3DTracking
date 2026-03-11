import pyvista as pv
import numpy as np 

class VirtualEnviroment:
    def __init__(self):
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 400), title="Real-Time Control")

    def initialize_calibration(self, camera_count, cameras_intrinsics):
        self.calibration_info = [[] for i in range(camera_count)]
        # Single entry = (true_position, screen_position)
        self.cameras_intrinsics = cameras_intrinsics # (camera_count, 3, 3)

    def add_frame_for_calibration(self, true_pos, screen_pos, camera_index):
        self.calibration_info[camera_index].append((true_pos, screen_pos))

    def calibrate_single_frame(self, true_pos, screen_pos, camera_index):
        """
            Calibrating based on info from a single frame
        """

    def calibrate_camera(self, camera_index):
        """
            Find the best position based on info from many frames
        """