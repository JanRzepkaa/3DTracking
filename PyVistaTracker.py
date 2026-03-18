from VistaRod import Rod
import pyvista as pv 
import numpy as np 
import time
import cv2 
from AnalyzePyVistaVideo import AnalyzePyVistaVideo
from VirtualEnviroment import VirtualEnviroment

class PyVistaTracker():
    def __init__(self, cameras_info, actors):
        self.cameras_info = cameras_info
        self.camera_count = len(cameras_info)
        self.actors = actors # {"group_name_1" : [], "group_name_2": []}

        self.initialize_plotters()
        self.initialize_virtual_env()

        self.video_analyzer = AnalyzePyVistaVideo(False)


    def initialize_plotters(self):
        self.plotters = []

        for i in range(self.camera_count):
            cam = self.cameras_info[i]
            new_plotter = pv.Plotter(off_screen=True, window_size=cam["resolution"])
            new_plotter.camera.position = cam["position"]
            new_plotter.camera.focal_point = cam["focal_point"]
            new_plotter.camera.up = (0, 0, 1)
            new_plotter.camera.clipping_range = (0.6, 1000)

            for actors_group in self.actors.values():
                for actor in actors_group:
                    new_plotter.add_actor(actor)

            my_light = pv.Light(
                position=cam["position"], 
                focal_point=(0, 0, 0), 
                color='white',
                light_type='scene light' # 'scene light' means it stays fixed in the 3D world
            )
            new_plotter.add_light(my_light)

            self.plotters.append(new_plotter)

    def start_plotters(self):
        for i in range(self.camera_count):
            self.plotters[i].show(interactive_update=True)
        self.virtual_env.show_plotter()

    def update_plotters(self):
        for i in range(self.camera_count):
            self.plotters[i].update()
        self.virtual_env.update_plotter()

    def close_plotter(self):
        for i in range(self.camera_count):
            self.plotters[i].close()
        self.virtual_env.close_plotter()
        
    def calculate_intrinsics(self, plotter):
        width, height = plotter.window_size

        cx = width / 2
        cy = height / 2

        fovy = plotter.camera.view_angle
        fovy_rad = np.radians(fovy)

        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy #* (width / height)

        camera_intrinsics = (fx, fy, cx, cy)
        return camera_intrinsics
    
    def initialize_virtual_env(self):
        self.virtual_env = VirtualEnviroment()

        all_cameras_intristic = []
        for plotter in self.plotters:
            intrinsics = self.calculate_intrinsics(plotter)
            all_cameras_intristic.append(intrinsics)
        
        self.virtual_env.initialize_calibration(self.camera_count, all_cameras_intristic)

    def fake_calibration(self):
        camera_positions = []
        for cam in self.cameras_info:
            camera_positions.append(cam["position"])
        self.virtual_env.fake_calibration(camera_positions)


    def get_raw_view_from_camera(self, camera_index):
        img_rgb = self.plotters[camera_index].screenshot(None, return_img=True)
        return img_rgb
    def get_bgr_view_from_camera(self, camera_index):
        img_rgb = self.get_raw_view_from_camera(camera_index)
        img_bgr = img_rgb[:, :, ::-1].copy()
        return img_bgr

    def get_position_of_actors_in_group(self, group_name):
        positions = []
        for actor in self.actors[group_name]:
            pos = actor.GetCenter()
            positions.append(pos)
        return positions

    def add_frame_for_calibration(self, camera_index, group_name):
        img_bgr = self.get_bgr_view_from_camera(camera_index)
        _, screen_positon = self.video_analyzer.find_centroids_and_contours(img_bgr)
        all_true_positions = self.get_position_of_actors_in_group(group_name)

        self.virtual_env.add_frame_for_calibration(all_true_positions, screen_positon, camera_index)

    def calibrate_virtual_cameras(self):
        for i in range(self.camera_count):
            self.virtual_env.calibrate_camera(i)

    def track_player(self):
        screen_positions = [None for i in range(self.camera_count)]
        for camera_index in range(self.camera_count):
            img_bgr = self.get_bgr_view_from_camera(camera_index)
            _, pos = self.video_analyzer.find_centroids_and_contours(frame=img_bgr.copy(), color="lime")
            screen_positions[camera_index] = pos if len(pos)>0 else None

        for i in range(self.camera_count):
            self.virtual_env.add_lines_to_all_points(i, screen_positions[i])

        self.virtual_env.match_rays()
