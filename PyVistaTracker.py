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
        self.actors = actors

        self.plotters = []
        self.initialize_plotters()


    def initialize_plotters(self):
        for i in range(self.camera_count):
            cam = self.cameras_info[i]
            new_plotter = pv.Plotter(off_screen=True, window_size=cam.resolution)
            new_plotter.camera.position = cam.position
            new_plotter.camera.focal_point = cam.focal_point
            new_plotter.camera.up = (0, 0, 1)
            new_plotter.camera.clipping_range = (0.6, 1000)

            for actor in self.actors:
                new_plotter.add_actor(actor)

            my_light = pv.Light(
                position=cam.position, 
                focal_point=(0, 0, 0), 
                color='white',
                light_type='scene light' # 'scene light' means it stays fixed in the 3D world
            )
            new_plotter.add_light(my_light)

            self.plotters.append(new_plotter)
