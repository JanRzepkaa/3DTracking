import pyvista as pv
import numpy as np 
from VirtualEnvHelpers import *

class VirtualPoint():
    def __init__(self, position = (0, 0, 0)):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = [0, 0, 0]

        self.vista = pv.Sphere(radius=0.3, center=self.position)

    def move(self, new_position):
        new_position = np.array(new_position, dtype=np.float64)
        self.velocity = new_position-self.position
        self.position = new_position

        self.vista.points += self.velocity


class VirtualCamera():
    def __init__(self, position=(0, 0, 0), R_matrix=None, intrinsics=None):
        self.position = np.array(position, dtype=np.float64)
        if type(R_matrix) == type(None):
            R_matrix = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.R_matrix = np.array(R_matrix, dtype=np.float64)
        self.intrinsics = intrinsics

        self.vista = create_camera_frustum(1)

        self.current_rays = []

    def add_pyvista_actor(self, actor):
        self.vista_actor = actor

    def move_camera(self, new_pos, new_R_matrix):
        self.position = np.array(new_pos, dtype=np.float64)
        self.R_matrix = np.array(new_R_matrix, dtype=np.float64)
        self.move_frustum()

    def move_frustum(self):
        """
        Teleports and rotates the wireframe frustum to match the solved camera pose.
        """
        if self.vista_actor is None:
            # Actor not yet defined
            return
        transform_matrix = np.eye(4, dtype=np.float64)
        transform_matrix[:3, :3] = self.R_matrix
        transform_matrix[:3, 3] = self.position

        print(transform_matrix)
        
        self.vista_actor.user_matrix = transform_matrix

    def ray_to_point(self, point):
        fx, fy, cx, cy = self.intrinsics

        pixel_x, pixel_y = point

        x = (pixel_x - cx) / fx
        y = (pixel_y - cy) / fy

        Z = 1
        X = x*Z
        Y = y*Z
        camera_to_point_rotated = np.array([X, Y, Z])

        ray_direction_world = self.R_matrix @ camera_to_point_rotated

        ray_direction_world = ray_direction_world / np.linalg.norm(ray_direction_world)

        return ray_direction_world
    
    def rays_to_all_points(self, points):
        new_rays = []
        for point in points:
            ray = self.ray_to_point(point)
            new_rays.append(ray)

        self.current_rays = new_rays
        return new_rays
        