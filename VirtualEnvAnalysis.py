import pyvista as pv
import numpy as np 
from VirtualEnvHelpers import *
import scipy

class VirtualPoint():
    def __init__(self, position = (0, 0, 0)):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = [0, 0, 0]

        self.vista = pv.Sphere(radius=0.3, center=self.position)

    def move(self, new_position):
        self.show()

        new_position = np.array(new_position, dtype=np.float64)
        self.velocity = new_position-self.position
        self.position = new_position

        self.vista.points += self.velocity

    def hide(self):
        self.actor.SetVisibility(False)
    
    def show(self):
        self.actor.SetVisibility(True)

    def add_to_plotter(self, plotter):
        self.actor = plotter.add_mesh(self.vista, color="lime")



class VirtualCamera():
    def __init__(self, position=(0, 0, 0), R_matrix=None, intrinsics=None):
        self.position = np.array(position, dtype=np.float64)
        if type(R_matrix) == type(None):
            R_matrix = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.R_matrix = np.array(R_matrix, dtype=np.float64)
        self.intrinsics = intrinsics

        self.vista = create_camera_frustum(1)

        self.current_rays = []

        self.dummy_tubes = []
        self.used_dummy_tubes = []
        self.none_line = pv.Line((0, 0, 0), (0, 0, 0.1)).tube(radius=0.001)
        for i in range(10):
            self.dummy_tubes.append(pv.Line((0, 0, 0), (0, 0, 0.1)).tube(radius=0.001))
            self.used_dummy_tubes.append(False)

    def __str__(self):
        return f"Position: {self.position} All rays: {self.current_rays}"

    def add_to_plotter(self, plotter):
        actor = plotter.add_mesh(
            self.vista, 
            style='wireframe',  # Makes it transparent with lines
            color='cyan', 
            line_width=2
        )
        self.vista_actor = actor
        self.ray_actors = []
        for i in self.dummy_tubes:
            self.ray_actors.append(plotter.add_mesh(i, color="yellow"))

    def hide_rays(self):
        for i in self.ray_actors:
            i.SetVisibility(False)

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
        
    def add_ray(self, ray, index=0):
        start = self.position
        end = start + np.linalg.norm(start)*1.2*ray

        line = pv.Line(start, end)
        self.ray_actors[index].SetVisibility(True)
        self.dummy_tubes[index].points = line.tube(radius=0.05).points

    def add_all_rays_from_points(self, all_points):
        all_rays = self.rays_to_all_points(all_points)
        
        self.hide_rays()
        
        for i, ray in enumerate(all_rays):
            self.add_ray(ray, i)
        return all_rays
    
    
class GlobalRayManager():
    def __init__(self, cameras : list[VirtualCamera]):
        self.cameras = cameras

        self.points = [VirtualPoint() for i in range(10)]

    def hide_points(self):
        for i in self.points:
            i.hide()

    def add_to_plotter(self, plotter):
        for i in self.points:
            i.add_to_plotter(plotter)
            i.hide()
    
    def match_rays_from_2_cameras(self, index_A=0, index_B=1):
        distance_threshold = 0.3

        rays_A, rays_B = self.cameras[index_A].current_rays, self.cameras[index_B].current_rays
        pos_A, pos_B = self.cameras[index_A].position, self.cameras[index_B].position
        num_A, num_B = len(rays_A), len(rays_B)
        cost_matrix = np.full((num_A, num_B), np.inf)

        # 1. Build the Cost Matrix
        # Initialize with infinity so missing balls don't get accidentally matched
        for i in range(num_A):
            i_ray = rays_A[i]
            for j in range(num_B):
                j_ray = rays_B[j]
                dist = calculate_ray_to_ray_distance(pos_A, i_ray, pos_B, j_ray)
                cost_matrix[i, j] = dist

        # 2. Run the Hungarian Algorithm
        # row_ind will be the indices for Camera A, col_ind for Camera B
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
        
        # 3. Filter out bad matches (Ghosts or missing balls)
        matched_pairs = []
        for i, j in zip(row_ind, col_ind):
            match_distance = cost_matrix[i, j]
            
            # Only accept the match if the rays actually cross close to each other
            if match_distance < distance_threshold:
                matched_pairs.append({
                    'cam_A_index': i,
                    'cam_B_index': j,
                    'error': match_distance
                })
        
        return matched_pairs
    
    def draw_matched_balls(self):
        index_A, index_B = 3, 4
        matched_pairs = self.match_rays_from_2_cameras(index_A, index_B)

        positions = [self.cameras[index_A].position, self.cameras[index_B].position]
        rays_A, rays_B = self.cameras[index_A].current_rays, self.cameras[index_B].current_rays

        self.hide_points()

        for i, match in enumerate(matched_pairs):
            correct_rays = []
            correct_rays.append(rays_A[match["cam_A_index"]])
            correct_rays.append(rays_B[match["cam_B_index"]])

            pos = triangulate_n_rays(positions, correct_rays)
            self.points[i].move(pos)
            
