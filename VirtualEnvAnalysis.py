import pyvista as pv
import numpy as np 
from VirtualEnvHelpers import *
import scipy
import networkx as nx
from itertools import combinations
from VirtualPointManager import PointManager
from Calibration import *


class VirtualCamera():
    def __init__(self, position=(0, 0, 0), R_matrix=None, intrinsics=None):
        self.position = np.array(position, dtype=np.float64)
        if type(R_matrix) == type(None):
            R_matrix = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.R_matrix = np.array(R_matrix, dtype=np.float64)
        self.intrinsics = intrinsics
        fx, fy, cx, cy = intrinsics
        self.camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float64)

        self.vista = create_camera_frustum(1)

        self.current_rays = []
        self.bool_rays_match = []
        self.unmatched_rays = []

        self.dummy_tubes = []
        self.used_dummy_tubes = []
        self.none_line = pv.Line((0, 0, 0), (0, 0, 0.1)).tube(radius=0.001)
        for i in range(15):
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

    def update_unmatched_rays(self):
        unmatched_rays = []
        for i in range(len(self.bool_rays_match)):
            if self.bool_rays_match[i] == False:
                unmatched_rays.append(self.current_rays[i])
        self.unmatched_rays = unmatched_rays

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
        self.current_points = points
        return new_rays
        
    def add_ray(self, ray, index=0, pos=(0, 0, 0)):
        pos = np.array(pos)
        start = self.position
        end = start + np.linalg.norm(start-pos)*ray

        line = pv.Line(start, end)
        self.ray_actors[index].SetVisibility(True)
        self.dummy_tubes[index].points = line.tube(radius=0.02).points

    def add_all_rays_from_points(self, all_points):
        all_rays = self.rays_to_all_points(all_points)
        
        self.hide_rays()
        
        for i, ray in enumerate(all_rays):
            self.add_ray(ray, i)
        return all_rays
    
    def project_point(self, positions):
        if len(positions) == 0:
            return None
        rvec, _ = cv2.Rodrigues(self.R_matrix.T)
        tvec = -np.dot(self.R_matrix.T, self.position)
        positions = np.array(positions)
        res, _ = cv2.projectPoints(positions, rvec, tvec, self.camera_matrix, distCoeffs=None)
        cv2_pixel_coords = res[:, 0]
        # Filp y coordinate to match OpenCV's convention (if needed)
        #cv2_pixel_coords[1] = self.camera_matrix[1, 2] - (cv2_pixel_coords[1] - self.camera_matrix[1, 2])
        return cv2_pixel_coords
    
    
class GlobalRayManager():
    def __init__(self, cameras : list[VirtualCamera]):
        self.cameras = cameras
        self.camera_count = len(cameras)

        self.ray_point = [[] for i in range(self.camera_count)]

        self.point_manager = PointManager()


    def add_to_plotter(self, plotter):
        self.point_manager.add_to_plotter(plotter)
    
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

        new_pos = []

        for i, match in enumerate(matched_pairs):
            correct_rays = []
            correct_rays.append(rays_A[match["cam_A_index"]])
            correct_rays.append(rays_B[match["cam_B_index"]])

            pos = triangulate_n_rays(positions, correct_rays)
            new_pos.append(pos)

        self.point_manager.update(new_pos)

    def cluster_n_camera_rays(self, distance_threshold=0.01, min_cameras=2):
        """
        Groups rays from N cameras into valid 3D balls using Graph Cliques.
        
        Parameters:
            all_cameras_rays: A list of lists. 
                E.g., [ [(o1, d1), (o2, d2)], [(o3, d3)], ... ]
                Index of the outer list is the camera ID.
            distance_threshold: Max distance for two rays to be considered intersecting.
            min_cameras: Minimum number of cameras that must see the ball to accept it.
            
        Returns:
            A list of grouped ray data ready for triangulation.
            [
                [ (o1, d1), (o3, d3), ... ], # Ball 1's rays
                [ (o2, d2), ... ]            # Ball 2's rays
            ]
        """
        G = nx.Graph()
        
        # 1. Add all rays as nodes to the graph
        # Node ID format: (camera_index, ray_index)
        for cam_idx, camera in enumerate(self.cameras):
            rays = camera.unmatched_rays
            for ray_idx, ray_data in enumerate(rays):
                G.add_node((cam_idx, ray_idx), origin=camera.position, direction=ray_data)
                
        # 2. Draw Edges between intersecting rays from DIFFERENT cameras
        # We use itertools.combinations to easily loop through unique pairs of cameras
        camera_indices = range(len(self.cameras))
        
        for idx_cam_A, idx_cam_B in combinations(camera_indices, 2):
            cam_A = self.cameras[idx_cam_A]
            cam_B = self.cameras[idx_cam_B]
            rays_A = cam_A.unmatched_rays
            rays_B = cam_B.unmatched_rays
            
            for idx_A, ray_A in enumerate(rays_A):
                for idx_B, ray_B in enumerate(rays_B):
                    
                    dist = calculate_ray_to_ray_distance(
                        cam_A.position, ray_A, 
                        cam_B.position, ray_B
                    )
                    
                    if dist < distance_threshold:
                        # They intersect! Connect them in the graph.
                        G.add_edge((idx_cam_A, idx_A), (idx_cam_B, idx_B))
                        
        # 3. Find Maximal Cliques using Bron-Kerbosch
        cliques = list(nx.find_cliques(G))
        
        # 4. Filter and Extract the valid balls
        valid_balls_rays = []
        
        for clique in cliques:
            # A clique is a list of node IDs: [(0, 1), (1, 0), (2, 2)]
            if len(clique) >= min_cameras:
                
                # Extract the actual mathematical ray data for this ball
                ball_rays = []
                for node_id in clique:
                    node_data = G.nodes[node_id]
                    ball_rays.append((node_id[0], node_data['direction']))
                    
                valid_balls_rays.append(ball_rays)
                
        return valid_balls_rays
            
    def draw_from_clique_finding(self):
        valid_balls_rays = self.cluster_n_camera_rays(min_cameras=3, distance_threshold=0.05)

        self.ray_point = [[] for i in range(self.camera_count)]
        new_positions = []
        for i, single_point_rays in enumerate(valid_balls_rays):
            positions, correct_rays = [], []
            for camera_idx, ray in single_point_rays:
                cam = self.cameras[camera_idx]
                positions.append(cam.position)
                correct_rays.append(ray)

            
            pos = triangulate_n_rays(positions, correct_rays)
            new_positions.append(pos)

            for camera_idx, ray in single_point_rays:
                self.ray_point[camera_idx].append((ray, pos))

        merged = merge_close_points(new_positions)
        self.point_manager.update(merged)


    def draw_rays_knowing_pos(self):
        for camera_idx, camera_info in enumerate(self.ray_point):
            self.cameras[camera_idx].hide_rays()
            for i, (ray, pos) in enumerate(camera_info):
                self.cameras[camera_idx].add_ray(ray, i, pos)


    def predict_next_points_for_camera(self, camera_index):
        cam = self.cameras[camera_index]
        predicted_points = []
        predicted_screen_pos = []

        points = self.point_manager.get_visible_points()
        for i in points:
            predicted_points.append(i.predict_position())
        predicted_screen_pos = cam.project_point(predicted_points)
        
        if predicted_screen_pos is None:
            return None
        return predicted_screen_pos
    
    def match_predicted_point_to_true(self, camera_index, max_dist=15):
        cam = self.cameras[camera_index]
        predicted_screen_pos = self.predict_next_points_for_camera(camera_index)
        cam.bool_rays_match = [False for i in range(len(cam.current_rays))]

        if predicted_screen_pos is None:
            cam.update_unmatched_rays()
            return None
        
        matched_pairs = [None for i in range(len(predicted_screen_pos))]
        true_screen_pos = cam.current_points

        for i, pred_pos in enumerate(predicted_screen_pos):
            min_dist, min_point = np.inf, 0
            for j, true_pos in enumerate(true_screen_pos):
                dist = np.linalg.norm(pred_pos-true_pos)
                if dist < min_dist:
                    min_dist = dist
                    min_point = j

            if min_dist < max_dist:
                matched_pairs[i] = cam.current_rays[min_point]
                cam.bool_rays_match[min_point] = True

        cam.update_unmatched_rays()
        
        return matched_pairs
    
    def position_based_on_match_rays(self):
        matched_ray_pos_cameras = []

        visible_points = self.point_manager.get_visible_points()
        point_count = len(visible_points)

        for i in range(self.camera_count):
            cam = self.cameras[i]
            matched_pairs = self.match_predicted_point_to_true(i)
            if matched_pairs is None:
                matched_ray_pos_cameras.append(None)
                continue
            matched_ray_pos_cameras.append(matched_pairs)

            
        possible_new_positions = [None for i in range(point_count)]
        for point_index in range(point_count):
            rays = []
            rays_origin = []
            for camera_index in range(self.camera_count):
                if matched_ray_pos_cameras[camera_index] is None:
                    continue
                ray = matched_ray_pos_cameras[camera_index][point_index]
                if ray is None:
                    continue
                rays.append((camera_index, ray))
            
            if len(rays)>=3:
                origin = []
                directions = []
                for cam_idx, ray in rays:
                    origin.append(self.cameras[cam_idx].position)
                    directions.append(ray)
                pos = triangulate_n_rays(origin, directions)

                visible_points[point_index].update_from_pixel_prediction(pos)


            
    def update_frame(self):
        self.draw_from_clique_finding()
        self.position_based_on_match_rays()
        #self.ray_manager.draw_rays_knowing_pos()
        
