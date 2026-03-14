import pyvista as pv
import numpy as np 
from Calibration import *
from VirtualEnvHelpers import *
from VirtualEnvAnalysis import *
import copy

class VirtualEnviroment:
    def __init__(self):
        self.plotter = pv.Plotter(shape=(1, 1), window_size=(1500, 400), title="Real-Time Control")

        line = pv.Line((0, 0, 0), (0, 0, 0.1))
        self.tubes = []
        self.last_rays = []
        for i in range(100):
            self.last_rays.append(None)
            self.tubes.append(line.tube(radius=0.05))

        self.plotter.subplot(0, 0)
        self.plotter.show_grid(bounds = (-5.0, 10.0, -5.0, 10.0, 0.0, 5.0))

        self.ball = VirtualPoint()
        self.plotter.add_mesh(self.ball.vista, color="blue")

        self.frustum_actors = []

    def show_plotter(self):
        self.plotter.show(interactive_update=True)

    def update_plotter(self):
        self.plotter.update()

    def close_plotter(self):
        self.plotter.close()

    def initialize_calibration(self, camera_count, cameras_intrinsics):
        self.camera_count = camera_count

        self.calibration_info = [[] for i in range(camera_count)]
        self.already_solved_frames = [[] for i in range(camera_count)]
        # Single entry = (true_position, screen_position)
        self.cameras = [VirtualCamera(intrinsics = cameras_intrinsics[i]) for i in range(camera_count)]

        self.calibrated_cameras = [False for i in range(camera_count)]
        self.plotter.subplot(0, 0)
        for i in range(camera_count):
            self.add_camera_frustum(i)
            self.plotter.add_mesh(self.tubes[i], color="yellow")
    
    def add_camera_frustum(self, camera_index):
        actor = self.plotter.add_mesh(
            self.cameras[camera_index].vista, 
            style='wireframe',  # Makes it transparent with lines
            color='cyan', 
            line_width=2
        )
        self.cameras[camera_index].add_pyvista_actor(actor)

    def fake_calibration(self, cameras_positions):
        for i, pos in enumerate(cameras_positions):
            new_rot = pyvista_to_opencv_rotation(pos, (0, 0, 0), (0, 0, 1))
            self.cameras[i].move_camera(pos, new_rot)
            self.calibrated_cameras[i] = True

    def add_frame_for_calibration(self, true_pos, screen_pos, camera_index):
        a, b = copy.deepcopy(true_pos), copy.deepcopy(screen_pos)
        self.calibration_info[camera_index].append([a, b])
        print(f"Added frame {len(self.calibration_info[camera_index])} for camera {camera_index}.")

    def calibrate_single_frame(self, true_pos, screen_pos, camera_index):
        """
            Calibrating based on info from a single frame
        """
        camera_intrinsics = self.cameras[camera_index].intrinsics
        solved_pnp, best_combo, best_error = find_camera_position_and_rotation_from_3_fixed_balls(
            true_positions=true_pos,
            video_positions=screen_pos,
            camera_intrinsics=camera_intrinsics)
        
        if solved_pnp is None:
            print("Could not solve PnP with the given points.")
            return None
        
        sorted_screen_pos = []

        for i in range(len(best_combo)):
            j = best_combo[i]
            sorted_screen_pos.append(screen_pos[j])
        
        camera_position, camera_rotation = rvec_tvec_to_camera_pose(solved_pnp[0], solved_pnp[1])
        #print(f"Best combo: {best_combo}, Reprojection error: {best_error}")

        return camera_position, camera_rotation, sorted_screen_pos

    def compile_list_of_all_solved_frames(self, camera_index):
        true_solved_frames = []
        screen_solved_frames = []
        N = len(self.already_solved_frames[camera_index])

        for i in range(N):
            if self.already_solved_frames[camera_index][i] is None:
                continue
            temp = self.calibration_info[camera_index][i]
            true_solved_frames.extend(temp[0])
            screen_solved_frames.extend(temp[1])

        return true_solved_frames, screen_solved_frames
 
    def calibrate_camera(self, camera_index):
        """
            Find the best position based on info from many frames
        """

        print(f"------------Calibrating camera {camera_index}------------")

        already_solved_count = len(self.already_solved_frames[camera_index])
        all_frames_count = len(self.calibration_info[camera_index])
        for i in range(already_solved_count, all_frames_count):
            i_frame = self.calibration_info[camera_index][i]
            i_true_pos, i_screen_pos = i_frame

            if len(i_true_pos) != len(i_screen_pos):
                res = None
            else:
                res = self.calibrate_single_frame(i_true_pos, i_screen_pos, camera_index)

            if res is not None:
                print(res[0])
                self.calibration_info[camera_index][i][1] = res[2]

                self.already_solved_frames[camera_index].append((res[0], res[1]))
            else:
                self.already_solved_frames[camera_index].append(None)

        true_solved_frames, screen_solved_frames = self.compile_list_of_all_solved_frames(camera_index)
        camera_intrinsics = self.cameras[camera_index].intrinsics
        camera_matrix = intrictics_to_matrix(camera_intrinsics)
        print(len(true_solved_frames), len(screen_solved_frames))
        if len(screen_solved_frames) < 4:
            return
        
        rvec, tvec, inliers = solve_global_camera_ransac(true_solved_frames, screen_solved_frames, camera_matrix)

        camera_position_ransac, camera_rotation_ransac = rvec_tvec_to_camera_pose(rvec, tvec)
        print(f"Position found by Ransac: {camera_position_ransac}")

        self.cameras[camera_index].move_camera(camera_position_ransac, camera_rotation_ransac)
        self.calibrated_cameras[camera_index] = True

    def calculate_line_from_camera_to_points(self, camera_index, point):
        if self.calibrated_cameras[camera_index] == False:
            return None

        camera_pos = self.cameras[camera_index].position
        fx, fy, cx, cy = self.cameras[camera_index].intrinsics

        pixel_x, pixel_y = point

        x = (pixel_x - cx) / fx
        y = (pixel_y - cy) / fy

        Z = 1
        X = x*Z
        Y = y*Z

        R = self.cameras[camera_index].R_matrix
        camera_to_point_rotated = np.array([X, Y, Z])

        ray_direction_world = R @ camera_to_point_rotated

        ray_direction_world = ray_direction_world / np.linalg.norm(ray_direction_world)

        return ray_direction_world
    
    def add_line_from_camera_to_point(self, camera_index, point):
        if point == None:
            self.last_rays[camera_index] = None
            return
        ray = self.calculate_line_from_camera_to_points(camera_index, point)
        self.last_rays[camera_index] = ray
        start = self.cameras[camera_index].position
        end = start + np.linalg.norm(start)*1.2*ray

        line = pv.Line(start, end)
        self.tubes[camera_index].points = line.tube(radius=0.1).points

    def update_ball_position(self):
        real_rays = []
        real_pos = []
        for i in range(self.camera_count):
            if type(self.last_rays[i]) == type(None):
                continue
            real_rays.append(self.last_rays[i])
            real_pos.append(self.cameras[i].position)
        if len(real_pos) < 2:
            return
        ball_pos = triangulate_n_rays(real_pos, real_rays)
        self.ball.move(ball_pos)
        
        return ball_pos
