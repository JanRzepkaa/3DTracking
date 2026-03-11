import pyvista as pv
import numpy as np 
from Calibration import *
import copy

class VirtualEnviroment:
    def __init__(self):
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 400), title="Real-Time Control")

    def initialize_calibration(self, camera_count, cameras_intrinsics):
        self.calibration_info = [[] for i in range(camera_count)]
        self.already_solved_frames = [[] for i in range(camera_count)]
        # Single entry = (true_position, screen_position)
        self.cameras_intrinsics = cameras_intrinsics # (camera_count, 3, 3)
        self.cameras_positions = [[0, 0, 0] for i in range(camera_count)]
        self.cameras_rotations = [[] for i in range(camera_count)]

    def add_frame_for_calibration(self, true_pos, screen_pos, camera_index):
        a, b = copy.deepcopy(true_pos), copy.deepcopy(screen_pos)
        self.calibration_info[camera_index].append([a, b])
        print(f"Added frame {len(self.calibration_info[camera_index])} for camera {camera_index}.")

    def calibrate_single_frame(self, true_pos, screen_pos, camera_index):
        """
            Calibrating based on info from a single frame
        """
        camera_intrinsics = self.cameras_intrinsics[camera_index]
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

        print(f"------------Calibrating camera {camera_index+1}------------")

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
        camera_intrinsics = self.cameras_intrinsics[camera_index]
        camera_matrix = intrictics_to_matrix(camera_intrinsics)
        print(len(true_solved_frames), len(screen_solved_frames))
        rvec, tvec, inliers = solve_global_camera_ransac(true_solved_frames, screen_solved_frames, camera_matrix)

        camera_position_ransac, camera_rotation_ransac = rvec_tvec_to_camera_pose(rvec, tvec)
        print(f"Position found by Ransac: {camera_position_ransac}")

        self.cameras_positions[camera_index]=camera_position_ransac
        self.cameras_rotations[camera_index]=camera_rotation_ransac

        self.update_virtual_camera_position(camera_index)

    def update_virtual_camera_position(self, camera_index):
        """
            Update pyVista camera position
        """
        
        pos, rot = self.cameras_positions[camera_index], self.cameras_rotations[camera_index]

        pos = np.array(pos, dtype=np.float64).flatten()
        R = np.array(rot, dtype=np.float64)
        
        # 1. Extract the Forward vector (3rd row of R)
        forward_vector = R[2, :]
        
        # 2. Extract the Down vector (2nd row of R) and invert it to get Up
        down_vector = R[1, :]
        up_vector = -down_vector
        
        # 3. Calculate the focal point (what the camera is looking at)
        # We just add the forward vector to the camera's position
        focal_pt = pos + forward_vector

        self.plotter.camera.position = pos
        self.plotter.camera.focal_point = focal_pt
        self.plotter.camera.up = up_vector
        self.plotter.camera.clipping_range = (0.6, 1000)

        print(pos, focal_pt, up_vector)
