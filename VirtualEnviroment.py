import pyvista as pv
import numpy as np 
from Calibration import *
import copy

class VirtualEnviroment:
    def __init__(self):
        self.plotter = pv.Plotter(shape=(1, 2), window_size=(1500, 400), title="Real-Time Control")

        self.plotter.subplot(0, 0)
        self.plotter.show_grid(bounds = (-5.0, 10.0, -5.0, 10.0, 0.0, 5.0))

        self.frustum_actors = []

    def add_camera_frustum(self):
        # 1. Generate the base frustum at the world origin
        camera_frustum = create_camera_frustum(scale=1)

        self.plotter.subplot(0, 0)
        # 2. Add it to the plotter as a wireframe and save the actor
        self.frustum_actors.append(
            self.plotter.add_mesh(
                camera_frustum, 
                style='wireframe',  # Makes it transparent with lines
                color='cyan', 
                line_width=3
            )
        )

    def update_camera_frustum(self, camera_index, camera_position, R_matrix):
        """
        Teleports and rotates the wireframe frustum to match the solved camera pose.
        """
        # 1. Create a blank 4x4 identity matrix
        transform_matrix = np.eye(4, dtype=np.float64)
        
        # 2. Slot in the 3x3 Rotation Matrix
        # We transpose it (.T) because R_matrix maps World -> Camera.
        # To move a 3D object, we need to map Camera -> World.
        transform_matrix[:3, :3] = R_matrix
        
        # 3. Slot in the 3D Camera Position (Translation)
        transform_matrix[:3, 3] = camera_position
        
        # 4. Apply the transformation directly to the actor's GPU memory
        self.frustum_actors[camera_index].user_matrix = transform_matrix

    def show_plotter(self):
        self.plotter.show(interactive_update=True)

    def update_plotter(self):
        self.plotter.update()

    def close_plotter(self):
        self.plotter.close()

    def initialize_calibration(self, camera_count, cameras_intrinsics):
        self.calibration_info = [[] for i in range(camera_count)]
        self.already_solved_frames = [[] for i in range(camera_count)]
        # Single entry = (true_position, screen_position)
        self.cameras_intrinsics = cameras_intrinsics # (camera_count, 3, 3)
        self.cameras_positions = [[0, 0, 0] for i in range(camera_count)]
        self.cameras_rotations = [[] for i in range(camera_count)]
        for i in range(camera_count):
            self.add_camera_frustum()

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
        if len(screen_solved_frames) < 4:
            return
        
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
        forward_vector = R.T[2, :]
        
        # 2. Extract the Down vector (2nd row of R) and invert it to get Up
        down_vector = R.T[1, :]
        up_vector = -down_vector
        
        # 3. Calculate the focal point (what the camera is looking at)
        # We just add the forward vector to the camera's position
        focal_pt = pos + 3*forward_vector

        #self.plotter.subplot(0, camera_index+1)
        #self.plotter.show_grid()
#
        #self.plotter.camera.position = pos
        #self.plotter.camera.focal_point = focal_pt
        #self.plotter.camera.up = up_vector
        #self.plotter.camera.clipping_range = (0.6, 1000)

        self.update_camera_frustum(camera_index, pos, R)

        print(pos, focal_pt, up_vector)




def create_camera_frustum(scale=2.0, aspect_ratio=2.5):
    """
    Creates a wireframe pyramid representing a camera's field of view.
    scale: How long the camera frustum is drawn in the 3D world.
    """
    z = scale
    x = scale * 0.5 * aspect_ratio
    y = scale * 0.5
    
    # Define the 5 points of the camera
    points = np.array([
        [0, 0, 0],       # Point 0: The Camera Lens (Origin)
        [-x, -y, z],     # Point 1: Top-Left
        [x, -y, z],      # Point 2: Top-Right
        [x, y, z],       # Point 3: Bottom-Right
        [-x, y, z],      # Point 4: Bottom-Left
        [0, -y*1.5, z],  # Point 5: Middle
    ], dtype=np.float64)
    
    # Define the faces connecting the points
    # The first number is how many points make up the face, followed by the point indices
    faces = np.hstack([
        [4, 1, 2, 3, 4], # The rectangular "screen" base
        [3, 0, 1, 2],    # Top triangle
        [3, 0, 2, 3],    # Right triangle
        [3, 0, 3, 4],    # Bottom triangle
        [3, 0, 4, 1],     # Left triangle
        [3, 1, 2, 5]
    ])
    
    return pv.PolyData(points, faces)