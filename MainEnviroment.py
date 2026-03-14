from VistaRod import Rod
import pyvista as pv 
import numpy as np 
import time
import cv2 
from AnalyzePyVistaVideo import AnalyzePyVistaVideo
from VirtualEnviroment import VirtualEnviroment

class Simulation:
    def __init__(self):
        self.running = True
        self.show_cv_window = False
        self.cv_window = AnalyzePyVistaVideo(self.show_cv_window)

        self.virtual_env = VirtualEnviroment()
        # 1. Create the shared "Player" object
        # We save this as self.player so we can modify it later
        self.player_position = np.array([0.0, 3.0, 1.0])
        self.player_mesh = pv.Sphere(radius=0.5, center=tuple(self.player_position))
        # Create other static objects

        self.positions_of_spheres = [
            np.array([2, 0, 0], dtype=np.float32),
            np.array([-2, -1, 1], dtype=np.float32),
            np.array([-5, 0, 1], dtype=np.float32),
            np.array([-2.0, -2.0, 2.0], dtype=np.float32)
        ]

        self.static_spheres = [
            pv.Sphere(radius=0.3, center=tuple(self.positions_of_spheres[0].copy())),
            pv.Sphere(radius=0.3, center=tuple(self.positions_of_spheres[1].copy())),
            pv.Sphere(radius=0.3, center=tuple(self.positions_of_spheres[2].copy())),
            pv.Sphere(radius=0.3, center=tuple(self.positions_of_spheres[3].copy())),
            #pv.Sphere(radius=0.3, center=(-2, -2, 2))
        ]
        # Define camera positions
        self.camera_positions = {
            0: (5, 5, 5),
            1: (8, 0, 2),
            2: (0, 8, 1),
            3: (6, 6, 7),
            4: (-10, -10, 10)
        }

        self.pointer = Rod(length=4)
        self.pointer.initiate_random_movement()
        # Add to plotter

        # 2. Setup the Plotter with 3 subplots
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 400), title="Real-Time Control")

        self.camer_count = len(self.camera_positions)-1

        self.hidden_plotters = []
        for i in range(1, self.camer_count+1):
            self.hidden_plotters.append(pv.Plotter(off_screen=True, window_size=(1000, 400)))
            plt = self.hidden_plotters[i-1]

            self.add_all_meshes_to_plotter(plt)
            plt.camera.position = self.camera_positions[i]
            plt.camera.focal_point = (0, 0, 0)
            plt.camera.up = (0, 0, 1)
            plt.camera.clipping_range = (0.6, 1000)

            my_light = pv.Light(
                position=self.camera_positions[i], 
                focal_point=(0, 0, 0), 
                color='white',
                light_type='scene light' # 'scene light' means it stays fixed in the 3D world
            )
            plt.add_light(my_light)

        self.virtual_env.initialize_calibration(self.camer_count, [self.calculate_camera_intrinsics(self.hidden_plotters[1])]*self.camer_count)
        
        view = [(0, "View 1 Main"), (1, "View 2 Left"), (2, "View 3 Right")]
        for i, text in view:
            self.plotter.subplot(0, i)
            self.plotter.add_text(text, font_size=10)
            self.add_all_meshes_to_plotter(self.plotter, i)
            
        self.plotter.subplot(0, 0)
        self.plotter.show_grid(bounds = (-5.0, 10.0, -5.0, 10.0, 0.0, 5.0))

        self.reset_cameras()

        # 3. Bind Keys
        self.add_key_events()

    def calculate_camera_intrinsics(self, plotter, subplot_index=None):
        if subplot_index is not None:
            plotter.subplot(0, subplot_index)
        width, height = plotter.window_size

        cx = width / 2
        cy = height / 2

        fovy = plotter.camera.view_angle
        print("FOVY:", fovy)
        fovy_rad = np.radians(fovy)

        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy #* (width / height)

        print("Calculated Intrinsics - fx:", fx, "fy:", fy, "cx:", cx, "cy:", cy)

        camera_intrinsics = (fx, fy, cx, cy)
        return camera_intrinsics

    def add_key_events(self):
        key_events = [
            ("Up", lambda: self.update_position((0, 0.2, 0))),
            ("Down", lambda: self.update_position((0, -0.2, 0))),
            ("Left", lambda: self.update_position((-0.2, 0, 0))),
            ("Right", lambda: self.update_position((0.2, 0, 0))),
            ("space", lambda: self.update_position((0, 0, 0.2))),
            ("Shift_L", lambda: self.update_position((0, 0, -0.2))),
            ("a", lambda: self.reset_cameras()),
            ("q", self.shutdown),
            ("v", self.change_cv_window_visibility),
            ("c", lambda: self.calibrate_virtual_cameras()),
            ("f", self.fake_calibration),
            ("t", lambda: self.virtual_env.add_line_from_camera_to_point(0, (0, 0))),
            ("x", self.track_player)
        ]

        for i in range(1, self.camer_count+1):
            key_events.append(
                (str(i if i < 3 else i + 1),
                lambda i=i: self.add_frame_for_virtual_env_calibration(i-1))
            )
        

        for key, func in key_events:
            self.plotter.add_key_event(key, func)

    def fake_calibration(self):
        self.virtual_env.fake_calibration(list(self.camera_positions.values())[1:])

    def change_cv_window_visibility(self):
        self.show_cv_window = not self.show_cv_window
        self.cv_window.change_visibility(self.show_cv_window)

    def add_frame_for_virtual_env_calibration(self, camera_index):
        img_rgb = self.hidden_plotters[camera_index].screenshot(None, return_img=True)
        img_bgr = img_rgb[:, :, ::-1].copy()
        _, screen_positon = self.cv_window.find_centroids_and_contours(img_bgr)
        all_true_positions = self.positions_of_spheres
        self.virtual_env.add_frame_for_calibration(all_true_positions, screen_positon, camera_index)
        
    def calibrate_virtual_cameras(self):
        for i in range(self.camer_count):
            self.virtual_env.calibrate_camera(i)

    def add_all_meshes_to_plotter(self, local_plotter, subplot_index=None):
        if subplot_index is not None:
            local_plotter.subplot(0, subplot_index)
        local_plotter.add_mesh(self.player_mesh, color="blue")
        for s in self.static_spheres:
            local_plotter.add_mesh(s, color="lime")
        local_plotter.add_mesh(self.pointer.vista, color="yellow")

        self.plotter.add_mesh(pv.Cone(center=self.camera_positions[1], direction=(1,0,0)), color="green", opacity=0.9)
        self.plotter.add_mesh(pv.Cone(center=self.camera_positions[2], direction=(0,1,0)), color="yellow", opacity=0.9)

    def reset_cameras(self):
        for i in [1, 2]:
            self.plotter.subplot(0, i)

            self.plotter.camera.position = self.camera_positions[i]
            self.plotter.camera.focal_point = (0, 0, 0)
            self.plotter.camera.up = (0, 0, 1)
            self.plotter.camera.clipping_range = (0.6, 1000)

        self.plotter.subplot(0, 0)

    def rotate_rod(self):
        self.pointer.rotate_rod_z(0.01)
        self.pointer.rotate_rod_y(0.01)

        self.pointer.move_random()
    
    def track_player(self):
        screen_positions = [None for i in range(self.camer_count)]
        for i in range(self.camer_count):
            screenshot = self.hidden_plotters[i].screenshot(None, return_img=True)
            img_bgr = screenshot[:, :, ::-1].copy()

            _, pos = self.cv_window.find_centroids_and_contours(frame=img_bgr.copy(), color="blue")
            screen_positions[i] = pos[0] if len(pos)!=0 else None

        for i in range(self.camer_count):
            self.virtual_env.add_line_from_camera_to_point(i, screen_positions[i])


    def animate_step(self, ball_index, speed=0.01):
        pos = self.positions_of_spheres[ball_index]

        dist_from_center = 2
        current_angle = np.arctan(pos[1]/pos[0])
        if pos[0]<0:
            current_angle+=np.pi
        current_angle+=speed

        new_x = dist_from_center*np.cos(current_angle)
        new_y = dist_from_center*np.sin(current_angle)

        new_pos = np.array([new_x, new_y, pos[2]])

        difference = new_pos - pos

        self.static_spheres[ball_index].points += difference
        self.positions_of_spheres[ball_index] += difference

    def update_position(self, change_vector):
        self.player_mesh.points += np.array(change_vector)
        self.player_position += np.array(change_vector)
        self.plotter.render()

    def start(self):
        self.reset_cameras()
        self.plotter.show(interactive_update=True)
        for i in range(self.camer_count):
                self.hidden_plotters[i].show(interactive_update=True)
        self.virtual_env.show_plotter()

        self.cv_window.startWindow()
        self.fake_calibration()

        while self.running:
            self.animate_step(0)
            self.animate_step(1, 0.03)
            self.animate_step(2, -0.03)
            self.animate_step(3, -0.1)
            self.rotate_rod()
            self.plotter.update()
            for i in range(self.camer_count):
                self.hidden_plotters[i].update()
            self.virtual_env.update_plotter()

            img_rgb = self.hidden_plotters[3].screenshot(None, return_img=True)

            self.cv_window.update_from_pyvista_screenshot(img_rgb)

            self.track_player()
            virtual_pos = self.virtual_env.update_ball_position()
            dist = virtual_pos - self.player_position

            print(f"{np.linalg.norm(dist):.4f}")

            # 5. Handle OpenCV events (Required to keep window responsive)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            time.sleep(0.01)

            
        print("Closing simulation...")
        self.cv_window.shutdown()
        self.plotter.close()
        self.virtual_env.close_plotter()


    def shutdown(self):
        # This function is called when you press Q
        print("Shutdown signal received...")
        self.running = False

if __name__ == "__main__":
    sim = Simulation()
    sim.start()