from VistaRod import Rod
import pyvista as pv # type: ignore
import numpy as np # type: ignore
import time
import cv2 # type: ignore
from AnalyzePyVistaVideo import AnalyzePyVistaVideo

class Simulation:
    def __init__(self):
        self.running = True
        self.show_cv_window = True
        self.cv_window = AnalyzePyVistaVideo(self.show_cv_window)
        # 1. Create the shared "Player" object
        # We save this as self.player so we can modify it later
        self.player_mesh = pv.Sphere(radius=0.5, center=(0, 0, 1))
        self.player_position = np.array([0.0, 0.0, 1])
        # Create other static objects
        self.static_spheres = [
            pv.Sphere(radius=0.3, center=(2, 0, 0)),
            pv.Sphere(radius=0.3, center=(-2, 0, 1))
        ]

        self.positions_of_spheres = [
            np.array([2.0, 0.0, 0.0]),
            np.array([-2.0, 0.0, 1.0])
        ]

        # Define camera positions
        self.camera_positions = {
            0: (5, 5, 5),
            1: (8, 0, 2),
            2: (0, 8, 1)
        }

        self.pointer = Rod(length=4)
        self.pointer.initiate_random_movement()
        # Add to plotter

        # 2. Setup the Plotter with 3 subplots
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 600), title="Real-Time Control")
        self.hidden_plotter = pv.Plotter(off_screen=True, window_size=(1000, 400))  # Hidden plotter for off-screen rendering

        self.add_all_meshes_to_plotter(self.hidden_plotter)
        self.hidden_plotter.camera.position = self.camera_positions[2]
        self.hidden_plotter.camera.focal_point = (0, 0, 1)
        self.hidden_plotter.camera.up = (0, 0, 1)
        self.hidden_plotter.camera.clipping_range = (0.6, 1000)

        self.cv_window.initialaze_calibration_test_data(
            ball_position=self.player_position,
            camera_position=self.camera_positions[2],
            camera_rotation=(np.pi/2, 0, 0),
            camera_intrinsics=self.calculate_camera_intrinsics(self.hidden_plotter)
        )

        print("Camera Intrinsics:", self.calculate_camera_intrinsics(self.hidden_plotter))

        view = [(0, "View 1 Main"), (1, "View 2 Left"), (2, "View 3 Right")]
        for i, text in view:
            self.plotter.subplot(0, i)
            self.plotter.add_text(text, font_size=10)
            self.add_all_meshes_to_plotter(self.plotter, i)
            
        self.plotter.subplot(0, 0)
        self.plotter.show_grid()

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
        fovy_rad = np.deg2rad(fovy)

        fy = height / (2 * np.tan(fovy_rad / 2))
        fx = fy * (width / height)

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
            ("t", lambda: self.reset_cameras()),
            ("q", self.shutdown),
            ("v", self.change_cv_window_visibility)
        ]

        for key, func in key_events:
            self.plotter.add_key_event(key, func)

    def change_cv_window_visibility(self):
        self.show_cv_window = not self.show_cv_window
        self.cv_window.change_visibility(self.show_cv_window)
        
    def add_all_meshes_to_plotter(self, local_plotter, subplot_index=None):
        if subplot_index is not None:
            local_plotter.subplot(0, subplot_index)
        local_plotter.add_mesh(self.player_mesh, color="lime")
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


    def animate_step(self):
        pos = self.positions_of_spheres[0]

        dist_from_center = 2
        current_angle = np.arctan(pos[1]/pos[0])
        if pos[0]<0:
            current_angle+=np.pi
        current_angle+=0.01

        new_x = dist_from_center*np.cos(current_angle)
        new_y = dist_from_center*np.sin(current_angle)

        new_pos = np.array([new_x, new_y, pos[2]])

        difference = new_pos - pos

        self.static_spheres[0].points += difference
        self.positions_of_spheres[0] += difference

    def update_position(self, change_vector):
        self.player_mesh.points += np.array(change_vector)
        self.player_position += np.array(change_vector)
        self.plotter.render()

    def start(self):
        self.reset_cameras()
        self.plotter.show(interactive_update=True)

        self.cv_window.startWindow()

        while self.running:
            self.animate_step()
            self.rotate_rod()
            self.plotter.update()
            self.hidden_plotter.update()

            img_rgb = self.hidden_plotter.screenshot(None, return_img=True)

            self.cv_window.update_from_pyvista_screenshot(img_rgb)

            # 5. Handle OpenCV events (Required to keep window responsive)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            time.sleep(0.01)
            
        print("Closing simulation...")
        self.cv_window.shutdown()
        self.plotter.close()


    def shutdown(self):
        # This function is called when you press Q
        print("Shutdown signal received...")
        self.running = False

if __name__ == "__main__":
    sim = Simulation()
    sim.start()