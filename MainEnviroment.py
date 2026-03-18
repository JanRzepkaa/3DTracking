from PyVistaTracker import PyVistaTracker
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

        # 1. Create the shared "Player" object
        # We save this as self.player so we can modify it later
        self.player_position = np.array([0.0, 3.0, 1.0])
        self.player_mesh = pv.Sphere(radius=0.01, center=tuple(self.player_position))
        # Create other static objects

        self.positions_of_spheres = [
            np.array([2, 0, 0], dtype=np.float32),
            np.array([-2, -1, 1], dtype=np.float32),
            np.array([-5, 0, 1.5], dtype=np.float32),
            np.array([-2.0, -2.0, 2.0], dtype=np.float32),
            np.array([0.0, 0.0, 4.0], dtype=np.float32),
        ]

        self.static_spheres = [
            pv.Sphere(radius=0.2, center=tuple(self.positions_of_spheres[0].copy())),
            pv.Sphere(radius=0.2, center=tuple(self.positions_of_spheres[1].copy())),
            pv.Sphere(radius=0.2, center=tuple(self.positions_of_spheres[2].copy())),
            pv.Sphere(radius=0.2, center=tuple(self.positions_of_spheres[3].copy())),
            pv.Sphere(radius=0.2, center=tuple(self.positions_of_spheres[4].copy())),
            #pv.Sphere(radius=0.3, center=(-2, -2, 2))
        ]

        self.actors = {"tracked": [], "other": []}
        # Define camera positions
        self.camera_positions = {
            0: (5, 5, 5),
            1: (10, 0, 2),
            2: (0, 10, 1),
            3: (6, 9, 7),
            4: (-10, -10, 10),
            5: (10, -10, 10)
        }

        self.camera_count = len(self.camera_positions)-1

        self.pointer = Rod(length=4)
        self.pointer.initiate_random_movement()
        # Add to plotter

        # 2. Setup the Plotter with 3 subplots
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 400), title="Real-Time Control")
        
        view = [(0, "View 1 Main"), (1, "View 2 Left"), (2, "View 3 Right")]
        for i, text in view:
            self.plotter.subplot(0, i)
            self.plotter.add_text(text, font_size=10)
            self.add_all_meshes_to_plotter(self.plotter, i)
            
        self.plotter.subplot(0, 0)
        self.plotter.show_grid(bounds = (-5.0, 10.0, -5.0, 10.0, 0.0, 5.0))

        cameras_info = [
            {"position":(10, 0, 2), "resolution":(1200, 800), "focal_point":(0, 0, 1)},
            {"position":(0, 10, 1), "resolution":(1200, 800), "focal_point":(0, 0, 1)},
            {"position":(6, 9, 7), "resolution":(1200, 800), "focal_point":(0, 0, 1)},
            {"position":(-10, -10, 10), "resolution":(1200, 800), "focal_point":(0, 0, 1)},
            {"position":(10, -10, 10), "resolution":(1200, 800), "focal_point":(0, 0, 1)},
        ]

        self.vista_tracker = PyVistaTracker(cameras_info, self.actors)

        self.reset_cameras()

        # 3. Bind Keys
        self.add_key_events()

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
            ("c", lambda: self.vista_tracker.calibrate_virtual_cameras()),
            ("f", self.vista_tracker.fake_calibration),
        ]

        for i in range(1, self.camera_count+1):
            key_events.append(
                (str(i if i < 3 else i + 1),
                lambda i=i: self.vista_tracker.add_frame_for_calibration(i-1, "tracked"))
            )

        for key, func in key_events:
            self.plotter.add_key_event(key, func)

    def change_cv_window_visibility(self):
        self.show_cv_window = not self.show_cv_window
        self.cv_window.change_visibility(self.show_cv_window)

    def add_all_meshes_to_plotter(self, local_plotter, subplot_index=None):
        if subplot_index is not None:
            local_plotter.subplot(0, subplot_index)
        actor = local_plotter.add_mesh(self.player_mesh, color="blue")
        if subplot_index == 1: self.actors["other"].append(actor)
        for s in self.static_spheres:
            actor = local_plotter.add_mesh(s, color="lime")
            if subplot_index == 1: self.actors["tracked"].append(actor)
        actor = local_plotter.add_mesh(self.pointer.vista, color="yellow")
        if subplot_index == 1: self.actors["other"].append(actor)
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
        self.vista_tracker.start_plotters()

        self.cv_window.startWindow()
        self.vista_tracker.fake_calibration()

        while self.running:
            self.animate_step(0)
            self.animate_step(1, 0.03)
            self.animate_step(2, -0.03)
            self.animate_step(3, -0.1)
            self.rotate_rod()
            self.plotter.update()
            self.vista_tracker.update_plotters()

            img_rgb = self.vista_tracker.get_raw_view_from_camera(3)

            self.cv_window.update_from_pyvista_screenshot(img_rgb)

            self.vista_tracker.track_player()

            # 5. Handle OpenCV events (Required to keep window responsive)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.shutdown()
            time.sleep(0.01)

            
        print("Closing simulation...")
        self.cv_window.shutdown()
        self.plotter.close()
        self.vista_tracker.close_plotter()


    def shutdown(self):
        # This function is called when you press Q
        print("Shutdown signal received...")
        self.running = False

if __name__ == "__main__":
    sim = Simulation()
    sim.start()