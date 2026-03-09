from VistaRod import Rod
import pyvista as pv # type: ignore
import numpy as np # type: ignore
import time

class Simulation:
    def __init__(self):
        self.running = True
        # 1. Create the shared "Player" object
        # We save this as self.player so we can modify it later
        self.player_mesh = pv.Sphere(radius=0.5, center=(0, 0, 0))
        
        # Create other static objects
        self.static_spheres = [
            pv.Sphere(radius=0.3, center=(2, 0, 0)),
            pv.Sphere(radius=0.3, center=(-2, 0, 1))
        ]

        self.positions_of_spheres = [
            np.array([2.0, 0.0, 0.0]),
            np.array([-2.0, 0.0, 1.0])
        ]

        self.pointer = Rod(length=4)
        self.pointer.initiate_random_movement()
        # Add to plotter

        # 2. Setup the Plotter with 3 subplots
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 600), title="Real-Time Control")

        # Define camera positions
        self.cam2_pos = (8, 0, 2)
        self.cam3_pos = (0, 8, 2)

        # --- SETUP VIEW 1 (Main) ---
        self.plotter.subplot(0, 0)
        self.plotter.add_text("View 1: Main\n[WASD] Move\n[Space/Shift] Up/Down", font_size=10)
        self.add_all_meshes_to_plotter(self.plotter)
        
        # Add Visual Cameras
        
        self.plotter.show_grid()

        # --- SETUP VIEW 2 (Top Down) ---
        self.plotter.subplot(0, 1)
        self.plotter.add_text("View 2: Top-Down", font_size=10)
        self.add_all_meshes_to_plotter(self.plotter)
        
        # --- SETUP VIEW 3 (Side) ---
        self.plotter.subplot(0, 2)
        self.plotter.add_text("View 3: Side", font_size=10)
        self.add_all_meshes_to_plotter(self.plotter)
            
        self.reset_cameras()

        # 3. Bind Keys
        # We pass specific vectors for each key press
        self.plotter.add_key_event("Up", lambda: self.update_position((0, 0.2, 0)))
        self.plotter.add_key_event("Down", lambda: self.update_position((0, -0.2, 0)))
        self.plotter.add_key_event("Left", lambda: self.update_position((-0.2, 0, 0)))
        self.plotter.add_key_event("Right", lambda: self.update_position((0.2, 0, 0)))
        self.plotter.add_key_event("space", lambda: self.update_position((0, 0, 0.2)))
        self.plotter.add_key_event("Shift_L", lambda: self.update_position((0, 0, -0.2)))
        self.plotter.add_key_event("t", lambda: self.reset_cameras())
        self.plotter.add_key_event("q", self.shutdown)

    def add_all_meshes_to_plotter(self, local_plotter):
        local_plotter.add_mesh(self.player_mesh, color="red")
        for s in self.static_spheres:
            local_plotter.add_mesh(s, color="cyan")
        local_plotter.add_mesh(self.pointer.vista, color="yellow")

        self.plotter.add_mesh(pv.Cone(center=self.cam2_pos, direction=(1,0,0)), color="green", opacity=0.9)
        self.plotter.add_mesh(pv.Cone(center=self.cam3_pos, direction=(0,1,0)), color="yellow", opacity=0.9)

    def reset_cameras(self):
        self.plotter.subplot(0, 1)

        self.plotter.camera.position = self.cam2_pos
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)
        self.plotter.camera.clipping_range = (0.6, 1000)

        self.plotter.subplot(0, 2)
        self.plotter.camera.position = self.cam3_pos
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
        self.plotter.render()

    def start(self):
        self.reset_cameras()
        self.plotter.show(interactive_update=True)

        while self.running:
            self.animate_step()
            self.rotate_rod()
            self.plotter.update()
            time.sleep(0.01)
            
        print("Closing simulation...")
        self.plotter.close()

    def shutdown(self):
        # This function is called when you press Q
        print("Shutdown signal received...")
        self.running = False

if __name__ == "__main__":
    sim = Simulation()
    sim.start()