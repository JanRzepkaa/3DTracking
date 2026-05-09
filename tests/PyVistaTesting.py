import pyvista as pv

class Simulation:
    def __init__(self):
        # 1. Create the shared "Player" object
        # We save this as self.player so we can modify it later
        self.player_mesh = pv.Sphere(radius=0.5, center=(0, 0, 0))
        
        # Create other static objects
        self.static_spheres = [
            pv.Sphere(radius=0.3, center=(2, 0, 0)),
            pv.Sphere(radius=0.3, center=(-2, 0, 1))
        ]

        # 2. Setup the Plotter with 3 subplots
        self.plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 600), title="Real-Time Control")

        # Define camera positions
        self.cam2_pos = (0, 0, 8)
        self.cam3_pos = (8, 0, 0)

        # --- SETUP VIEW 1 (Main) ---
        self.plotter.subplot(0, 0)
        self.plotter.add_text("View 1: Main\n[WASD] Move\n[Space/Shift] Up/Down", font_size=10)
        self.plotter.add_mesh(self.player_mesh, color="red", label="Player") # Red is player
        for s in self.static_spheres:
            self.plotter.add_mesh(s, color="cyan")
        
        # Add Visual Cameras
        self.plotter.add_mesh(pv.Cone(center=self.cam2_pos, direction=(0,0,-1)), color="green", opacity=0.5)
        self.plotter.add_mesh(pv.Cone(center=self.cam3_pos, direction=(-1,0,0)), color="yellow", opacity=0.5)
        self.plotter.show_grid()

        # --- SETUP VIEW 2 (Top Down) ---
        self.plotter.subplot(0, 1)
        self.plotter.add_text("View 2: Top-Down", font_size=10)
        # We add the SAME self.player_mesh object here
        self.plotter.add_mesh(self.player_mesh, color="red") 
        for s in self.static_spheres:
            self.plotter.add_mesh(s, color="cyan")
        
        self.plotter.camera.position = self.cam2_pos
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 1, 0)
        self.plotter.camera.zoom(1.2)

        # --- SETUP VIEW 3 (Side) ---
        self.plotter.subplot(0, 2)
        self.plotter.add_text("View 3: Side", font_size=10)
        # We add the SAME self.player_mesh object here too
        self.plotter.add_mesh(self.player_mesh, color="red")
        for s in self.static_spheres:
            self.plotter.add_mesh(s, color="cyan")
            
        self.plotter.camera.position = self.cam3_pos
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.up = (0, 0, 1)

        # 3. Bind Keys
        # We pass specific vectors for each key press
        self.plotter.add_key_event("w", lambda: self.update_position((0, 0.2, 0)))
        self.plotter.add_key_event("s", lambda: self.update_position((0, -0.2, 0)))
        self.plotter.add_key_event("a", lambda: self.update_position((-0.2, 0, 0)))
        self.plotter.add_key_event("d", lambda: self.update_position((0.2, 0, 0)))
        self.plotter.add_key_event("space", lambda: self.update_position((0, 0, 0.2)))
        self.plotter.add_key_event("Shift_L", lambda: self.update_position((0, 0, -0.2)))

    def update_position(self, change_vector):
        # This is the magic part:
        # We translate the MESH itself. Because all 3 views share this mesh,
        # they all update instantly.
        self.player_mesh.translate(change_vector, inplace=True)
        
        # We must tell the plotter to redraw
        self.plotter.render()

    def start(self):
        self.plotter.show()

if __name__ == "__main__":
    sim = Simulation()
    sim.start()