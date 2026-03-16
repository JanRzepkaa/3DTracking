from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import pyvista as pv
import random
import colorsys

class VirtualVector:
    def __init__(self, color="red", scale_multiplier=10.0):
        self.color = color
        self.scale_multiplier = scale_multiplier
        self.actor = None
        
        # Create a tiny dummy arrow to initialize the GPU actor once
        self.dummy_mesh = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=0.001)

    def add_to_plotter(self, plotter):
        self.actor = plotter.add_mesh(self.dummy_mesh, color=self.color)
        self.hide()

    def update(self, start_pos, vector):
        norm = np.linalg.norm(vector)
        
        # If the vector is basically zero, hide the arrow entirely
        if norm < 0.001:
            self.hide()
            return
            
        self.show()
        
        # Rebuild the arrow geometry so the arrowhead doesn't distort
        # We multiply the visual length so it sticks out of the sphere
        new_arrow = pv.Arrow(
            start=tuple(start_pos), 
            direction=tuple(vector), 
            scale=norm * self.scale_multiplier
        )
        
        # THE TRICK: Swap the mesh data directly in the pipeline 
        # This completely skips the heavy CPU add/remove plotter overhead
        if self.actor is not None:
            self.actor.mapper.dataset = new_arrow

    def hide(self):
        if self.actor is not None:
            self.actor.SetVisibility(False)
    
    def show(self):
        if self.actor is not None:
            self.actor.SetVisibility(True)

class VirtualPoint():
    def __init__(self, point_id, alpha_v=0.1, alpha_a=0.05):
        self.id = point_id

        self.predicted_from_pixels = False
        
        # EMA Smoothing Factors (0.0 to 1.0)
        # alpha_v: How quickly velocity reacts to changes
        # alpha_a: Acceleration is naturally noisier, so we use a lower alpha to smooth it more
        self.alpha_v = alpha_v
        self.alpha_a = alpha_a
        
        # State variables
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Tracking variables
        self.is_active = False
        self.age = 0
        self.missed_frames = 0
        
        # Assign a random bright color for visual debugging
        self.color = [random.uniform(0.0, 1.0), random.uniform(0.8, 1.0),random.uniform(0.8, 1.0)]
        self.color = colorsys.hsv_to_rgb(*self.color)
        
        # Create the mesh
        self.vista = pv.Sphere(radius=0.15)
        self.predict_vista = pv.Sphere(radius=0.15)
        self.actor = None

        # Add the two visual vectors
        # Velocity = Cyan (Scaled 15x so we can see it)
        self.vel_vector = VirtualVector(color="cyan", scale_multiplier=15.0)
        # Acceleration = Magenta (Scaled heavily because accel per frame is tiny)
        self.acc_vector = VirtualVector(color="magenta", scale_multiplier=50.0)

    def add_to_plotter(self, plotter):
        self.actor = plotter.add_mesh(self.vista, color=self.color)
        self.predict_actor = plotter.add_mesh(self.vista, color=self.color, opacity=0.5)

        self.vel_vector.add_to_plotter(plotter)
        self.acc_vector.add_to_plotter(plotter)

        self.hide()

    def move(self):
        if self.actor is not None:
            self.actor.position = tuple(self.position)
            predict_pos = self.predict_position()
            self.predict_actor.position = tuple(predict_pos)

            # Update the visual vectors!
            self.vel_vector.update(self.position, self.velocity)
            self.acc_vector.update(self.position, self.acceleration)

    def update_state(self, new_position):
        """Updates physics state with EMA smoothing."""
        new_position = np.array(new_position, dtype=np.float64)
        
        if self.age == 1:
            # We need 2 frames to calculate velocity. No acceleration yet.
            self.velocity = new_position - self.position
            
        elif self.age > 1:
            # 1. Calculate the raw, noisy measurements
            raw_velocity = new_position - self.position
            raw_acceleration = raw_velocity - self.velocity
            
            # 2. Apply Exponential Moving Average (EMA)
            self.velocity = (self.alpha_v * raw_velocity) + ((1 - self.alpha_v) * self.velocity)
            self.acceleration = (self.alpha_a * raw_acceleration) + ((1 - self.alpha_a) * self.acceleration)
            
        # Update the position
        self.position = new_position
        
        # Instantly teleport the actor on the GPU
        self.move()

        # Reset miss counter and increase age
        self.missed_frames = 0
        self.age += 1
        self.predicted_from_pixels = False

    def update_from_pixel_prediction(self, new_pos):
        self.update_state(new_pos)
        self.change_color()
        self.predicted_from_pixels = True

    def predict_position(self):
        """
        Uses Newtonian kinematics to predict the next frame's position.
        """
        return self.position + self.velocity + (0.5 * self.acceleration)

    def coast(self):
        """
        Advances the point blindly along its predicted parabolic 
        trajectory when it goes missing.
        """
        # 1. Move to the predicted position
        self.position = self.predict_position()
        
        # 2. Update velocity (because gravity/acceleration is still acting on it)
        self.velocity += self.acceleration
        
        # 3. Teleport the actor
        self.move()
            
        self.mark_missed()

    def mark_missed(self):
        self.missed_frames += 1

    def reset_track(self):
        """Frees this point back to the pool."""
        self.is_active = False
        self.age = 0
        self.missed_frames = 0
        
        # Zero out physics
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        self.hide()

    def hide(self):
        if self.actor is not None:
            self.actor.SetVisibility(False)
            self.predict_actor.SetVisibility(False)

        self.vel_vector.hide()
        self.acc_vector.hide()
    
    def show(self):
        if self.actor is not None:
            self.actor.SetVisibility(True)
            self.predict_actor.SetVisibility(True)

    def change_color(self, new_color=None):
        if new_color is None:
            self.actor.prop.color = self.color
            self.predict_actor.SetVisibility(True)
            return
        self.actor.prop.color = new_color
        self.predict_actor.SetVisibility(False)


class PointManager():
    def __init__(self, pool_size=20, max_distance=0.5, min_hits=10, max_misses=30):
        """
        pool_size: Max surplus points created in memory.
        max_distance: Max physical distance (units) to match an old point to a new one.
        min_hits: Frames required to survive before becoming visible.
        max_misses: Frames allowed to be unseen before the point is deleted.
        """
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_misses = max_misses
        
        # Initialize the Object Pool
        self.pool = [VirtualPoint(i) for i in range(pool_size)]
        
    def add_to_plotter(self, plotter):
        for pt in self.pool:
            pt.add_to_plotter(plotter)


    def get_active_points(self):
        return [pt for pt in self.pool if pt.is_active]
    
    def get_active_not_predicted(self):
        return [pt for pt in self.get_active_points() if pt.predicted_from_pixels == False]
    
    def get_visible_points(self):
        return [pt for pt in self.pool if pt.is_active and pt.age > self.min_hits]

    def get_free_point(self):
        for pt in self.pool:
            if not pt.is_active:
                return pt
        return None

    def single_point_update(self, point, new_pos):
        point.update_state(new_pos)

    def update(self, new_3d_coordinates):
        """
        Takes a list of raw (X,Y,Z) tuples from the triangulator, 
        matches them, filters ghosts, and updates the PyVista screen.
        """
        new_coords = np.array(new_3d_coordinates)
        active_points = self.get_active_not_predicted()
        
        matched_new_indices = set()
        matched_active_indices = set()

        # 1. Hungarian Matching
        if len(active_points) > 0 and len(new_coords) > 0:
            # Extract positions of current active trackers
            active_coords = np.array([pt.predict_position() for pt in active_points])
            
            # Calculate distance matrix (Rows: Active, Cols: New)
            cost_matrix = cdist(active_coords, new_coords)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.max_distance:
                    # Successful match
                    active_points[r].update_state(new_coords[c])
                    active_points[r].change_color()
                    matched_active_indices.add(r)
                    matched_new_indices.add(c)

        # 2. Handle Unmatched Active Points (The ball hid or disappeared)
        for i, pt in enumerate(active_points):
            if i not in matched_active_indices:
                # Let it keep flying on its trajectory instead of freezing!
                pt.coast()
                pt.change_color((0, 0, 0))
                if pt.missed_frames > self.max_misses:
                    pt.reset_track() # Kill the track, send to pool

        # 3. Handle Unmatched New Coordinates (Brand new ball or ghost)
        for j, coord in enumerate(new_coords):
            if j not in matched_new_indices:
                free_pt = self.get_free_point()
                if free_pt is not None:
                    free_pt.is_active = True
                    free_pt.update_state(coord)
                    # Note: We do NOT show it yet. Age is only 1.

        # 4. Visibility Pass (The Ghost Filter)
        for pt in self.get_active_points():
            if pt.age >= self.min_hits:
                pt.show()
            else:
                pt.hide()