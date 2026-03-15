from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import numpy as np
import pyvista as pv


class VirtualPoint():
    def __init__(self, point_id):
        self.id = point_id
        
        # State variables
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
        # Tracking variables
        self.is_active = False  # Is this point currently tracking something?
        self.age = 0            # How many consecutive frames it has been tracked
        self.missed_frames = 0  # How many frames it has been missing
        
        # 1. Create the base mesh at the exact origin (0,0,0) once
        self.vista = pv.Sphere(radius=0.15)
        self.actor = None

    def add_to_plotter(self, plotter):
        """Adds to plotter but immediately hides it."""
        self.actor = plotter.add_mesh(self.vista, color="lime")
        self.hide()

    def update_state(self, new_position):
        """Teleports the point using the actor's transformation matrix."""
        new_position = np.array(new_position, dtype=np.float64)
        
        # Calculate velocity (optional, but good for Kalman filtering later)
        if self.age > 0:
            self.velocity = new_position - self.position
            
        self.position = new_position
        
        # 2. Instantly teleport the actor on the GPU
        if self.actor is not None:
            self.actor.position = tuple(self.position)
            
        # Reset miss counter and increase age
        self.missed_frames = 0
        self.age += 1

    def mark_missed(self):
        self.missed_frames += 1

    def reset_track(self):
        """Frees this point back to the pool."""
        self.is_active = False
        self.age = 0
        self.missed_frames = 0
        self.hide()

    def hide(self):
        if self.actor is not None:
            self.actor.SetVisibility(False)
    
    def show(self):
        if self.actor is not None:
            self.actor.SetVisibility(True)

class PointManager():
    def __init__(self, pool_size=20, max_distance=0.3, min_hits=5, max_misses=3):
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

    def get_free_point(self):
        for pt in self.pool:
            if not pt.is_active:
                return pt
        return None

    def update(self, new_3d_coordinates):
        """
        Takes a list of raw (X,Y,Z) tuples from the triangulator, 
        matches them, filters ghosts, and updates the PyVista screen.
        """
        new_coords = np.array(new_3d_coordinates)
        active_points = self.get_active_points()
        
        matched_new_indices = set()
        matched_active_indices = set()

        # 1. Hungarian Matching
        if len(active_points) > 0 and len(new_coords) > 0:
            # Extract positions of current active trackers
            active_coords = np.array([pt.position for pt in active_points])
            
            # Calculate distance matrix (Rows: Active, Cols: New)
            cost_matrix = cdist(active_coords, new_coords)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.max_distance:
                    # Successful match
                    active_points[r].update_state(new_coords[c])
                    matched_active_indices.add(r)
                    matched_new_indices.add(c)

        # 2. Handle Unmatched Active Points (The ball hid or disappeared)
        for i, pt in enumerate(active_points):
            if i not in matched_active_indices:
                pt.mark_missed()
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