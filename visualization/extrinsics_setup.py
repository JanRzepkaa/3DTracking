import pyvista as pv
import numpy as np
import json
import os

def create_camera_frustum(scale=100.0, aspect_ratio=4/3):
    """
    Creates a wireframe pyramid representing a camera's field of view.
    scale: How long the camera frustum is drawn in the 3D world (in mm).
    """
    z = scale
    x = scale * 0.5 * aspect_ratio
    y = scale * 0.5
    
    # Define the 6 points of the camera
    points = np.array([
        [0, 0, 0],       # Point 0: The Camera Lens (Origin)
        [-x, -y, z],     # Point 1: Top-Left
        [x, -y, z],      # Point 2: Top-Right
        [x, y, z],       # Point 3: Bottom-Right
        [-x, y, z],      # Point 4: Bottom-Left
        [0, -y*1.5, z],  # Point 5: Middle (Shows the "Up" direction indicator)
    ], dtype=np.float64)
    
    # Define the faces connecting the points
    faces = np.hstack([
        [4, 1, 2, 3, 4], # The rectangular "screen" base
        [3, 0, 1, 2],    # Top triangle
        [3, 0, 2, 3],    # Right triangle
        [3, 0, 3, 4],    # Bottom triangle
        [3, 0, 4, 1],    # Left triangle
        [3, 1, 2, 5]     # The "Up" indicator triangle
    ])
    
    return pv.PolyData(points, faces)

def main():
    filepath = "calibration/info/camera_extrinsics.json"
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run your extrinsics calculation first!")
        return

    # 1. Load the Extrinsics Data
    with open(filepath, 'r') as f:
        extrinsics_data = json.load(f)

    # 2. Initialize the PyVista Plotter
    plotter = pv.Plotter(title="Stewart Platform Camera Tracking Setup")
    plotter.set_background('white')

    # 3. Draw the Physical Calibration Rectangle (325,218)
    rect_pts = np.array([
        [0, 0, 0],          # Bottom Left
        [325, 0, 0],        # Bottom Right
        [325, 0, 218],      # Top Right
        [0, 0, 218]         # Top Left
    ], dtype=np.float64)
    
    # The first number '4' tells PyVista this face is a quad made of 4 points
    rect_faces = np.hstack([[4, 0, 1, 2, 3]])
    rectangle = pv.PolyData(rect_pts, rect_faces)
    
    plotter.add_mesh(rectangle, color='lightblue', show_edges=True, opacity=0.5)
    plotter.add_point_labels(rect_pts, ["Bottom Left", "Bottom Right", "Top Right", "Top Left"], 
                             point_size=5, font_size=12, text_color='black')

    # Colors for different cameras to tell them apart easily
    colors = ['red', 'green', 'blue', 'orange']

    # 4. Render Each Camera
    for idx, (cam_id, data) in enumerate(extrinsics_data.items()):
        color = colors[idx % len(colors)]
        
        # Extract and format the math from the JSON
        position = np.array(data["position"])
        # Reshape the flattened 9-element list back into a 3x3 rotation matrix
        rotation_matrix = np.array(data["rotation_matrix"]).reshape(3, 3)

        # Generate a standard frustum sitting at the origin (0,0,0)
        frustum = create_camera_frustum()

        # --- The Core 3D Math (Moving from Camera Space to World Space) ---
        # We multiply the local points by the transposed rotation matrix, 
        # and then add the real-world position to shift it into place.
        frustum.points = np.dot(frustum.points, rotation_matrix.T) + position

        # Add the wireframe frustum to the scene
        plotter.add_mesh(frustum, color=color, style='wireframe', line_width=3)
        
        # Add a solid sphere to represent the actual lens of the camera
        plotter.add_mesh(pv.Sphere(center=position, radius=15), color=color)
        
        # Add a text label floating above the camera
        plotter.add_point_labels([position], [f"Camera {cam_id}"], 
                                 point_size=0, font_size=14, text_color=color, shape_opacity=0.0)

    # 5. Add Reference Grids and Axes
    plotter.add_axes()
    plotter.show_grid(color='gray')
    
    print("Launching interactive 3D visualizer. Close the window to exit.")
    plotter.show()

if __name__ == "__main__":
    main()