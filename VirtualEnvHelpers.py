import numpy as np
import pyvista as pv

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


def pyvista_to_opencv_rotation(pos, focal_pt, up_vector):
    """
    Converts PyVista camera parameters to an OpenCV 3x3 Rotation Matrix (R).
    
    Parameters:
        pos: (3,) array, Camera Position
        focal_pt: (3,) array, Focal Point
        up_vector: (3,) array, Up Vector
        
    Returns:
        R: (3, 3) array, World-to-Camera Rotation Matrix
    """
    # 1. Calculate the Forward Vector (+Z)
    # Points from camera to focal point
    f = np.array(focal_pt, dtype=np.float64) - np.array(pos, dtype=np.float64)
    f /= np.linalg.norm(f)
    
    # 2. Calculate the Right Vector (+X)
    # Cross product of Forward and PyVista Up
    # PyVista 'Up' is typically World Up (0,0,1)
    r = np.cross(f, np.array(up_vector))
    r /= np.linalg.norm(r)
    
    # 3. Calculate the Down Vector (+Y)
    # In OpenCV, Y is down. Cross Forward and Right to get it.
    d = np.cross(f, r)
    # No need to normalize here if f and r are unit vectors and orthogonal
    
    # 4. Construct the Camera-to-World Matrix (R_inv)
    # These vectors are the columns of the transformation matrix
    R_world_to_cam_T = np.column_stack((r, d, f))
    
    # 5. Return the Transpose to get the World-to-Camera Matrix (R)
    # Because it is an orthogonal matrix, transpose == inverse
    R = R_world_to_cam_T
    
    return R

def triangulate_n_rays(origins, directions):
    """
    Finds the 3D point closest to all given rays using Least Squares.
    
    Parameters:
        origins: A list or (N, 3) array of camera positions.
        directions: A list or (N, 3) array of ray direction vectors.
        
    Returns:
        P: (3,) numpy array representing the estimated (X, Y, Z) ball position.
    """
    origins = np.array(origins, dtype=np.float64)
    directions = np.array(directions, dtype=np.float64)
    
    # 1. Initialize the A matrix and b vector
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    
    # 2. Accumulate the matrices for each ray
    for o, d in zip(origins, directions):
        # Ensure direction is normalized (length of 1)
        d = d / np.linalg.norm(d)
        
        # Reshape to a 3x1 column vector for the outer product
        d_col = d.reshape(3, 1)
        
        # M = I - (d * d.T)
        M = I - (d_col @ d_col.T)
        
        # Add to our global A and b
        A += M
        b += M @ o # Matrix multiplication of M and origin
        
    # 3. Solve the linear system A * P = b
    # np.linalg.lstsq is used instead of np.linalg.solve because it perfectly 
    # handles numerical instabilities if the rays are nearly parallel.
    P, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    return P

def calculate_ray_to_ray_distance(o1, d1, o2, d2):
    """
    Calculates the shortest 3D distance between two rays.
    """
    w = o1 - o2
    # Cross product of the two direction vectors
    cross_dir = np.cross(d1, d2)
    cross_norm = np.linalg.norm(cross_dir)
    
    # If lines are exactly parallel (rare in real tracking, but possible)
    if cross_norm < 1e-6:
        return np.linalg.norm(np.cross(w, d1)) / np.linalg.norm(d1)
        
    # Shortest distance is the projection of w onto the cross product
    distance = np.abs(np.dot(w, cross_dir)) / cross_norm
    return distance