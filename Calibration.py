import numpy as np


def find_camera_position_and_rotation_from_3_fixed_balls(true_positions, video_positions, camera_intrinsics):
    """
        Find position and rotation of the camera from 3 fixed balls. 
        We know the true positions of the balls, and we can find their positions in the video.
        We can then use this information to find the position of the camera.
    """

    pass

def rotate_vector(vector, rotation):
    """
        Rotate a vector by a given rotation (euler angles).
        This is used to simulate the camera view of a ball given the camera rotation.

        Parameters:
        vector: (x, y, z) vector to rotate
        rotation: (theta_x, theta_y, theta_z) euler angles representing the rotation in radians
    """

    # Create rotation matrices for each axis
    theta_x, theta_y, theta_z = rotation

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    # Rotate the vector
    rotated_vector = R @ vector

    return rotated_vector

def find_euler_angles_between_vectors(v1, v2):
    """
        Find the euler angles between two vectors. 
        This is used to find the rotation needed to align one vector with another.

        Parameters:
            v1: (x, y, z) first vector
            v2: (x, y, z) second vector
        Returns:
            (theta_x, theta_y, theta_z) euler angles representing the rotation in radians
    """
    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate the cross product and dot product
    cross_prod = np.cross(v1_norm, v2_norm)
    dot_prod = np.dot(v1_norm, v2_norm)

    # Calculate the angle between the vectors
    angle = np.arccos(dot_prod)

    # Calculate the rotation axis
    if np.linalg.norm(cross_prod) < 1e-6:
        # Vectors are parallel, no rotation needed
        return (0, 0, 0)
    
    rotation_axis = cross_prod / np.linalg.norm(cross_prod)

    # Convert the rotation axis and angle to euler angles
    theta_x = rotation_axis[0] * angle
    theta_y = rotation_axis[1] * angle
    theta_z = rotation_axis[2] * angle

    return (theta_x, theta_y, theta_z)

def simulate_camera(ball_position, camera_position, camera_rotation, camera_intrinsics):
    """
        Simulate the camera view of a ball given the ball position, camera position and rotation.
        This is used to test the find_camera_position_and_rotation_from_3_fixed_balls function.

        Parameters:
            ball_position: (x, y, z) position of the ball in world coordinates
            camera_position: (x, y, z) position of the camera in world coordinates
            camera_rotation: (theta_x, theta_y, theta_z) euler angles representing the rotation of the camera
            camera_intrinsics: (fx, fy, cx, cy) camera intrinsics parameters

        Returns:
            (x, y) position of the ball in the camera's image plane
    """

    # Create a 3D vector from camera to center of the ball
    camera_to_ball = ball_position - camera_position
    
    camera_to_ball_rotated = rotate_vector(camera_to_ball, camera_rotation)
    print("Camera to ball vector after rotation:", camera_to_ball_rotated)

    normalized_xy = camera_to_ball_rotated[:2] / camera_to_ball_rotated[2]
    print("Normalized image plane coordinates:", normalized_xy)

    fx, fy, cx, cy = camera_intrinsics
    pixel_x = fx * normalized_xy[0] + cx
    pixel_y = fy * normalized_xy[1] + cy

    return (pixel_x, pixel_y)


if __name__ == "__main__":
    # Test func simulate_camera
    ball_position = np.array([1, 2, 1])
    camera_position = np.array([0, 0, 0])
    camera_rotation = (0, 0, 0) # Rotate 90 degrees around z-axis
    camera_intrinsics = (1000, 1000, 640, 480) # Example camera intrinsics
    pixel_x, pixel_y = simulate_camera(ball_position, camera_position, camera_rotation, camera_intrinsics)
    print("Pixel coordinates:", (pixel_x, pixel_y))