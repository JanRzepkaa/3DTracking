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

def simulate_camera(ball_position, camera_position, camera_rotation):
    """
        Simulate the camera view of a ball given the ball position, camera position and rotation.
        This is used to test the find_camera_position_and_rotation_from_3_fixed_balls function.

        Parameters:
        ball_position: (x, y, z) position of the ball in world coordinates
        camera_position: (x, y, z) position of the camera in world coordinates
        camera_rotation: (theta_x, theta_y, theta_z) euler angles representing the rotation of the camera
    """

    # Create a 3D vector from camera to center of the ball
    vector_to_ball = ball_position - camera_position
    
    # Create a 3D vector perpendicular to the camera's forward direction (which is the camera rotation)
    camera_forward = np.array([0, 0, 1]) # Assuming camera looks along the positive z-axis

if __name__ == "__main__":
    # Test func rotate_vector
    vector = np.array([1, 0, 0])
    rotation = (np.pi/2, np.pi/2, np.pi) # Rotate 90 degrees around z-axis
    rotated_vector = rotate_vector(vector, rotation)
    print("Original vector:", vector)
    print("Rotated vector:", rotated_vector)