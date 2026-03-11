import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
from itertools import permutations

def intrictics_to_matrix(camera_intrinsics):
    fx, fy, cx, cy = camera_intrinsics
    camera_matrix = np.array([
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
    ])
    return camera_matrix

def solve_pnp(object_points, image_points, camera_matrix):
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
    if not success:
        raise ValueError("Could not solve PnP")
    return rvec.flatten(), tvec.flatten()

def find_camera_position_and_rotation_from_3_fixed_balls(true_positions, video_positions, camera_intrinsics):
    """
        Find position and rotation of the camera from 3 fixed balls. 
        We know the true positions of the balls, and we can find their positions in the video.
        We can then use this information to find the position of the camera.
    """

    object_points = np.array(true_positions, dtype=np.float32)

    image_points = np.array(video_positions, dtype=np.float32)

    assert object_points.shape[0] == image_points.shape[0], "Number of object points and image points must be the same"

    #print("Object points:\n", object_points)
    #print("Image points:\n", image_points)

    N = object_points.shape[0]
    
    camera_matrix = intrictics_to_matrix(camera_intrinsics)

    # Run through all combinations of 3 balls
    best_error = float('inf')
    best_parameters = None
    best_combo = None

    for combo in permutations(range(N)):
        try:
            rvec, tvec = solve_pnp(object_points[list(range(N))], image_points[list(combo)], camera_matrix)

            if check_if_we_can_reject_solution(rvec, tvec):
                #print(f"Combination {combo} rejected based on constraints.")
                continue

            # Project all object points using the estimated parameters
            projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distCoeffs=None)
            projected_points = projected_points.reshape(-1, 2)

            # Calculate reprojection Mean Squared Error
            error = np.mean(np.linalg.norm(projected_points - image_points, axis=1))

            if error < best_error:
                best_error = error
                best_parameters = (rvec, tvec)
                best_combo = combo

            #print(rvec_tvec_to_camera_pose(rvec, tvec)[0], error)
        except Exception as e:
            print(f"Combination {combo} failed: {e}")
            continue
    

    return best_parameters, best_combo, best_error

def solve_global_camera_ransac(all_object_points, all_image_points, camera_matrix):
    """
    Solves for the camera using all frames at once, ignoring wrongly sorted points.
    all_object_points: shape (N, 3) 
    all_image_points: shape (N, 2)
    """
    all_object_points = np.array(all_object_points, dtype=np.float32)

    all_image_points = np.array(all_image_points, dtype=np.float32)


    dist_coeffs = np.zeros((4,1))
    
    # 1. RANSAC Parameters
    # How many pixels a projection can be off by and still be considered "correct"
    reprojection_threshold = 3.0 
    
    # How many random 4-point combinations to test. 
    # Since 95% of your data is good, 100 iterations is more than enough to find the truth.
    confidence = 0.99
    iterations = 2000
    
    # 2. Run solvePnPRansac
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        all_object_points, 
        all_image_points, 
        camera_matrix, 
        dist_coeffs,
        iterationsCount=iterations,
        reprojectionError=reprojection_threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_SQPNP # It still uses SQPNP under the hood for the math
    )
    
    if not success:
        raise ValueError("RANSAC could not find a consensus among the points.")
        
    return rvec.flatten(), tvec.flatten(), inliers

def check_if_we_can_reject_solution(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)

    r_31 = R[2, 0] 
    r_32 = R[2, 1]
    # 3. Apply the Zero-Roll Constraint
    # We use a small tolerance (e.g., 0.1) instead of exactly 0 to account 
    # for floating-point inaccuracies during the optimization process.
    # An r_31 of 0.1 equates to about 5.7 degrees of roll.
    roll_tolerance = 0.1
    if abs(r_31) > roll_tolerance:
        return True # Reject: Camera is tilted sideways
    
    # 4. Apply the "Not Upside Down" Constraint
    if r_32 > 0:
        return True # Reject: Camera is upside down
    return False


def rvec_tvec_to_camera_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    return camera_position, R.T

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

def euler_to_rvec_cv2(theta_x, theta_y, theta_z):
    """
    Converts Euler angles (in radians) to an rvec using pure NumPy/OpenCV.
    Assumes rotation order X -> Y -> Z.
    """
    # 1. Calculate rotation matrices for each axis
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta_x), -math.sin(theta_x)],
                    [0, math.sin(theta_x), math.cos(theta_x)]])
                    
    R_y = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
                    [0, 1, 0],
                    [-math.sin(theta_y), 0, math.cos(theta_y)]])
                    
    R_z = np.array([[math.cos(theta_z), -math.sin(theta_z), 0],
                    [math.sin(theta_z), math.cos(theta_z), 0],
                    [0, 0, 1]])
                    
    # 2. Multiply them together to get the final 3x3 rotation matrix
    # Order matters here! R = R_z * R_y * R_x
    R_combined = np.dot(R_z, np.dot(R_y, R_x))
    
    # 3. Convert the 3x3 matrix to a 3x1 rvec using Rodrigues
    rvec, _ = cv2.Rodrigues(R_combined)
    
    return rvec

def projectPoints(ball_position, camera_position, camera_rotation, camera_intrinsics):
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

    ball_pos = np.array(ball_position, dtype=np.float64)
    cam_pos = np.array(camera_position, dtype=np.float64)

    # 1. Get rvec and extract the 3x3 Rotation Matrix (R)
    rvec = euler_to_rvec_cv2(*camera_rotation)
    R, _ = cv2.Rodrigues(rvec)

    # 2. CORRECT TVEC CALCULATION
    # tvec = -R * C
    tvec = -np.dot(R, cam_pos)

    # 3. Setup Camera Matrix
    fx, fy, cx, cy = camera_intrinsics
    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float64)

    # --- METHOD A: OPENCV projectPoints ---
    true_pts = np.array([ball_pos])
    res, _ = cv2.projectPoints(true_pts, rvec, tvec, camera_matrix, distCoeffs=None)
    cv2_pixel_coords = res[0].flatten()

    # Filp y coordinate to match OpenCV's convention (if needed)
    cv2_pixel_coords[1] = camera_matrix[1, 2] - (cv2_pixel_coords[1] - camera_matrix[1, 2])

    # --- METHOD B: MANUAL MATH ---
    # Create a 3D vector from camera to center of the ball
    camera_to_ball = ball_position - camera_position
    
    camera_to_ball_rotated = rotate_vector(camera_to_ball, -np.array(camera_rotation))
    X, Y, Z = camera_to_ball_rotated
    
    if Z <= 0:
        return None, None
 
    # 4. Perspective projection
    x = - X / Z
    y = - Y / Z

    # 5. Apply camera intrinsics
    pixel_x = fx * x + cx
    pixel_y = fy * y + cy

    # Currently 0, 0 

    return (pixel_x, pixel_y), cv2_pixel_coords


if __name__ == "__main__":
    # Test func simulate_camera
    ball_position = np.array([1, 2, 1])
    camera_position = np.array([0, 0, 0])
    camera_rotation = (0, 0, 0) # Rotate 90 degrees around z-axis
    camera_intrinsics = (1000, 1000, 640, 480) # Example camera intrinsics
    pixel_x, pixel_y = projectPoints(ball_position, camera_position, camera_rotation, camera_intrinsics)
    print("Pixel coordinates:", (pixel_x, pixel_y))