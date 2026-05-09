import json
import numpy as np
import os
from Calibration import solve_global_camera_ransac, intrictics_to_matrix, rvec_tvec_to_camera_pose

def main():
    # 1. Define the True 3D Physical Points (in millimeters)
    # The user defined a 400x200 standing rectangle. 
    # Let's assume it stands on the ground plane (Z=0) and the height extends upwards in the Y direction.
    basic_rectangle_size = (325,218)
    w, h = basic_rectangle_size
    
    # These names MUST match the targets you clicked in the calibration GUI
    physical_points_3d = {
        "Bottom Left":  [0, 0, 0],
        "Bottom Right": [w, 0, 0],
        "Top Right":    [w, 0, h],
        "Top Left":     [0, 0, h]
    }
    
    # 2. Define the Estimated PS3 Eye Intrinsics
    # (Assuming 640x480 resolution, Blue Dot / Narrow FOV lens setting)
    estimated_intrinsics = (600.0, 600.0, 320.0, 240.0)
    cam_matrix = intrictics_to_matrix(estimated_intrinsics)
    
    print(f"Using Estimated Camera Matrix:\n{cam_matrix}\n")

    # 3. Load the 2D Clicked Pixel Data
    input_file = "calibration/calibration_data.json"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run your GUI calibration first!")
        return
        
    with open(input_file, 'r') as f:
        calibration_data = json.load(f)

    # 4. Process each camera
    final_extrinsics = {}
    
    for cam_id_str, clicked_points in calibration_data.items():
        print(f"--- Processing Camera {cam_id_str} ---")
        
        object_points = []
        image_points = []
        
        # Match the 3D physical coordinates to the 2D clicked pixels
        for target_name, point_3d in physical_points_3d.items():
            if target_name in clicked_points:
                coords_2d = clicked_points[target_name]
                object_points.append(point_3d)
                image_points.append([coords_2d["x"], coords_2d["y"]])
            else:
                print(f"Warning: '{target_name}' was not clicked for Camera {cam_id_str}.")

        # We need at least 4 points to solve PnP securely
        if len(object_points) < 4:
            print(f"Skipping Camera {cam_id_str}: Not enough points (found {len(object_points)}, need 4).")
            continue

        try:
            # 5. Run the Math (Using your RANSAC wrapper)
            rvec, tvec, inliers = solve_global_camera_ransac(
                all_object_points=object_points,
                all_image_points=image_points,
                camera_matrix=cam_matrix
            )
            
            # 6. Convert rvec/tvec into human-readable Position and Rotation
            cam_position, cam_rotation_matrix = rvec_tvec_to_camera_pose(rvec, tvec)
            
            print(f"Success! Camera {cam_id_str} solved with {len(inliers)} inliers.")
            print(f" -> 3D Position (X, Y, Z in mm): [{cam_position[0]:.1f}, {cam_position[1]:.1f}, {cam_position[2]:.1f}]")
            
            # Save the results to our output dictionary
            final_extrinsics[cam_id_str] = {
                "position": cam_position.tolist(),
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                # Flatten the rotation matrix so it saves cleanly to JSON
                "rotation_matrix": cam_rotation_matrix.flatten().tolist() 
            }
            
        except Exception as e:
            print(f"Failed to solve for Camera {cam_id_str}. Error: {e}")

    # 7. Save the final calculated Extrinsics for the RL Engine to use
    output_file = "calibration/camera_extrinsics.json"
    with open(output_file, 'w') as f:
        json.dump(final_extrinsics, f, indent=4)
        
    print(f"\nExtrinsics successfully saved to {os.path.abspath(output_file)}")
    print("Your RL model can now load this file to triangulate 3D space!")

if __name__ == "__main__":
    main()