import cv2
import numpy as np
from Calibration import *

class AnalyzePyVistaVideo:
    def __init__(self, show_window=True):
        self.window_name = "PyVista Video Analysis"
        self.show_window = show_window
        self.last_frame = None

    def initialaze_calibration_test_data(self, ball_position, camera_position, camera_rotation, camera_intrinsics):
        self.ball_position = ball_position
        self.camera_position = camera_position
        self.camera_rotation = camera_rotation
        self.camera_intrinsics = camera_intrinsics

    def startWindow(self):
        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def update(self, frame):
        if self.show_window:
            cv2.imshow(self.window_name, frame)
    
    def shutdown(self):
        cv2.destroyAllWindows()

    def update_from_pyvista_screenshot(self, screenshot):
        if screenshot is not None:
            # 3. Convert RGB to BGR for OpenCV (Fast NumPy operation)
            img_bgr = screenshot[:, :, ::-1]
            self.last_frame = img_bgr.copy()  # Store the last frame for later use

            new_frame = img_bgr.copy() # Create a copy for drawingx 
            
            contours, centers = self.find_centroids_and_contours(new_frame)
            self.draw_contours(new_frame, contours)
            self.draw_all_ceneters(new_frame, centers)
            
            # 4. Display the processed frame
            self.update(new_frame)

    def find_centroids_and_contours(self, frame, color="lime"):
        # 2. Isolate the color of interest (e.g., green)
        if color == "lime":
            lower = np.array([40, 100, 100])
            upper = np.array([80, 255, 255])
        if color == "blue":
            lower = np.array([100, 100, 100])
            upper = np.array([130, 255, 255])
        thresh = self.isolate_color(frame, (lower, upper))

        # Find contours and centroids
        contours = self.find_contours(thresh)
        centers = self.find_all_ceneters(contours)
        return contours, centers
        
    def isolate_color(self, img_bgr, color_range):
        # Convert BGR to HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Create a mask for the specified color range
        mask = cv2.inRange(img_hsv, color_range[0], color_range[1])
        return mask
    
    def find_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def draw_contours(self, frame, contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)  # Draw blue contours

    def find_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None
        
    def find_all_ceneters(self, contours):
        centers = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small contours
                centroid = self.find_centroid(contour)
                if centroid is not None:
                    centers.append(centroid)
        return centers
    
    def draw_all_ceneters(self, frame, centers):
        for center in centers:
            cv2.circle(frame, center, 10, (0, 0, 255), -1)  # Draw red circle at centroid
            self.write_pixel_coordinates(frame, center)
            #self.write_simulated_coordinates(frame, center)  # Placeholder for simulated coordinates

    def write_pixel_coordinates(self, frame, centroid):
        text = f"({centroid[0]}, {centroid[1]})"
        cv2.putText(frame, text, (centroid[0] + 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def write_simulated_coordinates(self, frame, coordinates):
        sim_coridinates, norm = projectPoints(
            ball_position=self.ball_position,
            camera_position=self.camera_position,
            camera_rotation=self.camera_rotation,
            camera_intrinsics=self.camera_intrinsics
        )
    
        text = f"Sim({sim_coridinates[0]:.0f}, {sim_coridinates[1]:.0f})"
        text2 = f"Norm({norm[0]:.0f}, {norm[1]:.0f})"
        cv2.putText(frame, text, (coordinates[0] + 10, coordinates[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, text2, (coordinates[0] + 10, coordinates[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def change_visibility(self, show):
        self.show_window = show
        if not show:
            self.shutdown()
        else:
            self.startWindow()

    def find_camera_position_and_rotation_from_3_fixed_balls(self, ball_positions):
        """
        Given the 3D positions of 3 fixed balls in the world and their corresponding 2D pixel coordinates in the camera view,
        find the position and rotation of the camera.
        """

        video_positions = []

        _, centers = self.find_centroids_and_contours(self.last_frame)
        video_positions.extend(centers)

        solved_pnp, best_combo, best_error = find_camera_position_and_rotation_from_3_fixed_balls(
            true_positions=ball_positions,
            video_positions=video_positions,
            camera_intrinsics=self.camera_intrinsics)
        
        if solved_pnp is None:
            print("Could not solve PnP with the given points.")
            return None
        
        camera_position, camera_rotation = rvec_tvec_to_camera_pose(solved_pnp[0], solved_pnp[1])
        print(f"Best combo: {best_combo}, Reprojection error: {best_error}")

        return camera_position, camera_rotation