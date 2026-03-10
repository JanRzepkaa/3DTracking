import cv2
import numpy as np

class AnalyzePyVistaVideo:
    def __init__(self, show_window=True):
        self.window_name = "PyVista Video Analysis"
        self.show_window = show_window

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

            new_frame = img_bgr.copy() # Create a copy for drawing
            
            self.find_and_draw_centroids(new_frame)
            
            # 4. Display the processed frame
            self.update(new_frame)

    def find_and_draw_centroids(self, frame):
        # 2. Isolate the color of interest (e.g., green)
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        thresh = self.isolate_color(frame, (lower_green, upper_green))

        # Find contours and centroids
        contours = self.find_contours(thresh)
        self.draw_contours(frame, contours)
        self.draw_all_ceneters(frame, contours)


    def isolate_color(self, img_bgr, color_range):
        # Convert BGR to HSV
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # Create a mask for the specified color range
        mask = cv2.inRange(img_hsv, color_range[0], color_range[1])
        return mask
    
    def find_contours(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def find_centroid(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            return None
        
    def draw_all_ceneters(self, frame, contours):
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                centroid = self.find_centroid(contour)
                if centroid is not None:
                    cv2.circle(frame, centroid, 10, (0, 0, 255), -1)  # Draw green circle at centroid

    def draw_contours(self, frame, contours):
        cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)  # Draw blue contours

    def change_visibility(self, show):
        self.show_window = show
        if not show:
            self.shutdown()
        else:
            self.startWindow()