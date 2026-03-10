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
            
            # 2. Isolate the color of interest (e.g., green)
            lower_green = np.array([40, 100, 100])
            upper_green = np.array([80, 255, 255])
            thresh = self.isolate_color(img_bgr, (lower_green, upper_green))
            
            # 4. Display the processed frame
            self.update(thresh)

    def isolate_color(self, img_rgb, color_range):
        # Convert RGB to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        # Create a mask for the specified color range
        mask = cv2.inRange(img_hsv, color_range[0], color_range[1])
        return mask
    

    def change_visibility(self, show):
        self.show_window = show
        if not show:
            self.shutdown()
        else:
            self.startWindow()