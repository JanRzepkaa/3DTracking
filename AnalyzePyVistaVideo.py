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
            
            # --- YOUR ANALYSIS HERE ---
            # Example: Simple thresholding
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 4. Display the processed frame
            self.update(thresh)

    def change_visibility(self, show):
        self.show_window = show
        if not show:
            self.shutdown()
        else:
            self.startWindow()