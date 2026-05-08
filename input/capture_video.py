from pseyepy import Camera, Display
import cv2
import numpy as np
import yaml
from input.analyze_video import CameraAnalysis
from input.image_processor import ImageProcessor


class CameraCapture:
    def __init__(self, cam_id=0, config=None):
        self.cam_id = cam_id
        self.config = config
        self.cam = None

        self.last_timestamp = 0.0
        self.last_frame = None

        self.analysis = CameraAnalysis(config=config)

    def start_capture(self):
        try:
            self.cam = Camera(self.cam_id, fps=60, resolution=Camera.RES_LARGE, colour=True)
            print(f"Camera {self.cam_id} initialized successfully!")
        except Exception as e:
            print(f"Error initializing camera {self.cam_id}: {e}")
            self.cam = None

    def capture_frame(self):
        """ Capture last frame and timestamp from the camera. Returns (frame, timestamp) or (None, None) on failure. """
        if self.cam is None:
            print("Camera not initialized. Call start_capture() first.")
            return None, None

        frame, timestamp = self.cam.read()
        self.last_timestamp = timestamp
        self.last_frame = frame

        if frame is None:
            return None, None

        return frame, timestamp

    def stop_capture(self):
        if self.cam is not None:
            self.cam.end()
        cv2.destroyAllWindows()

    def analyze_current_frame(self, debug=False):
        """ Analyze the last captured frame for detected centers. Returns list of (x, y) centers. """
        if self.last_frame is None:
            print("No frame captured yet. Call capture_frame() first.")
            return []

        return self.analysis.process_frame(self.last_frame, self.last_timestamp, debug=debug)
    
    def analyze_current_frame_with_adjustments(self, debug=False):
        """Analyze the last captured frame with camera adjustments applied."""
        if self.last_frame is None:
            print("No frame captured yet. Call capture_frame() first.")
            return [] if not debug else ([], None, None)
        
        return self.analysis.process_frame_auto_adjusted(self.last_frame, self.last_timestamp, debug=debug)
    
    def get_adjusted_frame(self):
        """Get the last captured frame with camera adjustments applied."""
        if self.last_frame is None:
            return None
        
        return self.analysis._apply_camera_adjustments(self.last_frame)

    def show_feed(self, show_analysis=False):
        """ Continuously capture and display the camera feed until 'q' is pressed. """
        if self.cam is None:
            print("Camera not initialized. Call start_capture() first.")
            return

        print("Starting camera feed... Press 'q' to quit.")
        while True:
            frame, timestamp = self.capture_frame()
            if frame is None:
                continue

            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if show_analysis:
                result = self.analysis.process_frame_auto_adjusted(frame, timestamp, debug=False)

                # Draw the detected centers on the frame                
                for (cx, cy) in result:
                    cv2.circle(bgr_frame, (cx, cy), 10, (0, 255, 0), -1)

            # Convert RGB to BGR for OpenCV display
            cv2.putText(bgr_frame, f"HW Time: {timestamp:.4f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Camera {self.cam_id} Feed", bgr_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_capture()

if __name__ == "__main__":
    # Load config if needed, e.g. from YAML
    with open("config/ps3eye.yaml", 'r') as f:
        config = yaml.safe_load(f)

    cam_capture = CameraCapture(cam_id=1, config=config)
    cam_capture.start_capture()
    cam_capture.show_feed(show_analysis=True)