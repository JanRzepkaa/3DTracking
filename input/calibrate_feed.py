import cv2
import numpy as np
import yaml
from pathlib import Path
from analyze_video import CameraAnalysis
from capture_video import CameraCapture
from image_processor import ImageProcessor


class CameraCalibration:
    """Interactive camera calibration tool with HSV range adjustment and real-time mask preview."""
    
    def __init__(self, cam_id=0, config_path="config/ps3eye.yaml"):
        self.cam_id = cam_id
        self.config_path = Path(config_path)
        
        # Load or create default config
        self.config = self._load_config()
        
        # Initialize camera capture
        self.camera_capture = CameraCapture(cam_id, self.config)
        self.camera_capture.start_capture()
        
        # Create analysis with config
        self.analysis = CameraAnalysis(cam_id, self.config)
        
        # Window configuration
        self.window_name = f"Camera {cam_id} Calibration"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Trackbar callback (needs to be a lambda since cv2 trackbars don't support instance methods directly)
        self._create_trackbars()
        
        print(f"Camera Calibration initialized for camera {cam_id}")
        print("Controls:")
        print("  CAMERA ADJUSTMENTS: Brightness, Contrast, Saturation")
        print("  COLOR DETECTION: HSV ranges (Lower/Upper H, S, V)")
        print("  MORPHOLOGY: Erode/Dilate iterations")
        print("  DETECTION: Min area threshold")
        print("Keyboard:")
        print("  's' - Save current config")
        print("  'r' - Reset to default values")
        print("  'q' - Quit calibration")
    
    def _load_config(self):
        """Load config from YAML file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or self._get_default_config()
        return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration."""
        return {
            'camera': {
                'brightness': 0,
                'contrast': 100,      # 50-200 range mapped to 0.5-2.0
                'saturation': 100,    # 0-200 range mapped to 0.0-2.0
                'exposure': 0,
                'gain': 0
            },
            'hsv': {
                'lower': {'h': 10, 's': 100, 'v': 100},
                'upper': {'h': 25, 's': 255, 'v': 255}
            },
            'morphology': {
                'erode_iterations': 1,
                'dilate_iterations': 2,
                'kernel_size': 5
            },
            'detection': {
                'min_contour_area': 50
            }
        }
    
    def _create_trackbars(self):
        """Create trackbars for all adjustments."""
        # Camera adjustments (brightness, contrast, saturation)
        cv2.createTrackbar('Brightness', self.window_name, 
                          int(self.config['camera']['brightness']), 100, 
                          lambda x: self._update_config('brightness', x))
        cv2.createTrackbar('Contrast', self.window_name, 
                          int(self.config['camera']['contrast']), 200, 
                          lambda x: self._update_config('contrast', x))
        cv2.createTrackbar('Saturation', self.window_name, 
                          int(self.config['camera']['saturation']), 200, 
                          lambda x: self._update_config('saturation', x))
        
        # Lower HSV bounds
        cv2.createTrackbar('Lower H', self.window_name, 
                          int(self.config['hsv']['lower']['h']), 180, 
                          lambda x: self._update_config('lower_h', x))
        cv2.createTrackbar('Lower S', self.window_name, 
                          int(self.config['hsv']['lower']['s']), 255, 
                          lambda x: self._update_config('lower_s', x))
        cv2.createTrackbar('Lower V', self.window_name, 
                          int(self.config['hsv']['lower']['v']), 255, 
                          lambda x: self._update_config('lower_v', x))
        
        # Upper HSV bounds
        cv2.createTrackbar('Upper H', self.window_name, 
                          int(self.config['hsv']['upper']['h']), 180, 
                          lambda x: self._update_config('upper_h', x))
        cv2.createTrackbar('Upper S', self.window_name, 
                          int(self.config['hsv']['upper']['s']), 255, 
                          lambda x: self._update_config('upper_s', x))
        cv2.createTrackbar('Upper V', self.window_name, 
                          int(self.config['hsv']['upper']['v']), 255, 
                          lambda x: self._update_config('upper_v', x))
        
        # Morphological operations
        cv2.createTrackbar('Erode Iter', self.window_name, 
                          int(self.config['morphology']['erode_iterations']), 5, 
                          lambda x: self._update_config('erode_iterations', x))
        cv2.createTrackbar('Dilate Iter', self.window_name, 
                          int(self.config['morphology']['dilate_iterations']), 5, 
                          lambda x: self._update_config('dilate_iterations', x))
        
        # Minimum contour area
        cv2.createTrackbar('Min Area', self.window_name, 
                          int(self.config['detection']['min_contour_area']), 500, 
                          lambda x: self._update_config('min_contour_area', x))
    
    def _update_config(self, key, value):
        """Update config based on trackbar input."""
        # Camera adjustments
        if key == 'brightness':
            self.config['camera']['brightness'] = value
        elif key == 'contrast':
            self.config['camera']['contrast'] = value
        elif key == 'saturation':
            self.config['camera']['saturation'] = value
        # HSV adjustments
        elif key == 'lower_h':
            self.config['hsv']['lower']['h'] = value
        elif key == 'lower_s':
            self.config['hsv']['lower']['s'] = value
        elif key == 'lower_v':
            self.config['hsv']['lower']['v'] = value
        elif key == 'upper_h':
            self.config['hsv']['upper']['h'] = value
        elif key == 'upper_s':
            self.config['hsv']['upper']['s'] = value
        elif key == 'upper_v':
            self.config['hsv']['upper']['v'] = value
        # Morphology adjustments
        elif key == 'erode_iterations':
            self.config['morphology']['erode_iterations'] = value
        elif key == 'dilate_iterations':
            self.config['morphology']['dilate_iterations'] = value
        # Detection adjustments
        elif key == 'min_contour_area':
            self.config['detection']['min_contour_area'] = value
        
        # Update the analysis object with new config
        self.analysis.update_config(self.config)
    
    def _create_side_by_side(self, frame, mask, centers, adjusted_frame):
        """Create side-by-side display of original frame, adjusted frame and mask."""
        # Convert mask to BGR for display
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Draw detected centers on adjusted frame
        frame_with_centers = adjusted_frame.copy()
        for (cx, cy) in centers:
            cv2.circle(frame_with_centers, (cx, cy), 10, (0, 255, 0), -1)
            cv2.circle(frame_with_centers, (cx, cy), 12, (0, 255, 0), 2)
        
        # Resize to match dimensions
        h, w = frame_with_centers.shape[:2]
        mask_bgr = cv2.resize(mask_bgr, (w, h))
        
        # Create side-by-side (adjusted frame + mask)
        side_by_side = np.hstack([frame_with_centers, mask_bgr])
        
        # Add labels
        cv2.putText(side_by_side, "Adjusted Frame", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(side_by_side, "Color Mask", (w + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add camera adjustment info
        camera = self.config['camera']
        camera_info = (f"Brightness: {camera['brightness']:+3d} | "
                      f"Contrast: {camera['contrast']/100:.2f}x | "
                      f"Saturation: {camera['saturation']/100:.2f}x")
        cv2.putText(side_by_side, camera_info, (20, h - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add color range info
        info_text = (f"HSV Range: H[{self.config['hsv']['lower']['h']}-{self.config['hsv']['upper']['h']}] "
                    f"S[{self.config['hsv']['lower']['s']}-{self.config['hsv']['upper']['s']}] "
                    f"V[{self.config['hsv']['lower']['v']}-{self.config['hsv']['upper']['v']}]")
        cv2.putText(side_by_side, info_text, (20, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add detection info
        cv2.putText(side_by_side, f"Detected: {len(centers)} objects | "
                   f"Morphology: E{self.config['morphology']['erode_iterations']} "
                   f"D{self.config['morphology']['dilate_iterations']} | "
                   f"Min Area: {self.config['detection']['min_contour_area']}", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return side_by_side
    
    def run(self):
        """Run the calibration loop."""
        print("Starting calibration loop... Press 'q' to quit.")
        
        while True:
            frame, timestamp = self.camera_capture.capture_frame()
            
            if frame is None:
                continue
            
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Apply camera adjustments (brightness, contrast, saturation)
            adjusted_frame = self._apply_camera_adjustments(frame_bgr)
            
            # Process frame to get mask and centers (using adjusted frame for detection)
            centers, processed_mask = self.analysis.process_frame_with_mask(adjusted_frame, timestamp)
            
            # Create side-by-side display
            display = self._create_side_by_side(frame_bgr, processed_mask, centers, adjusted_frame)
            
            # Show the display
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting calibration...")
                break
            elif key == ord('s'):
                self.save_config()
                print(f"Config saved to {self.config_path}")
            elif key == ord('r'):
                self.config = self._get_default_config()
                self.analysis.update_config(self.config)
                # Recreate trackbars with default values
                cv2.destroyWindow(self.window_name)
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._create_trackbars()
                print("Reset to default config")
        
        self.cleanup()
    
    def _apply_camera_adjustments(self, frame):
        """Apply camera adjustments (brightness, contrast, saturation) to frame."""
        camera = self.config['camera']
        
        adjusted = frame.copy()
        
        # Apply brightness adjustment
        if camera['brightness'] != 0:
            adjusted = ImageProcessor.adjust_brightness(adjusted, camera['brightness'])
        
        # Apply contrast adjustment (map 0-200 range to 0.5-2.0)
        contrast_value = camera['contrast'] / 100.0
        if contrast_value != 1.0:
            adjusted = ImageProcessor.adjust_contrast(adjusted, contrast_value)
        
        # Apply saturation adjustment (map 0-200 range to 0.0-2.0)
        saturation_value = camera['saturation'] / 100.0
        if saturation_value != 1.0:
            adjusted = ImageProcessor.adjust_saturation(adjusted, saturation_value)
        
        return adjusted
    
    def save_config(self):
        """Save current config to YAML file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def cleanup(self):
        """Clean up resources."""
        self.camera_capture.stop_capture()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibration = CameraCalibration(cam_id=0, config_path="config/ps3eye.yaml")
    calibration.run()
