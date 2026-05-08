import cv2
import numpy as np
from input.image_processor import ImageProcessor

class CameraAnalysis:
    def __init__(self, cam_id, config=None):
        self.cam_id = cam_id
        self.last_known_centers = []
        self.last_timestamp = 0.0
        
        # Load configuration
        if config is None:
            config = self._get_default_config()
        
        self.config = config
        self._update_from_config()
    
    def _get_default_config(self):
        """Return default configuration."""
        return {
            'camera': {
                'brightness': 0,
                'contrast': 100,
                'saturation': 100,
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
    
    def _update_from_config(self):
        """Update internal parameters from config dictionary."""
        hsv = self.config.get('hsv', {})
        lower = hsv.get('lower', {})
        upper = hsv.get('upper', {})
        
        self.lower_orange = np.array([
            lower.get('h', 10),
            lower.get('s', 100),
            lower.get('v', 100)
        ])
        self.upper_orange = np.array([
            upper.get('h', 25),
            upper.get('s', 255),
            upper.get('v', 255)
        ])
        
        morph = self.config.get('morphology', {})
        self.erode_iterations = morph.get('erode_iterations', 1)
        self.dilate_iterations = morph.get('dilate_iterations', 2)
        self.kernel_size = morph.get('kernel_size', 5)
        
        detection = self.config.get('detection', {})
        self.min_contour_area = detection.get('min_contour_area', 50)
    
    def update_config(self, config):
        """Update configuration and refresh internal parameters."""
        self.config = config
        self._update_from_config() 

    def process_frame(self, frame, timestamp, debug=False):
        """
        Takes a raw BGR frame and returns a list of (x, y) centers for the detected objects.
        """
        self.last_timestamp = timestamp
        
        # 1. Convert to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Create a binary mask (White = Target color, Black = Everything else)
        mask = cv2.inRange(hsv_frame, self.lower_orange, self.upper_orange)
        
        # 3. Clean the mask (Morphological Operations)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)
        
        # 4. Find the contours (the outlines of the white blobs)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centers = []
        
        # 5. Extract the mathematical center of each valid contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > self.min_contour_area:
                # Calculate Image Moments (Physics concept applied to pixels to find center of mass)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centers.append((cx, cy))
                    
                    # Optional: Draw on the frame for debugging
                    if debug:
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.drawContours(frame, [cnt], -1, (255, 0, 0), 2)

        # Update state
        self.last_known_centers = current_centers
        
        if debug:
            return current_centers, frame, mask
            
        return current_centers
    
    def process_frame_with_mask(self, frame, timestamp):
        """
        Takes a raw BGR frame and returns both centers and the mask.
        Useful for calibration and visualization.
        """
        self.last_timestamp = timestamp
        
        # 1. Convert to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. Create a binary mask
        mask = cv2.inRange(hsv_frame, self.lower_orange, self.upper_orange)
        
        # 3. Clean the mask (Morphological Operations)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        mask = cv2.erode(mask, kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iterations)
        
        # 4. Find the contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centers = []
        
        # 5. Extract the center of each valid contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area > self.min_contour_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centers.append((cx, cy))

        # Update state
        self.last_known_centers = current_centers
        
        return current_centers, mask
    
    def _apply_camera_adjustments(self, frame):
        """Apply camera adjustments (brightness, contrast, saturation) to frame."""
        camera = self.config.get('camera', {})
        
        adjusted = frame.copy()
        
        brightness = camera.get('brightness', 0)
        if brightness != 0:
            adjusted = ImageProcessor.adjust_brightness(adjusted, brightness)
        
        # Map 0-200 range to 0.5-2.0
        contrast_value = camera.get('contrast', 100) / 100.0
        if contrast_value != 1.0:
            adjusted = ImageProcessor.adjust_contrast(adjusted, contrast_value)
        
        # Map 0-200 range to 0.0-2.0
        saturation_value = camera.get('saturation', 100) / 100.0
        if saturation_value != 1.0:
            adjusted = ImageProcessor.adjust_saturation(adjusted, saturation_value)
        
        return adjusted
    
    def process_frame_auto_adjusted(self, frame, timestamp, debug=False):
        """
        Process frame with automatic camera adjustments applied.
        Takes raw BGR frame and returns detected centers.
        
        Args:
            frame: Input BGR frame
            timestamp: Frame timestamp
            debug: Return additional debug info (centers, frame, mask)
        
        Returns:
            If debug: (centers, frame, mask)
            Else: centers list
        """
        # Apply camera adjustments first
        adjusted_frame = self._apply_camera_adjustments(frame)
        
        # Then process with color detection
        return self.process_frame(adjusted_frame, timestamp, debug=debug)
    
    def process_frame_with_mask_auto_adjusted(self, frame, timestamp):
        """
        Process frame with automatic camera adjustments applied, returning mask.
        
        Args:
            frame: Input BGR frame
            timestamp: Frame timestamp
        
        Returns:
            (centers, mask) tuple
        """
        # Apply camera adjustments first
        adjusted_frame = self._apply_camera_adjustments(frame)
        
        # Then process with color detection
        return self.process_frame_with_mask(adjusted_frame, timestamp)
    
