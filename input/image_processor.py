"""
Image processing utilities for camera frame adjustments.
Handles brightness, contrast, saturation, and other global frame adjustments.
"""

import cv2
import numpy as np


class ImageProcessor:
    """Apply global camera adjustments to frames."""
    
    @staticmethod
    def adjust_brightness(frame, brightness):
        """
        Adjust frame brightness.
        
        Args:
            frame: Input BGR frame
            brightness: -100 to 100 (0 = no change, negative = darker, positive = brighter)
        
        Returns:
            Adjusted frame
        """
        if brightness == 0:
            return frame
        
        adjusted = frame.astype(np.int16)
        
        # Brightness adjustment (add constant to all pixels)
        adjustment = int((brightness / 100.0) * 255)
        adjusted = adjusted + adjustment
        
        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    @staticmethod
    def adjust_contrast(frame, contrast):
        """
        Adjust frame contrast.
        
        Args:
            frame: Input BGR frame
            contrast: 0.5 to 3.0 (1.0 = no change, <1 = lower contrast, >1 = higher contrast)
        
        Returns:
            Adjusted frame
        """
        if contrast == 1.0:
            return frame
        
        # Contrast adjustment using CLAHE or simple scaling
        adjusted = frame.astype(np.float32)
        
        # Scale pixels around midpoint (128)
        midpoint = 128
        adjusted = (adjusted - midpoint) * contrast + midpoint
        
        # Clip to valid range
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    @staticmethod
    def adjust_saturation(frame, saturation):
        """
        Adjust frame saturation (color intensity).
        
        Args:
            frame: Input BGR frame
            saturation: 0.0 to 2.0 (1.0 = no change, 0.0 = grayscale, >1 = more saturated)
        
        Returns:
            Adjusted frame
        """
        if saturation == 1.0:
            return frame
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation channel (S channel, index 1)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        
        # Clip saturation to valid range
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return adjusted
    
    @staticmethod
    def adjust_gamma(frame, gamma):
        """
        Adjust frame gamma (brightness curve).
        
        Args:
            frame: Input BGR frame
            gamma: 0.5 to 2.5 (1.0 = no change, <1 = brighter, >1 = darker)
        
        Returns:
            Adjusted frame
        """
        if gamma == 1.0:
            return frame
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        
        # Apply gamma correction using lookup table
        return cv2.LUT(frame, table)
    
    @staticmethod
    def apply_all_adjustments(frame, config):
        """
        Apply all camera adjustments from config to frame.
        
        Args:
            frame: Input BGR frame
            config: Configuration dictionary with camera adjustments
        
        Returns:
            Adjusted frame
        """
        if config is None:
            return frame
        
        camera_config = config.get('camera', {})
        
        # Apply adjustments in order
        brightness = camera_config.get('brightness', 0)
        contrast = camera_config.get('contrast', 1.0)
        saturation = camera_config.get('saturation', 1.0)
        
        # Apply adjustments
        adjusted = frame.copy()
        
        if brightness != 0:
            adjusted = ImageProcessor.adjust_brightness(adjusted, brightness)
        
        if contrast != 1.0:
            adjusted = ImageProcessor.adjust_contrast(adjusted, contrast)
        
        if saturation != 1.0:
            adjusted = ImageProcessor.adjust_saturation(adjusted, saturation)
        
        return adjusted
    
    @staticmethod
    def get_default_camera_config():
        """Return default camera adjustment configuration."""
        return {
            'brightness': 0,
            'contrast': 1.0,
            'saturation': 1.0,
            'exposure': 0,
            'gain': 0
        }
