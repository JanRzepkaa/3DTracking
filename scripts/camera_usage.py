"""
Example usage of the camera calibration system with config-based detection.
"""

import yaml
from pathlib import Path
from input.capture_video import CameraCapture
from input.analyze_video import CameraAnalysis
from input.calibrate_feed import CameraCalibration


def load_config(config_path="config/ps3eye.yaml"):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")


def example_calibration():
    """Run interactive calibration tool."""
    print("Starting camera calibration tool...")
    print("Adjust sliders to tune color detection parameters.")
    print("Press 's' to save config, 'r' to reset, 'q' to quit.")
    
    calibration = CameraCalibration(cam_id=0, config_path="config/ps3eye.yaml")
    calibration.run()


def example_with_config():
    """Run camera feed with loaded config."""
    print("Loading configuration...")
    config = load_config("config/ps3eye.yaml")
    
    print("Initializing camera with config...")
    cam_capture = CameraCapture(cam_id=1, config=config)
    cam_capture.start_capture()
    
    print("Starting feed with analysis... Press 'q' to quit.")
    cam_capture.show_feed(show_analysis=True)


def example_direct_analysis():
    """Directly use CameraAnalysis with config."""
    config = load_config("config/ps3eye.yaml")
    analysis = CameraAnalysis(cam_id=0, config=config)
    
    # Now you can use the analysis object with the loaded configuration
    print("CameraAnalysis initialized with config")
    print(f"HSV Lower: {analysis.lower_orange}")
    print(f"HSV Upper: {analysis.upper_orange}")
    print(f"Min contour area: {analysis.min_contour_area}")
    print(f"Camera settings:")
    print(f"  Brightness: {analysis.config['camera']['brightness']}")
    print(f"  Contrast: {analysis.config['camera']['contrast']/100:.2f}x")
    print(f"  Saturation: {analysis.config['camera']['saturation']/100:.2f}x")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'calibrate':
            example_calibration()
        elif command == 'run':
            example_with_config()
        elif command == 'info':
            example_direct_analysis()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python example_usage.py [calibrate|run|info]")
    else:
        print("Camera Calibration System Examples")
        print("=" * 50)
        print("Usage: python example_usage.py [calibrate|run|info]")
        print()
        print("Commands:")
        print("  calibrate - Run interactive calibration tool (recommended first step)")
        print("              Adjust brightness, contrast, saturation")
        print("              Tune HSV color ranges")
        print("              Configure morphological operations")
        print("  run       - Run camera with calibrated config")
        print("  info      - Display loaded config information")
