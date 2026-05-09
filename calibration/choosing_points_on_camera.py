import cv2
import numpy as np
import json
import os
from input.capture_video import CameraCapture

class CalibrationManager:
    def __init__(self, camera_ids, targets):
        """
        Initializes the bulky, single-process calibration manager with zoom support.
        :param camera_ids: List of camera IDs (e.g., [0, 1])
        :param targets: List of string names for the points you need to calibrate.
        """
        self.camera_ids = camera_ids
        self.targets = targets
        
        self.colors = [
            (0, 0, 255),   # Red
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (255, 255, 0)  # Cyan
        ]
        
        self.calibration_data = {cam_id: {} for cam_id in camera_ids}
        self.current_cam_idx = 0
        self.current_target_idx = 0
        self.is_calibration_complete = False
        self.history = [] 

        # --- Zoom & Pan State Variables ---
        self.zoom_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Initialize hardware
        self.cameras = []
        for cid in camera_ids:
            cam = CameraCapture(cam_id=cid)
            cam.start_capture()
            self.cameras.append(cam)
            
        self.window_name = "Multi-Camera Calibration Dashboard"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        """Handles mouse clicks, scrolling, and dragging on the stitched window."""
        
        # 1. Handle Zooming (Scroll Wheel)
        if event == cv2.EVENT_MOUSEWHEEL:
            # flags > 0 implies scroll up, flags < 0 implies scroll down
            zoom_factor = 1.1 if flags > 0 else 0.9
            new_scale = self.zoom_scale * zoom_factor
            new_scale = max(1.0, min(new_scale, 15.0)) # Clamp between 1x and 15x

            # Shift the pan center so the zoom occurs exactly at the mouse cursor
            real_x = self.pan_x + x / self.zoom_scale
            real_y = self.pan_y + y / self.zoom_scale
            self.pan_x = real_x - x / new_scale
            self.pan_y = real_y - y / new_scale
            self.zoom_scale = new_scale

        # 2. Handle Panning (Middle Mouse Button Drag)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.last_mouse_x = x
            self.last_mouse_y = y
            
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                # Calculate movement relative to current scale
                dx = (x - self.last_mouse_x) / self.zoom_scale
                dy = (y - self.last_mouse_y) / self.zoom_scale
                self.pan_x -= dx
                self.pan_y -= dy
                self.last_mouse_x = x
                self.last_mouse_y = y

        # 3. Handle Calibration Point Selection (Left Click)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.is_calibration_complete:
                print("Calibration complete. Press 's' to save or 'z' to undo.")
                return

            # Translate window coordinates back to the original unzoomed image coordinates
            real_x = int(self.pan_x + x / self.zoom_scale)
            real_y = int(self.pan_y + y / self.zoom_scale)

            frame_width = 640 
            clicked_cam_idx = real_x // frame_width
            local_x = real_x % frame_width
            expected_cam_idx = self.current_cam_idx
            
            if clicked_cam_idx != expected_cam_idx:
                print(f"Please click on Camera {self.camera_ids[expected_cam_idx]} feed!")
                return

            cam_id = self.camera_ids[expected_cam_idx]
            target_name = self.targets[self.current_target_idx]
            
            self.calibration_data[cam_id][target_name] = (local_x, real_y)
            self.history.append((cam_id, target_name))
            
            print(f"Recorded {target_name} on Cam {cam_id} at ({local_x}, {real_y})")
            self._advance_state()

    def _advance_state(self):
        self.current_target_idx += 1
        if self.current_target_idx >= len(self.targets):
            self.current_target_idx = 0
            self.current_cam_idx += 1
            if self.current_cam_idx >= len(self.camera_ids):
                self.is_calibration_complete = True

    def _undo_last_click(self):
        if not self.history:
            print("Nothing to undo.")
            return
            
        last_cam_id, last_target = self.history.pop()
        del self.calibration_data[last_cam_id][last_target]
        
        self.is_calibration_complete = False
        self.current_target_idx -= 1
        
        if self.current_target_idx < 0:
            self.current_cam_idx -= 1
            self.current_target_idx = len(self.targets) - 1
            
        print(f"Undid {last_target} on Camera {last_cam_id}.")

    def _save_data(self):
        if not self.is_calibration_complete:
            print("Warning: Saving incomplete calibration data.")
            
        clean_data = {}
        for cam, targets in self.calibration_data.items():
            clean_data[str(cam)] = {
                t: {"x": int(coords[0]), "y": int(coords[1])} 
                for t, coords in targets.items()
            }
            
        # Ensure directory exists before saving
        os.makedirs("calibration", exist_ok=True)
        filepath = "calibration/calibration_data.json"
        
        with open(filepath, 'w') as f:
            json.dump(clean_data, f, indent=4)
        print(f"Calibration saved successfully to {os.path.abspath(filepath)}")

    def run(self):
        print("Starting Calibration Manager...")
        print("Controls: [L-Click] Select Point | [Scroll] Zoom | [M-Click Drag] Pan")
        print("          [Z] Undo | [R] Reset | [S] Save | [Q] Quit")
        
        try:
            while True:
                frames = []
                for cam in self.cameras:
                    frame, _ = cam.capture_frame()
                    if frame is None:
                        frame = np.zeros((480, 640, 3), dtype=np.uint8) 
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(bgr_frame)

                # Base stitched image
                stitched_view = np.hstack(frames)
                img_h, img_w = stitched_view.shape[:2]

                # Draw physical points on the base image before zooming
                for c_idx, cam_id in enumerate(self.camera_ids):
                    for t_idx, target in enumerate(self.targets):
                        if target in self.calibration_data[cam_id]:
                            local_x, y = self.calibration_data[cam_id][target]
                            global_x = local_x + (c_idx * 640) 
                            color = self.colors[t_idx % len(self.colors)]
                            
                            cv2.circle(stitched_view, (global_x, y), 8, color, -1)
                            cv2.putText(stitched_view, target, (global_x + 10, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # --- Execute Zoom and Pan Logic ---
                view_w = img_w / self.zoom_scale
                view_h = img_h / self.zoom_scale

                # Constrain panning to the image boundaries
                self.pan_x = max(0, min(self.pan_x, img_w - view_w))
                self.pan_y = max(0, min(self.pan_y, img_h - view_h))

                # Crop Region of Interest (ROI)
                x1, y1 = int(self.pan_x), int(self.pan_y)
                x2, y2 = int(self.pan_x + view_w), int(self.pan_y + view_h)
                roi = stitched_view[y1:y2, x1:x2]

                # Stretch ROI back to original window dimensions
                zoomed_view = cv2.resize(roi, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

                # --- Draw HUD on the Zoomed Image ---
                hud_text = ""
                if self.is_calibration_complete:
                    hud_text = "CALIBRATION COMPLETE. Press 'S' to Save or 'Q' to Quit."
                    hud_color = (0, 255, 0)
                else:
                    active_cam = self.camera_ids[self.current_cam_idx]
                    active_target = self.targets[self.current_target_idx]
                    hud_text = f"ACTION: Click '{active_target}' on Camera {active_cam} | Zoom: {self.zoom_scale:.1f}x"
                    hud_color = (0, 255, 255)
                    
                cv2.rectangle(zoomed_view, (0, 0), (img_w, 40), (0, 0, 0), -1)
                cv2.putText(zoomed_view, hud_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)

                # Render final frame
                cv2.imshow(self.window_name, zoomed_view)

                # Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('z'):
                    self._undo_last_click()
                elif key == ord('r'):
                    print("Resetting all data...")
                    self.calibration_data = {cam_id: {} for cam_id in self.camera_ids}
                    self.history = []
                    self.current_cam_idx = 0
                    self.current_target_idx = 0
                    self.is_calibration_complete = False
                    self.zoom_scale = 1.0
                    self.pan_x, self.pan_y = 0.0, 0.0
                elif key == ord('s'):
                    self._save_data()

        finally:
            print("Shutting down cameras...")
            for cam in self.cameras:
                cam.stop_capture()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    CAMS = [0, 1, 2]
    TARGETS = [
        "Bottom Left",
        "Bottom Right",
        "Top Right",
        "Top Left"
    ]
    
    manager = CalibrationManager(camera_ids=CAMS, targets=TARGETS)
    manager.run()