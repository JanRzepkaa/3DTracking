import multiprocessing as mp
import time

import yaml
from input.capture_video import CameraCapture

def camera_worker(cam_id, config, pipe_conn):
    """
    This function runs in a completely separate CPU process.
    It handles its own camera initialization, capturing, and analysis.
    """
    print(f"[Worker {cam_id}] Starting process...")
    
    # Initialize the camera inside the worker process
    cam = CameraCapture(cam_id=cam_id, config=config)
    cam.start_capture()
    
    if cam.cam is None:
        print(f"[Worker {cam_id}] Failed to start. Exiting.")
        pipe_conn.send(None)
        return

    try:
        while True:
            # Check if the main process sent a 'STOP' signal
            if pipe_conn.poll():
                msg = pipe_conn.recv()
                if msg == "STOP":
                    break

            # Capture and analyze
            frame, timestamp = cam.capture_frame()
            if frame is not None:
                # We use the auto_adjusted method based on your analyze_video.py
                centers = cam.analyze_current_frame_with_adjustments(debug=False)
                # Send ONLY the lightweight data across the IPC pipe
                pipe_conn.send({
                    'cam_id': cam_id,
                    'timestamp': timestamp,
                    'centers': centers
                })
            else:
                # Small sleep to prevent CPU thrashing if the camera drops a frame
                time.sleep(0.001) 
                
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[Worker {cam_id}] Shutting down...")
        cam.stop_capture()
        pipe_conn.close()


class MultiCameraManager:
    def __init__(self, camera_ids, config=None):
        self.camera_ids = camera_ids
        self.config = config
        
        self.processes = []
        self.pipes = {}  # Stores the main-process end of the pipes

    def start(self):
        """Spawns a worker process for each camera."""
        for cam_id in self.camera_ids:
            # Create a 2-way pipe for IPC. 
            # parent_conn stays here, child_conn goes to the worker.
            parent_conn, child_conn = mp.Pipe()
            self.pipes[cam_id] = parent_conn
            
            p = mp.Process(target=camera_worker, args=(cam_id, self.config, child_conn))
            p.daemon = True # Ensures workers die if the main script crashes
            p.start()
            self.processes.append(p)
            
        print("All camera processes started.")

    def get_latest_data(self):
        """
        Polls all pipes to grab the most recent data from each camera.
        Returns a dictionary keyed by cam_id.
        """
        latest_data = {}
        
        for cam_id, pipe in self.pipes.items():
            # A camera might have processed 3 frames since we last checked.
            # We want to flush the pipe and only keep the newest one.
            cam_data = None
            while pipe.poll():
                cam_data = pipe.recv()
                
            if cam_data is not None:
                latest_data[cam_id] = cam_data
                
        return latest_data

    def stop(self):
        """Sends stop signals and cleans up processes."""
        for pipe in self.pipes.values():
            pipe.send("STOP")
            
        for p in self.processes:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        print("Camera Manager shutdown complete.")

# --- Example Usage ---
if __name__ == "__main__":
    # Assuming you have two PS3 eyes plugged in (IDs 0 and 1)
    # Note: On Windows, multiprocessing MUST be protected by __main__
    # Load config from config/ps3eye.yaml or pass None for defaults
    with open("config/ps3eye.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Print loaded config for verification
    print("Loaded Camera Config:")
    print(config)

    cam_manager = MultiCameraManager(camera_ids=[0, 1], config=config)
    cam_manager.start()
    
    try:
        print("Listening for data... Press Ctrl+C to stop.")
        while True:
            # This is your fast RL loop
            data = cam_manager.get_latest_data()
            
            if data:
                # Just printing to verify it works
                for cam_id, info in data.items():
                    print(f"Cam {cam_id} @ {info['timestamp']:.4f}: {info['centers']}")
                    
            time.sleep(1/120.0) # Simulate a 60Hz RL loop
            
    except KeyboardInterrupt:
        cam_manager.stop()