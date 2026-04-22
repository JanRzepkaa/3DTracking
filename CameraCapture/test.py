import cv2
import time

def high_speed_capture():
    # 1. Use CAP_DSHOW on Windows to bypass generic OS wrappers
    cap = cv2.VideoCapture(-1, cv2.CAP_DSHOW)
    
    # 2. Force the PS3 Eye into its high-speed modes
    # Option A: 640x480 @ 60fps
    # Option B: 320x240 @ 120fps (Change these values for Option B)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # 3. CRITICAL: Kill the OS Buffer
    # This tells the OS to only keep the absolute newest frame and trash the rest.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Disable auto-exposure and auto-white balance if possible
    # This stops the camera's internal CPU from lagging while guessing lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) 
    cap.set(cv2.CAP_PROP_EXPOSURE, 2) # Manual exposure (adjust to your room)

    if not cap.isOpened():
        print("Error: Could not open PS3 Eye.")
        return

    print("Starting high-speed capture... Press 'q' to quit.")
    
    last_time = time.perf_counter()

    while True:
        # Grabbing the frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Stamp it the absolute microsecond Python gets it
        current_time = time.perf_counter()
        
        # Calculate the exact time difference since the last frame
        dt = current_time - last_time
        last_time = current_time
        
        # Calculate instantaneous FPS
        fps = 1.0 / dt if dt > 0 else 0.0
        
        # Display the stats on the screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"dt: {dt:.4f} sec", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("PS3 Eye Benchmark", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    high_speed_capture()