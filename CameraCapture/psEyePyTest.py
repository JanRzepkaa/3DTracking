from pseyepy import Camera, Display
import cv2
import numpy as np

def libusb_high_speed_capture():
    # Initialize the camera via libusb
    # RES_LARGE is 640x480. RES_SMALL is 320x240
    try:
        cam = Camera(0, fps=60, resolution=Camera.RES_SMALL, colour=True)
    except Exception as e:
        print("Could not find PS3 Eye on libusb. Did Zadig finish successfully?")
        return

    print("Direct libusb stream started! Press 'q' to quit.")

    last_timestamp = 0
    while True:
        # pseyepy natively returns the exact hardware timestamp of the frame!
        frame, timestamp = cam.read()
        
        # Sometimes the buffer is empty if we loop faster than 60fps
        if frame is None:
            continue
            
        # The frame comes in as an RGB array. OpenCV expects BGR for display.
        # We must convert it to BGR before showing it on the screen.
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the hardware timestamp directly on the feed
        cv2.putText(bgr_frame, f"HW Time: {timestamp:.4f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(bgr_frame, f"FPS: {1.0/(timestamp - last_timestamp+1e-6):.2f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        last_timestamp = timestamp
        
        cv2.imshow("PS3 Eye Direct libusb", bgr_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Safely close the USB connection
    cam.end()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    libusb_high_speed_capture()