import cv2

def scan_cameras():
    print("Scanning for cameras...")
    
    # Check the first 5 indices
    for index in range(5):
        print(f"\n--- Testing Index {index} ---")
        
        # Test 1: Default OS Backend
        cap_default = cv2.VideoCapture(index)
        if cap_default.isOpened():
            print(f"SUCCESS: Found camera at Index {index} (Default Backend)")
            cap_default.release()
        else:
            print(f"Failed: Index {index} (Default Backend)")

        # Test 2: DirectShow (DSHOW)
        cap_dshow = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap_dshow.isOpened():
            print(f"SUCCESS: Found camera at Index {index} (DirectShow)")
            cap_dshow.release()
        else:
            print(f"Failed: Index {index} (DirectShow)")
            
        # Test 3: Media Foundation (MSMF) - Modern Windows standard
        cap_msmf = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if cap_msmf.isOpened():
            print(f"SUCCESS: Found camera at Index {index} (MSMF)")
            cap_msmf.release()
        else:
            print(f"Failed: Index {index} (MSMF)")

if __name__ == "__main__":
    scan_cameras()