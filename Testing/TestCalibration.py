import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Calibration import *

def test_rotate_vector():
    tests = [
            # --- Your original tests ---
        (np.array([1, 0, 0]), (0, 0, np.pi/2), np.array([0, 1, 0])),
        (np.array([0, 1, 0]), (0, 0, np.pi/2), np.array([-1, 0, 0])),
        (np.array([1, 0, 0]), (0, np.pi/2, 0), np.array([0, 0, -1])),
        (np.array([0, 0, 1]), (0, np.pi/2, 0), np.array([1, 0, 0])),
        (np.array([0, 1, 0]), (np.pi/2, 0, 0), np.array([0, 0, 1])),
        (np.array([0, 0, 1]), (np.pi/2, 0, 0), np.array([0, -1, 0])),

        # --- Identity rotation ---
        (np.array([1, 2, 3]), (0, 0, 0), np.array([1, 2, 3])),

        # --- 180 degree rotations ---
        (np.array([1, 0, 0]), (0, 0, np.pi), np.array([-1, 0, 0])),
        (np.array([0, 1, 0]), (np.pi, 0, 0), np.array([0, -1, 0])),
        (np.array([0, 0, 1]), (0, np.pi, 0), np.array([0, 0, -1])),

        # --- Negative rotations ---
        (np.array([1, 0, 0]), (0, 0, -np.pi/2), np.array([0, -1, 0])),
        (np.array([0, 1, 0]), (-np.pi/2, 0, 0), np.array([0, 0, -1])),

        # --- Non-axis aligned vector ---
        (np.array([1, 1, 0]), (0, 0, np.pi/2), np.array([-1, 1, 0])),

        # --- Combined rotations ---
        (np.array([1, 0, 0]), (np.pi/2, np.pi/2, 0), np.array([0, 0, -1])),

        # --- Another general vector ---
        (np.array([1, 2, 3]), (0, 0, np.pi/2), np.array([-2, 1, 3])),
    ]

    
    for i, (vector, rotation, expected) in enumerate(tests):
        result = rotate_vector(vector, rotation)
        assert np.allclose(result, expected), f"Test {i} failed: expected {expected}, got {result}"
    print("All tests passed for rotate_vector!")

if __name__ == "__main__":
    test_rotate_vector()