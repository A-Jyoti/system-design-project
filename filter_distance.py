"""
RSSI → Distance Estimation using:
1️⃣ Log-Distance Path Loss Model
2️⃣ Moving Average Filter (MAF)
3️⃣ Automatic Path Loss Exponent Calibration
"""

import math
import numpy as np
from scipy.optimize import curve_fit


# --- Step 1: Fit Path Loss Model (A and n) -------------------------
def path_loss_model(d, A, n):
    """Model: RSSI = A - 10 * n * log10(d)"""
    return A - 10 * n * np.log10(d)


def calibrate_path_loss(distances, rssi_values):
    """Estimate A and n from calibration data."""
    distances = np.array(distances)
    rssi_values = np.array(rssi_values)
    popt, _ = curve_fit(path_loss_model, distances, rssi_values, p0=(-59, 2))
    A_fit, n_fit = popt
    return A_fit, n_fit


# --- Step 2: Moving Average Filter --------------------------------
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []

    def update(self, value):
        """Add new RSSI reading and return averaged RSSI."""
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return sum(self.window) / len(self.window)


# --- Step 3: RSSI → Distance Conversion ----------------------------
def rssi_to_distance(rssi_filtered, A, n):
    """Convert filtered RSSI to distance using log-distance model."""
    return 10 ** ((A - rssi_filtered) / (10 * n))


# --- Step 4: Example Usage ----------------------------------------
if __name__ == "__main__":
    # Calibration data (measured distances vs RSSI at those distances)
    calib_distances = [1, 2, 3, 4, 5]
    calib_rssi = [-59, -65, -69, -72, -74]

    A, n = calibrate_path_loss(calib_distances, calib_rssi)
    print(f"Calibrated Parameters → A = {A:.2f}, n = {n:.2f}\n")

    # Initialize filter and simulate incoming RSSI readings
    filter = MovingAverageFilter(window_size=5)
    rssi_stream = [-70, -74, -70, -68, -72, -74, -70, -72, -75, -70]

    print("RSSI  |  Filtered RSSI  |  Estimated Distance (m)")
    print("------|-----------------|------------------------")
    for rssi in rssi_stream:
        filtered = filter.update(rssi)
        distance = rssi_to_distance(filtered, A, n)
        print(f"{rssi:5.1f} | {filtered:15.2f} | {distance:23.2f}")