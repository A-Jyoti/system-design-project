from flask import Flask, request, render_template_string
from time import time

import math
import numpy as np
from scipy.optimize import curve_fit

app = Flask(__name__)

#filtering and distance calculation code would be imported here if needed

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
def rssi_to_distance(rssi_filtered):
    """Convert filtered RSSI to distance using log-distance model."""
    calib_distances = [1, 2, 3, 4, 5]
    calib_rssi = [-59, -65, -69, -72, -74]

    A, n = calibrate_path_loss(calib_distances, calib_rssi)
    print(f"Calibrated Parameters → A = {A:.2f}, n = {n:.2f}\n")
    return 10 ** ((A - rssi_filtered) / (10 * n))


# Store RSSI values and timestamps
esp_data = {
    "rssi1": {"value": [], "timestamp": 0},
    "rssi2": {"value": [], "timestamp": 0},
    "rssi3": {"value": [], "timestamp": 0},
}

SYNC_TIMEOUT = 2  # seconds - consider values "in sync" if updated within 3s

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ESP32 RSSI Monitor</title>
    <meta http-equiv="refresh" content="1.0">
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { color: #2E8B57; }
        p { font-size: 22px; }
    </style>
</head>
<body>
    <h1>ESP32 RSSI Data</h1>
    <p>RSSI 1: {{ r1 }}</p>
    <p>RSSI 2: {{ r2 }}</p>
    <p>RSSI 3: {{ r3 }}</p>
    <h2>Sum (if in sync): {{ total }}</h2>
    <p>x: {{ x }}</p>
    <p>y: {{ y }}</p>
    <p>z: {{ z }}</p>
    <p>coordinates:({{ l }}, {{ m }})</p>
</body>
</html>
"""

@app.route('/')
def index():
    current_time = time()
    if all(esp_data[k]["value"] and current_time - esp_data[k]["timestamp"] < SYNC_TIMEOUT for k in esp_data):
        total = sum(v["value"][-1] for v in esp_data.values() if v["value"])
        r1 = esp_data["rssi1"]["value"][-1]
        r2 = esp_data["rssi2"]["value"][-1]
        r3 = esp_data["rssi3"]["value"][-1]

        ma_filter = MovingAverageFilter()
        filtered_x = ma_filter.update(r1)
        x = rssi_to_distance(filtered_x)
        filtered_y = ma_filter.update(r2)
        y = rssi_to_distance(filtered_y)
        filtered_z = ma_filter.update(r3)
        z = rssi_to_distance(filtered_z)
        #calculate distances
        d1=2
        d2=0.75
        try:
            expr1 = abs(x**2 -((x**2 - y**2 + d1**2)/(2*d1))**2)
            expr2 = abs(z**2 -((x**2 - y**2 + d2**2)/(2*d2))**2)
            
            if expr1 < 0 or expr2 < 0:
                print(expr1, expr2)
                l = "Invalid configuration"
                m = "Invalid configuration"
            else:
                l = math.sqrt(expr1)
                m = math.sqrt(expr2)
            
        except ValueError as e:
            print(f"Math error: {e}")
            l = "Error"
            m = "Error"
            
        print(f"Distances → x: {x:.2f} m, y: {y:.2f} m, z: {z:.2f} m")
        return render_template_string(HTML_TEMPLATE, r1=r1, r2=r2, r3=r3, total=total, x=x, y=y, z=z, l=l, m=m)

    else:
        total = "Waiting for sync..."
        print("Data not in sync or missing; cannot compute total or distances.")
        return render_template_string(
            HTML_TEMPLATE,
            r1=esp_data["rssi1"]["value"][-1] if esp_data["rssi1"]["value"] else None,
            r2=esp_data["rssi2"]["value"][-1] if esp_data["rssi2"]["value"] else None,
            r3=esp_data["rssi3"]["value"][-1] if esp_data["rssi3"]["value"] else None,
            total=total,
            x="N/A",
            y="N/A",
            z="N/A",
            l="N/A",
            m="N/A",
        )

    

@app.route('/rssi', methods=['POST'])
def update_data():
    if not request.is_json:
        return "Expected JSON data", 400

    data = request.json
    current_time = time()

    updated_keys = []
    for key in ["rssi1", "rssi2", "rssi3"]:
        if key in data:
            try:
                esp_data[key]["value"].append(float(data[key]))
                esp_data[key]["timestamp"] = current_time
                updated_keys.append(key)
            except ValueError:
                pass

    if updated_keys:
        print(f"Received update from {updated_keys} → {[(k, esp_data[k]['value'][-1] if esp_data[k]['value'] else None) for k in updated_keys]}")
        return "Data received", 200
    else:
        return "No valid RSSI keys found", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)