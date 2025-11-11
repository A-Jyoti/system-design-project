from flask import Flask, request, render_template_string
from time import time
import csv
import os

app = Flask(__name__)

# Store RSSI values and timestamps
esp_data = {
    "rssi1": {"value": None, "timestamp": 0},
    "rssi2": {"value": None, "timestamp": 0},
    "rssi3": {"value": None, "timestamp": 0},
}

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

rssi_filter = MovingAverageFilter(window_size=5)

SYNC_TIMEOUT = 2  # seconds
CSV_FILE = "dataset1.csv"

# Hardcoded parameters
D1 = 8.0
D2 = 3.0
DX = 5.0
DY = 1.5

# Create CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d1", "d2", "raw_rssi1","filtered_rssi1", "raw_rssi2","filtered_rssi2", "raw_rssi3","filtered_rssi3", "dx", "dy"])

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
</body>
</html>
"""

@app.route('/')
def index():
    current_time = time()
    if all(current_time - esp_data[k]["timestamp"] < SYNC_TIMEOUT and esp_data[k]["value"] is not None
           for k in esp_data):
        total = sum(v["value"] for v in esp_data.values())

        # Append to CSV when all readings are in sync
        with open(CSV_FILE, "a", newline="") as f:
            raw_rssi1 = float(esp_data["rssi1"]["value"])
            filtered_rssi1 = rssi_filter.update(raw_rssi1)
            rssi1_value = round(filtered_rssi1, 2)
            
            raw_rssi2 = float(esp_data["rssi2"]["value"])
            filtered_rssi2 = rssi_filter.update(raw_rssi2)
            rssi2_value = round(filtered_rssi2, 2)
            
            raw_rssi3 = float(esp_data["rssi3"]["value"])
            filtered_rssi3 = rssi_filter.update(raw_rssi3)
            rssi3_value = round(filtered_rssi3, 2)
            
            writer = csv.writer(f)
            writer.writerow([
                D1, D2,
                esp_data["rssi1"]["value"],
                rssi1_value,
                esp_data["rssi2"]["value"],
                rssi2_value,
                esp_data["rssi3"]["value"],
                rssi3_value,
                DX, DY
            ])
    else:
        total = "Waiting for sync..."

    return render_template_string(
        HTML_TEMPLATE,
        r1=esp_data["rssi1"]["value"],
        r2=esp_data["rssi2"]["value"],
        r3=esp_data["rssi3"]["value"],
        total=total
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
                esp_data[key]["value"] = float(data[key])
                esp_data[key]["timestamp"] = current_time
                updated_keys.append(key)
            except ValueError:
                pass

    if updated_keys:
        print(f"Received update from {updated_keys} → {[(k, esp_data[k]['value']) for k in updated_keys]}")
        return "Data received", 200
    else:
        return "No valid RSSI keys found", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
