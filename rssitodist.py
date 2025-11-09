from flask import Flask, request, render_template_string
from time import time
import csv
import os

app = Flask(__name__)

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

rssi1_value = None
CSV_FILE = "dataset.csv"
DISTANCE = 12.3  

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_rssi","rssi1_filtered", "distance"])

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ESP32 RSSI Logger</title>
    <meta http-equiv="refresh" content="1.0">
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { color: #2E8B57; }
        p { font-size: 22px; }
    </style>
</head>
<body>
    <h1>ESP32 RSSI Data</h1>
    <p>Current Raw RSSI: {{ rssi }}</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, rssi=raw_rssi)

@app.route('/rssi', methods=['POST'])
def update_data():
    global raw_rssi

    if not request.is_json:
        return "Expected JSON data", 400

    data = request.json

    if "rssi1" in data:
        try:
            raw_rssi = float(data["rssi1"])
            filtered_rssi = rssi_filter.update(raw_rssi)
            rssi1_value = round(filtered_rssi, 2)

            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([raw_rssi, rssi1_value, DISTANCE])

            print(f"Raw RSSI: {raw_rssi}  |  Logged RSSI1 (filtered): {rssi1_value}  |  Distance: {DISTANCE}")
            return "Data received", 200

        except ValueError:
            return "Invalid RSSI value", 400

    return "No rssi1 key found", 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)