from flask import Flask, render_template_string
from flask_sock import Sock
from time import time
import math
import json
import numpy as np
from scipy.optimize import curve_fit

app = Flask(__name__)
sock = Sock(app)

# ------------------ Path Loss Model ------------------
def path_loss_model(d, A, n):
    return A - 10 * n * np.log10(d)

def calibrate_path_loss(distances, rssi_values):
    distances = np.array(distances)
    rssi_values = np.array(rssi_values)
    popt, _ = curve_fit(path_loss_model, distances, rssi_values, p0=(-59, 2))
    return popt  # (A, n)

def rssi_to_distance(rssi_filtered):
    calib_distances = [1, 2, 3, 4, 5]
    calib_rssi = [-59, -65, -69, -72, -74]
    A, n = calibrate_path_loss(calib_distances, calib_rssi)
    return 10 ** ((A - rssi_filtered) / (10 * n))

# ------------------ Moving Average -------------------
class MovingAverageFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []

    def update(self, value):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return sum(self.window) / len(self.window)

# ------------------ Shared Data ----------------------
esp_data = {
    "rssi1": {"value": [], "timestamp": 0},
    "rssi2": {"value": [], "timestamp": 0},
    "rssi3": {"value": [], "timestamp": 0},
}

clients = []  # connected web browsers
SYNC_TIMEOUT = 2  # seconds

# ------------------ Web Page -------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ESP32 RSSI Monitor (WebSocket)</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { color: #2E8B57; }
        p { font-size: 22px; }
    </style>
</head>
<body>
    <h1>ESP32 RSSI Data</h1>
    <p>RSSI 1: <span id="r1">--</span></p>
    <p>RSSI 2: <span id="r2">--</span></p>
    <p>RSSI 3: <span id="r3">--</span></p>
    <h2>Computed Distances</h2>
    <p>x: <span id="x">--</span></p>
    <p>y: <span id="y">--</span></p>
    <p>z: <span id="z">--</span></p>
    <p>Coordinates (l, m): (<span id="l">--</span>, <span id="m">--</span>)</p>

    <script>
        const socket = new WebSocket("ws://10.114.196.229:5000/ws");
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            document.getElementById("r1").textContent = data.rssi1;
            document.getElementById("r2").textContent = data.rssi2;
            document.getElementById("r3").textContent = data.rssi3;
            document.getElementById("x").textContent = data.x;
            document.getElementById("y").textContent = data.y;
            document.getElementById("z").textContent = data.z;
            document.getElementById("l").textContent = data.l;
            document.getElementById("m").textContent = data.m;
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# ------------------ WebSocket endpoint ----------------
@sock.route('/ws')
def ws(ws):
    clients.append(ws)
    print("New client connected.")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break  # client disconnected
            data = json.loads(msg)
            handle_rssi_data(data)
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        clients.remove(ws)
        print("Client disconnected 1.")

# ------------------ Handle ESP32 data -----------------
def handle_rssi_data(data):
    print(data)
    current_time = time()
    for key in ["rssi1", "rssi2", "rssi3"]:
        if key in data:
            esp_data[key]["value"].append(float(data[key]))
            esp_data[key]["timestamp"] = current_time

    # Only process if all are recent
    if all(esp_data[k]["value"] and current_time - esp_data[k]["timestamp"] < SYNC_TIMEOUT for k in esp_data):
        r1 = esp_data["rssi1"]["value"][-1]
        r2 = esp_data["rssi2"]["value"][-1]
        r3 = esp_data["rssi3"]["value"][-1]

        ma_filter = MovingAverageFilter()
        filtered_r1 = ma_filter.update(r1)
        x = rssi_to_distance(filtered_r1)
        filtered_r2 = ma_filter.update(r2)
        y = rssi_to_distance(filtered_r2)
        filtered_r3 = ma_filter.update(r3)
        z = rssi_to_distance(filtered_r3)

        d1 = 2
        d2 = 0.75
        try:
            expr1 = abs(x**2 - ((x**2 - y**2 + d1**2)/(2*d1))**2)
            expr2 = abs(z**2 - ((x**2 - y**2 + d2**2)/(2*d2))**2)
            l = math.sqrt(expr1)
            m = math.sqrt(expr2)
        except ValueError:
            l = "Err"
            m = "Err"

        payload = {
            "rssi1": r1, "rssi2": r2, "rssi3": r3,
            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
            "l": l if isinstance(l, str) else round(l, 2),
            "m": m if isinstance(m, str) else round(m, 2)
        }

        # Broadcast to all connected browsers
        print("Here for braodcasting")
        for c in clients[:]:
            try:
                print("Broadcasting:", payload)
                c.send(json.dumps(payload))
            except:
                print("Hula Hoops")
                # clients.remove(c)
    else:
        r1 = 0.0
        r2 = 0.0
        r3 = 0.0

        try:
            r1 = esp_data["rssi1"]["value"][-1]
        except IndexError:
            pass

        try:
            r2 = esp_data["rssi2"]["value"][-1]
        except IndexError:
            pass

        try:
            r3 = esp_data["rssi3"]["value"][-1]
        except IndexError:
            pass
        # r2 = esp_data["rssi2"]["value"][-1]
        # r3 = esp_data["rssi3"]["value"][-1]

        payload = {
            "rssi1": r1, "rssi2": r2, "rssi3": r3,
            # "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
            # "l": l if isinstance(l, str) else round(l, 2),
            # "m": m if isinstance(m, str) else round(m, 2)
        }

        # Broadcast to all connected browsers
        print("Here for braodcasting2")
        for c in clients[:]:
            try:
                print("Broadcasting2:", payload)
                c.send(json.dumps(payload))
            except:
                print("Hula hoops2")
        print("Data not in sync or missing; skipping computation2.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
