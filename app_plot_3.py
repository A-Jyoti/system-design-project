from flask import Flask, render_template_string
from flask_sock import Sock
from time import time
import math
import json
import numpy as np
from scipy.optimize import curve_fit
import statistics

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

# ----------------- Coordinate Calculation ----------------
def sanitize_distances(distances, room_diag, min_d=0.05):
    """Clamp distances to reasonable positive range."""
    distances = np.array(distances, dtype=float)
    max_d = room_diag * 1.2
    distances = np.clip(distances, min_d, max_d)
    return distances

def multilateration_least_squares(positions, distances, weights=None):
    p = np.asarray(positions, dtype=float)
    r = np.asarray(distances, dtype=float)
    N = p.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 receivers for 2D multilateration")

    x0, y0 = p[0]
    r0 = r[0]
    A, b = [], []

    for i in range(1, N):
        xi, yi = p[i]
        ri = r[i]
        A.append([2*(xi - x0), 2*(yi - y0)])
        b.append((r0**2 - ri**2) - (x0**2 - xi**2) - (y0**2 - yi**2))
    A, b = np.array(A, dtype=float), np.array(b, dtype=float)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        W = np.sqrt(w)[:, None]
        A = A * W
        b = b * W.ravel()

    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    px, py = sol
    est_dists = np.sqrt((p[:,0]-px)**2 + (p[:,1]-py)**2)
    resid_norm = np.linalg.norm(est_dists - r)
    return (px, py, resid_norm, rank >= 2)

def coarse_grid_search(positions, distances, room_bounds, grid_res=0.1):
    xmin, xmax, ymin, ymax = room_bounds
    xs = np.arange(xmin, xmax + 1e-9, grid_res)
    ys = np.arange(ymin, ymax + 1e-9, grid_res)
    P = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
    p = np.asarray(positions)
    r = np.asarray(distances)
    best = (None, None, np.inf)
    for px, py in P:
        d = np.sqrt((p[:,0]-px)**2 + (p[:,1]-py)**2)
        resid = np.linalg.norm(d - r)
        if resid < best[2]:
            best = (px, py, resid)
    return best

def estimate_position_from_rssi(r1, r2, r3, D1, D2, room_margin=0.1):
    positions = [(0.0, 0.0), (D1, 0.0), (0.0, D2)]
    room_diag = np.sqrt(D1**2 + D2**2)
    dists = sanitize_distances([r1, r2, r3], room_diag)
    px, py, resid_norm, success = multilateration_least_squares(positions, dists)

    if not success or np.isnan(px) or np.isnan(py) or resid_norm > max(0.5, 0.1 * room_diag):
        xmin, xmax, ymin, ymax = -room_margin, D1 + room_margin, -room_margin, D2 + room_margin
        bx, by, bres = coarse_grid_search(positions, dists, (xmin, xmax, ymin, ymax), grid_res=0.05)
        return bx, by, bres, False
    return px, py, resid_norm, True

# ------------------ Filters ----------------------
class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []
    def update(self, value):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        return statistics.median(self.window)

class ExponentialSmoother:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last = None
    def update(self, value):
        if self.last is None:
            self.last = value
        else:
            self.last = self.alpha * value + (1 - self.alpha) * self.last
        return self.last

# Persistent filters
ma_filter_x = MedianFilter(window_size=7)
ma_filter_y = MedianFilter(window_size=7)
ma_filter_z = MedianFilter(window_size=7)
smooth_x = ExponentialSmoother(alpha=0.25)
smooth_y = ExponentialSmoother(alpha=0.25)
smooth_z = ExponentialSmoother(alpha=0.25)
coord_smooth_l = ExponentialSmoother(alpha=0.2)
coord_smooth_m = ExponentialSmoother(alpha=0.2)

# ------------------ Shared Data ----------------------
esp_data = {
    "rssi1": {"value": [], "timestamp": 0},
    "rssi2": {"value": [], "timestamp": 0},
    "rssi3": {"value": [], "timestamp": 0},
}

browser_clients = []
SYNC_TIMEOUT = 2

# Room parameters
D1 = 4.8
D2 = 6.6
SAFETY_RADIUS = 1.0

# ------------------ Web Page -------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ESP32 RSSI Position Tracker</title>
    <style>
        body { 
            font-family: Arial; 
            text-align: center; 
            margin: 20px;
            background-color: #f0f0f0;
        }
        h1 { color: #2E8B57; }
        .container {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .data-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .data-panel p { 
            font-size: 18px; 
            margin: 10px 0;
        }
        .visualization {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        canvas {
            border: 2px solid #333;
            background: #fafafa;
        }
        .label {
            font-weight: bold;
            color: #555;
        }
        .status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-left: 5px;
        }
        .status.connected {
            background-color: #2ecc71;
        }
        .status.disconnected {
            background-color: #e74c3c;
        }
    </style>
</head>
<body>
    <h1>🎯 ESP32 Position Tracker</h1>
    <p>Connection: <span class="status disconnected" id="connectionStatus"></span></p>
    
    <div class="container">
        <div class="data-panel">
            <h2>📡 RSSI Values</h2>
            <p><span class="label">RSSI 1:</span> <span id="r1">--</span> dBm</p>
            <p><span class="label">RSSI 2:</span> <span id="r2">--</span> dBm</p>
            <p><span class="label">RSSI 3:</span> <span id="r3">--</span> dBm</p>
            
            <h2>📏 Distances</h2>
            <p><span class="label">x:</span> <span id="x">--</span> m</p>
            <p><span class="label">y:</span> <span id="y">--</span> m</p>
            <p><span class="label">z:</span> <span id="z">--</span> m</p>
            
            <h2>📍 Position</h2>
            <p><span class="label">l:</span> <span id="l">--</span> m</p>
            <p><span class="label">m:</span> <span id="m">--</span> m</p>
        </div>
        
        <div class="visualization">
            <h2>🗺️ Room Map ({{ d1 }}m × {{ d2 }}m)</h2>
            <canvas id="roomCanvas" width="600" height="450"></canvas>
            <p style="font-size: 14px; color: #666; margin-top: 10px;">
                🔵 Person | ⭕ Safety Radius ({{ safety_radius }}m)
            </p>
        </div>
    </div>

    <script>
        const D1 = {{ d1 }};
        const D2 = {{ d2 }};
        const SAFETY_RADIUS = {{ safety_radius }};
        const canvas = document.getElementById('roomCanvas');
        const ctx = canvas.getContext('2d');
        const statusIndicator = document.getElementById('connectionStatus');
        
        // Calculate scale to fit room in canvas with padding
        const padding = 40;
        const scaleX = (canvas.width - 2 * padding) / D1;
        const scaleY = (canvas.height - 2 * padding) / D2;
        const scale = Math.min(scaleX, scaleY);
        
        // Current and target positions for smooth animation
        let currentPos = { l: D1/2, m: D2/2 };
        let targetPos = { l: D1/2, m: D2/2 };
        let animationProgress = 1;
        
        // Convert room coordinates to canvas coordinates
        function toCanvasCoords(l, m) {
            return {
                x: padding + l * scale,
                y: canvas.height - padding - m * scale
            };
        }
        
        // Draw the room and person
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw room outline
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 3;
            const roomStart = toCanvasCoords(0, 0);
            const roomEnd = toCanvasCoords(D1, D2);
            ctx.strokeRect(roomStart.x, roomEnd.y, roomEnd.x - roomStart.x, roomStart.y - roomEnd.y);
            
            // Draw grid
            ctx.strokeStyle = '#ddd';
            ctx.lineWidth = 1;
            const gridSteps = 10;
            for (let i = 1; i < gridSteps; i++) {
                const x = padding + (canvas.width - 2*padding) * i / gridSteps;
                ctx.beginPath();
                ctx.moveTo(x, padding);
                ctx.lineTo(x, canvas.height - padding);
                ctx.stroke();
                
                const y = padding + (canvas.height - 2*padding) * i / gridSteps;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(canvas.width - padding, y);
                ctx.stroke();
            }
            
            // Draw ESP32 positions (corners)
            ctx.fillStyle = '#ff6b6b';
            ctx.font = '12px Arial';
            
            const esp1 = toCanvasCoords(0, 0);
            ctx.fillRect(esp1.x - 5, esp1.y - 5, 10, 10);
            ctx.fillText('ESP1', esp1.x + 8, esp1.y + 15);
            
            const esp2 = toCanvasCoords(D1, 0);
            ctx.fillRect(esp2.x - 5, esp2.y - 5, 10, 10);
            ctx.fillText('ESP2', esp2.x - 35, esp2.y + 15);
            
            const esp3 = toCanvasCoords(0, D2);
            ctx.fillRect(esp3.x - 5, esp3.y - 5, 10, 10);
            ctx.fillText('ESP3', esp3.x + 8, esp3.y - 8);
            
            // Smooth animation
            if (animationProgress < 1) {
                animationProgress += 0.1;
                if (animationProgress > 1) animationProgress = 1;
                
                currentPos.l = currentPos.l + (targetPos.l - currentPos.l) * 0.1;
                currentPos.m = currentPos.m + (targetPos.m - currentPos.m) * 0.1;
            }
            
            const pos = toCanvasCoords(currentPos.l, currentPos.m);
            
            // Draw safety radius
            ctx.fillStyle = 'rgba(74, 144, 226, 0.15)';
            ctx.strokeStyle = 'rgba(74, 144, 226, 0.4)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, SAFETY_RADIUS * scale, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Draw person
            ctx.fillStyle = '#4a90e2';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 3, 0, 2 * Math.PI);
            ctx.fill();
            
            if (animationProgress < 1) {
                requestAnimationFrame(draw);
            }
        }
        
        draw();
        
        // WebSocket connection for BROWSER (receive-only)
        const socket = new WebSocket("ws://10.114.196.229:5000/ws/client");
        
        socket.onopen = () => {
            console.log('WebSocket connected');
            statusIndicator.className = 'status connected';
        };
        
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
            
            if (typeof data.l === 'number' && typeof data.m === 'number') {
                // Calculate Euclidean distance between current and new position
                const dx = data.l - currentPos.l;
                const dy = data.m - currentPos.m;
                const distance = Math.sqrt(dx * dx + dy * dy);

                // Only update if new position is outside the safety radius
                if (distance > SAFETY_RADIUS) {
                    targetPos.l = Math.max(0, Math.min(D1, data.l));
                    targetPos.m = Math.max(0, Math.min(D2, data.m));
                    animationProgress = 0;
                    requestAnimationFrame(draw);
                } else {
                    console.log("Point within safety radius — not moving");
                }
            }
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            statusIndicator.className = 'status disconnected';
        };
        
        socket.onclose = () => {
            console.log('WebSocket connection closed');
            statusIndicator.className = 'status disconnected';
            // Auto-reconnect after 3 seconds
            setTimeout(() => {
                window.location.reload();
            }, 3000);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, d1=D1, d2=D2, safety_radius=SAFETY_RADIUS)

# ------------------ WebSockets -----------------------
@sock.route('/ws')
def ws_esp(ws):
    print("ESP32 connected.")
    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            data = json.loads(msg)
            print(f"Received from ESP32: {data}")
            handle_rssi_data(data)
    except Exception as e:
        print(f"ESP32 WebSocket error: {e}")
    finally:
        print("ESP32 disconnected.")

@sock.route('/ws/client')
def ws_client(ws):
    browser_clients.append(ws)
    print(f"Browser client connected. Total clients: {len(browser_clients)}")
    try:
        while True:
            msg = ws.receive(timeout=60)
            if msg is None:
                break
    except Exception as e:
        print(f"Browser client error: {e}")
    finally:
        if ws in browser_clients:
            browser_clients.remove(ws)
        print(f"Browser client disconnected. Remaining: {len(browser_clients)}")

# ------------------ Handle ESP32 Data -----------------
def handle_rssi_data(data):
    current_time = time()
    for key in ["rssi1", "rssi2", "rssi3"]:
        if key in data:
            esp_data[key]["value"].append(float(data[key]))
            esp_data[key]["timestamp"] = current_time

    if all(esp_data[k]["value"] and current_time - esp_data[k]["timestamp"] < SYNC_TIMEOUT for k in esp_data):
        r1 = esp_data["rssi1"]["value"][-1]
        r2 = esp_data["rssi2"]["value"][-1]
        r3 = esp_data["rssi3"]["value"][-1]

        # Step 1: Median + exponential smoothing for RSSI
        filtered_x = smooth_x.update(ma_filter_x.update(r1))
        filtered_y = smooth_y.update(ma_filter_y.update(r2))
        filtered_z = smooth_z.update(ma_filter_z.update(r3))

        # Step 2: Convert to distance
        x = rssi_to_distance(filtered_x)
        y = rssi_to_distance(filtered_y)
        z = rssi_to_distance(filtered_z)

        # Step 3: Calculate position
        l, m, resid, success = estimate_position_from_rssi(x, y, z, D1, D2, room_margin=0.1)

        # Step 4: Smooth coordinates
        l = coord_smooth_l.update(l)
        m = coord_smooth_m.update(m)

        payload = {
            "rssi1": round(r1, 2), "rssi2": round(r2, 2), "rssi3": round(r3, 2),
            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
            "l": round(l, 2), "m": round(m, 2)
        }

        print(f"Broadcasting to {len(browser_clients)} clients: {payload}")
        for client in browser_clients[:]:
            try:
                client.send(json.dumps(payload))
            except Exception as e:
                print(f"Failed to send to client: {e}")
                if client in browser_clients:
                    browser_clients.remove(client)
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

        payload = {
            "rssi1": r1, "rssi2": r2, "rssi3": r3,
        }

        print(f"Broadcasting to {len(browser_clients)} clients: {payload}")
        for client in browser_clients[:]:
            try:
                client.send(json.dumps(payload))
            except Exception as e:
                print(f"Failed to send to client: {e}")
                if client in browser_clients:
                    browser_clients.remove(client)
                    
        print("Data not in sync or missing; skipping computation.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
