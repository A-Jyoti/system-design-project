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

# -----------------coordinate calculation ----------------

import numpy as np

def sanitize_distances(distances, room_diag, min_d=0.05):
    """Clamp distances to reasonable positive range to avoid zeros/negatives.
       room_diag: approximate maximum distance in room (e.g., sqrt(D1^2 + D2^2)).
    """
    distances = np.array(distances, dtype=float)
    max_d = room_diag * 1.2
    distances = np.clip(distances, min_d, max_d)
    return distances

def multilateration_least_squares(positions, distances, weights=None):
    """
    Linearized least-squares multilateration (works with 3 or more receivers).
    positions: Nx2 array-like of receiver coordinates [(x1,y1),(x2,y2),...]
    distances: length-N array of measured distances (meters)
    weights: optional length-(N-1) weights for the linear system (or None)
    Returns: (px, py, residual_norm, success_flag)
    """
    p = np.asarray(positions, dtype=float)
    r = np.asarray(distances, dtype=float)
    N = p.shape[0]
    if N < 3:
        raise ValueError("Need at least 3 receivers for 2D multilateration")

    # Reference receiver (first)
    x0, y0 = p[0]
    r0 = r[0]

    # Build linear system: A * [px,py] = b
    A = []
    b = []
    for i in range(1, N):
        xi, yi = p[i]
        ri = r[i]
        Ai = [2*(xi - x0), 2*(yi - y0)]
        bi = (r0**2 - ri**2) - (x0**2 - xi**2) - (y0**2 - yi**2)
        A.append(Ai)
        b.append(bi)
    A = np.array(A, dtype=float)  # (N-1) x 2
    b = np.array(b, dtype=float)  # (N-1)

    # Apply weights if provided
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != A.shape[0]:
            raise ValueError("weights must match number of rows (N-1)")
        W = np.sqrt(w)[:,None]  # scale rows by sqrt(weight)
        A = A * W
        b = b * W.ravel()

    # Solve least squares
    sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    px, py = sol[0], sol[1]
    # compute residual norm in terms of distance error
    est_dists = np.sqrt((p[:,0]-px)**2 + (p[:,1]-py)**2)
    residuals_vec = est_dists - r
    resid_norm = np.linalg.norm(residuals_vec)

    # success flag: if rank < 2, it's degenerate
    success = (rank >= 2)
    return (px, py, resid_norm, success)

def coarse_grid_search(positions, distances, room_bounds, grid_res=0.1):
    """
    Fallback brute-force search over grid to find minimal residual.
    room_bounds: (xmin,xmax,ymin,ymax)
    grid_res: spacing in meters
    Returns best (px,py,resid)
    """
    xmin, xmax, ymin, ymax = room_bounds
    xs = np.arange(xmin, xmax + 1e-9, grid_res)
    ys = np.arange(ymin, ymax + 1e-9, grid_res)
    P = np.array(np.meshgrid(xs, ys)).T.reshape(-1,2)
    p = np.asarray(positions)
    r = np.asarray(distances)
    best = (None, None, np.inf)
    for px,py in P:
        d = np.sqrt((p[:,0]-px)**2 + (p[:,1]-py)**2)
        resid = np.linalg.norm(d - r)
        if resid < best[2]:
            best = (px, py, resid)
    return best

# -------------------------
# Example usage for your 3-corner setup:
# Assume receivers at corners: R1=(0,0), R2=(D1,0), R3=(0,D2)
# r1,r2,r3 are distances computed from filtered RSSI (in meters)
# -------------------------

def estimate_position_from_rssi(r1, r2, r3, D1, D2, room_margin=0.1):
    # receiver positions
    positions = [(0.0, 0.0), (D1, 0.0), (0.0, D2)]
    # sanitize and clamp distances
    room_diag = np.sqrt(D1**2 + D2**2)
    dists = sanitize_distances([r1, r2, r3], room_diag)

    # optional: compute simple weights from variance/stability if available
    # For now, equal weights:
    weights = None

    px, py, resid_norm, success = multilateration_least_squares(positions, dists, weights)
    if not success or np.isnan(px) or np.isnan(py):
        # fallback to coarse grid search over room
        xmin, xmax, ymin, ymax = -room_margin, D1 + room_margin, -room_margin, D2 + room_margin
        bx, by, bres = coarse_grid_search(positions, dists, (xmin, xmax, ymin, ymax), grid_res=0.05)
        return bx, by, bres, False

    # if residual very large (meaning distances inconsistent), try grid fallback too
    if resid_norm > max(0.5, 0.1 * room_diag):  # tune threshold as needed
        xmin, xmax, ymin, ymax = -room_margin, D1 + room_margin, -room_margin, D2 + room_margin
        bx, by, bres = coarse_grid_search(positions, dists, (xmin, xmax, ymin, ymax), grid_res=0.05)
        # if grid gives better residual, use it
        if bres < resid_norm:
            return bx, by, bres, False

    return px, py, resid_norm, True


# ---------------------------------------------------------

# ------------------ Moving Average -------------------
# class MovingAverageFilter:
#     def __init__(self, window_size=5):
#         self.window_size = window_size
#         self.window = []

#     def update(self, value):
#         self.window.append(value)
#         if len(self.window) > self.window_size:
#             self.window.pop(0)
#         return sum(self.window) / len(self.window)

#------------------ Median Filter ----------------------
import statistics

class MedianFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.window = []

    def update(self, value):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        # Return median of the current window
        return statistics.median(self.window)
# ------------------ Shared Data ----------------------
esp_data = {
    "rssi1": {"value": [], "timestamp": 0},
    "rssi2": {"value": [], "timestamp": 0},
    "rssi3": {"value": [], "timestamp": 0},
}

clients = []  # connected web browsers
SYNC_TIMEOUT = 2  # seconds

# Room dimensions (you can modify these)
D1 = 5.0  # meters
D2 = 4.0  # meters
SAFETY_RADIUS = 0.02  # meters - the safety bubble around the person

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
    </style>
</head>
<body>
    <h1>🎯 ESP32 Position Tracker</h1>
    
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
        
        // Calculate scale to fit room in canvas with padding
        const padding = 40;
        const scaleX = (canvas.width - 2 * padding) / D1;
        const scaleY = (canvas.height - 2 * padding) / D2;
        const scale = Math.min(scaleX, scaleY);
        
        // Current and target positions for smooth animation
        let currentPos = { l: D1/2, m: D2/2 };
        let targetPos = { l: D1/2, m: D2/2 };
        let animationProgress = 1; // 0 to 1
        
        // Convert room coordinates to canvas coordinates
        function toCanvasCoords(l, m) {
            return {
                x: padding + l * scale,
                y: canvas.height - padding - m * scale
            };
        }
        
        // Draw the room and person
        function draw() {
            // Clear canvas
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
            
            // ESP1 at (0,0)
            const esp1 = toCanvasCoords(0, 0);
            ctx.fillRect(esp1.x - 5, esp1.y - 5, 10, 10);
            ctx.fillText('ESP1', esp1.x + 8, esp1.y + 15);
            
            // ESP2 at (D1, 0)
            const esp2 = toCanvasCoords(D1, 0);
            ctx.fillRect(esp2.x - 5, esp2.y - 5, 10, 10);
            ctx.fillText('ESP2', esp2.x - 35, esp2.y + 15);
            
            // ESP3 at (0, D2)
            const esp3 = toCanvasCoords(D1, D2);
            ctx.fillRect(esp3.x - 5, esp3.y - 5, 10, 10);
            ctx.fillText('ESP3', esp3.x + 8, esp3.y - 8);
            
            // Smooth animation
            if (animationProgress < 1) {
                animationProgress += 0.1; // Adjust speed here
                if (animationProgress > 1) animationProgress = 1;
                
                currentPos.l = currentPos.l + (targetPos.l - currentPos.l) * 0.1;
                currentPos.m = currentPos.m + (targetPos.m - currentPos.m) * 0.1;
            }
            
            const pos = toCanvasCoords(currentPos.l, currentPos.m);
            
            // Draw safety radius (semi-transparent)
            ctx.fillStyle = 'rgba(74, 144, 226, 0.15)';
            ctx.strokeStyle = 'rgba(74, 144, 226, 0.4)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, SAFETY_RADIUS * scale, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Draw person (blue dot)
            ctx.fillStyle = '#4a90e2';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 8, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw white center
            ctx.fillStyle = 'white';
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, 3, 0, 2 * Math.PI);
            ctx.fill();
            
            // Request next frame if still animating
            if (animationProgress < 1) {
                requestAnimationFrame(draw);
            }
        }
        
        // Initial draw
        draw();
        
        // WebSocket connection
        const socket = new WebSocket("ws://" + window.location.host + "/ws");
        
        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Update text displays
            document.getElementById("r1").textContent = data.rssi1;
            document.getElementById("r2").textContent = data.rssi2;
            document.getElementById("r3").textContent = data.rssi3;
            document.getElementById("x").textContent = data.x;
            document.getElementById("y").textContent = data.y;
            document.getElementById("z").textContent = data.z;
            document.getElementById("l").textContent = data.l;
            document.getElementById("m").textContent = data.m;
            
            // Update position if valid
            if (typeof data.l === 'number' && typeof data.m === 'number') {
                targetPos.l = Math.max(0, Math.min(D1, data.l));
                targetPos.m = Math.max(0, Math.min(D2, data.m));
                animationProgress = 0;
                requestAnimationFrame(draw);
            }
        };
        
        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        socket.onclose = () => {
            console.log('WebSocket connection closed');
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        HTML_PAGE, 
        d1=D1, 
        d2=D2, 
        safety_radius=SAFETY_RADIUS
    )

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
        print("Client disconnected.")

# ------------------ Handle ESP32 data -----------------
def handle_rssi_data(data):
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

        # ma_filter = MovingAverageFilter()
        # filtered_x = ma_filter.update(r1)
        # x = rssi_to_distance(filtered_x)
        # filtered_y = ma_filter.update(r2)
        # y = rssi_to_distance(filtered_y)
        # filtered_z = ma_filter.update(r3)
        # z = rssi_to_distance(filtered_z)
        
        ma_filter_x = MedianFilter()
        ma_filter_y = MedianFilter()
        ma_filter_z = MedianFilter()

        filtered_x = ma_filter_x.update(r1)
        x = rssi_to_distance(filtered_x)
        
        filtered_y = ma_filter_y.update(r2)
        y = rssi_to_distance(filtered_y)

        filtered_z = ma_filter_z.update(r3)
        z = rssi_to_distance(filtered_z)
        
        l, m, resid, success = estimate_position_from_rssi(x, y, z, D1, D2, room_margin=0.1)
        payload = {
            "rssi1": r1, "rssi2": r2, "rssi3": r3,
            "x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
            "l": l if isinstance(l, str) else round(l, 2),
            "m": m if isinstance(m, str) else round(m, 2)
        }

        # Broadcast to all connected browsers
        for c in clients[:]:
            try:
                c.send(json.dumps(payload))
            except:
                clients.remove(c)
    else:
        print("Data not in sync or missing; skipping computation.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
