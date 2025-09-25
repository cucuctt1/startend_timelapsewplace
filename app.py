import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import time
import threading
from flask import Flask, send_file, render_template_string
import logging
from datetime import datetime

import os

# ---------------- LOGGING SETUP ----------------
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Store logs in memory for web display
log_entries = []
MAX_LOG_ENTRIES = 100

class LogHandler(logging.Handler):
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage()
        }
        log_entries.append(log_entry)
        # Keep only the latest entries
        if len(log_entries) > MAX_LOG_ENTRIES:
            log_entries.pop(0)

# Add custom handler to capture logs
log_handler = LogHandler()
logger.addHandler(log_handler)

def get_coords(env_name):
    val = os.getenv(env_name)
    if not val:
        raise ValueError(f"Missing env var: {env_name}")
    return tuple(map(int, val.split(",")))

cord1 = get_coords("CORD1")
cord2 = get_coords("CORD2")


# ---------------- CONFIG ----------------
TILE_SIZE = 1000
URL = "https://backend.wplace.live/files/s0/tiles/{}/{}.png"

TstartX, TstartY = cord1[0], cord1[1]
TendX, TendY = cord2[0], cord2[1]

numx = TendX - TstartX + 1
numy = TendY - TstartY + 1

output_file = "output.mp4"
interval = int(os.getenv("INTERVAL", 1800))  # default 30 min
fps = int(os.getenv("FPS", 2))     

# ---------------- IMAGE FUNCTIONS ----------------
def merge_chunks(chunks, rows, cols, chunk_h, chunk_w):
    decoded = []
    for c in chunks:
        img = Image.open(c).convert("RGBA")
        decoded.append(np.array(img))

    full_img = np.zeros((rows * chunk_h, cols * chunk_w, 4), dtype=np.uint8)
    for idx, chunk in enumerate(decoded):
        r = idx // cols
        c = idx % cols
        full_img[r*chunk_h:(r+1)*chunk_h, c*chunk_w:(c+1)*chunk_w, :] = chunk
    return Image.fromarray(full_img, mode="RGBA")

def build_snapshot():
    logger.info(f"Starting snapshot build - downloading {numx}x{numy} tiles")
    data_array = []
    failed_tiles = 0
    
    for y in range(TstartY, TendY+1):
        for x in range(TstartX, TendX+1):
            url = URL.format(x, y)
            print(f"Downloading tile {x},{y}")
            res = requests.get(url)
            if res.status_code == 200:
                data_array.append(BytesIO(res.content))
            else:
                failed_tiles += 1
                logger.warning(f"Failed to download tile {x},{y} - status code: {res.status_code}")
                # Create a placeholder image as BytesIO instead of PIL Image
                placeholder = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0,0,0,0))
                placeholder_bytes = BytesIO()
                placeholder.save(placeholder_bytes, format='PNG')
                placeholder_bytes.seek(0)
                data_array.append(placeholder_bytes)

    if failed_tiles > 0:
        logger.warning(f"Snapshot completed with {failed_tiles} failed tiles out of {numx * numy}")
    else:
        logger.info("All tiles downloaded successfully")
        
    final_image = merge_chunks(data_array, numy, numx, TILE_SIZE, TILE_SIZE)

    # Crop
    crop_x1, crop_y1 = cord1[2], cord1[3]
    crop_x2 = (numx * TILE_SIZE) - (1000 - cord2[2])
    crop_y2 = (numy * TILE_SIZE) - (1000 - cord2[3])
    crop_image = final_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    return crop_image

def rgba_to_rgb_with_bg(img, bg_color=(158, 189, 255)):
    """Convert RGBA → RGB with background color for transparent pixels."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    background = Image.new("RGB", img.size, bg_color)
    background.paste(img, mask=img.split()[3])  # use alpha channel
    return background

# ---------------- VIDEO LOOP ----------------
def run_video_loop():
    frame_size = None
    video = None
    frame_count = 0

    logger.info(f"Starting timelapse video loop - interval: {interval}s, fps: {fps}")
    
    try:
        while True:
            logger.info("Capturing new frame for timelapse")
            snapshot = build_snapshot()

            # Convert RGBA → RGB with background color #9EBDFF
            rgb_image = rgba_to_rgb_with_bg(snapshot, (158, 189, 255))

            # Convert RGB → BGR for OpenCV
            frame = np.array(rgb_image)[:, :, ::-1]

            if frame_size is None:
                height, width, _ = frame.shape
                frame_size = (width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(output_file, fourcc, fps, frame_size)
                logger.info(f"Initialized video writer - resolution: {width}x{height}, fps: {fps}")

            video.write(frame)
            frame_count += 1
            logger.info(f"✅ Added frame #{frame_count} to {output_file}")
            print(f"✅ Added frame to {output_file}")

            logger.info(f"Waiting {interval} seconds until next frame")
            time.sleep(interval)  # wait for the specified interval
    except Exception as e:
        logger.error(f"Error in video loop: {str(e)}")
    finally:
        if video:
            video.release()
            logger.info(f"🎬 Video finalized with {frame_count} frames")
            print("🎬 Video finalized.")

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/")
def main_page():
    """Main page that renders logs when entering the website"""
    logger.info("User accessed main page - rendering logs")
    
    # HTML template for displaying logs
    template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Timelapse Generator - Logs</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 8px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                padding: 30px; 
            }
            h1 { 
                color: #333; 
                border-bottom: 3px solid #007acc; 
                padding-bottom: 10px; 
            }
            .status { 
                background: #e7f3ff; 
                border: 1px solid #b3d9ff; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 20px 0; 
            }
            .logs { 
                background: #2d2d30; 
                color: #cccccc; 
                padding: 20px; 
                border-radius: 5px; 
                max-height: 600px; 
                overflow-y: auto; 
                font-family: 'Consolas', 'Monaco', monospace; 
                font-size: 14px; 
            }
            .log-entry { 
                margin-bottom: 8px; 
                padding: 5px; 
                border-radius: 3px; 
            }
            .log-info { background-color: rgba(0, 122, 204, 0.1); }
            .log-warning { background-color: rgba(255, 193, 7, 0.2); color: #fff3cd; }
            .log-error { background-color: rgba(220, 53, 69, 0.2); color: #f8d7da; }
            .timestamp { color: #569cd6; }
            .level { font-weight: bold; }
            .level-INFO { color: #4ec9b0; }
            .level-WARNING { color: #ffcc02; }
            .level-ERROR { color: #f14c4c; }
            .controls { 
                margin: 20px 0; 
                text-align: center; 
            }
            .btn { 
                background: #007acc; 
                color: white; 
                border: none; 
                padding: 10px 20px; 
                border-radius: 5px; 
                text-decoration: none; 
                display: inline-block; 
                margin: 0 10px; 
                cursor: pointer; 
            }
            .btn:hover { background: #005a9e; }
            .refresh-note { 
                color: #666; 
                font-style: italic; 
                margin-top: 10px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Timelapse Generator</h1>
            
            <div class="status">
                <h3>System Status</h3>
                <p><strong>Configuration:</strong></p>
                <ul>
                    <li>Tile coordinates: ({{ cord1[0] }},{{ cord1[1] }}) to ({{ cord2[0] }},{{ cord2[1] }})</li>
                    <li>Grid size: {{ numx }} × {{ numy }} tiles</li>
                    <li>Capture interval: {{ interval }} seconds ({{ interval_min }} minutes)</li>
                    <li>Video FPS: {{ fps }}</li>
                    <li>Total frames captured: Available in logs</li>
                </ul>
            </div>
            
            <div class="controls">
                <a href="/download" class="btn">📥 Download Video</a>
                <button onclick="location.reload()" class="btn">🔄 Refresh Logs</button>
            </div>
            
            <h3>📋 Application Logs</h3>
            <div class="logs">
                {% if log_entries %}
                    {% for log in log_entries %}
                    <div class="log-entry log-{{ log.level|lower }}">
                        <span class="timestamp">{{ log.timestamp }}</span> - 
                        <span class="level level-{{ log.level }}">{{ log.level }}</span> - 
                        <span class="message">{{ log.message }}</span>
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="log-entry">No logs available yet. The system may still be starting up.</div>
                {% endif %}
            </div>
            
            <div class="refresh-note">
                Logs are updated in real-time. Refresh the page to see the latest entries.
                Showing the last {{ log_entries|length }} entries (max {{ max_entries }}).
            </div>
        </div>
        
        <script>
            // Auto-scroll to bottom of logs
            const logsContainer = document.querySelector('.logs');
            logsContainer.scrollTop = logsContainer.scrollHeight;
            
            // Auto-refresh every 30 seconds
            setTimeout(function() {
                location.reload();
            }, 30000);
        </script>
    </body>
    </html>
    """
    
    return render_template_string(template, 
                                log_entries=log_entries,
                                max_entries=MAX_LOG_ENTRIES,
                                cord1=cord1,
                                cord2=cord2,
                                numx=numx,
                                numy=numy,
                                interval=interval,
                                interval_min=interval // 60,
                                fps=fps)

@app.route("/download")
def download_video():
    """Download the generated timelapse video"""
    logger.info("User requested video download")
    try:
        return send_file(output_file, as_attachment=True)
    except FileNotFoundError:
        logger.error(f"Video file {output_file} not found")
        return "Video file not found. The timelapse may still be generating.", 404

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🚀 Starting Timelapse Generator Application")
    logger.info(f"Configuration loaded:")
    logger.info(f"  - Coordinates: ({cord1[0]},{cord1[1]}) to ({cord2[0]},{cord2[1]})")
    logger.info(f"  - Grid size: {numx} × {numy} tiles")
    logger.info(f"  - Capture interval: {interval} seconds ({interval // 60} minutes)")
    logger.info(f"  - Video FPS: {fps}")
    logger.info(f"  - Output file: {output_file}")
    logger.info("=" * 50)
    
    # Start video loop in background
    logger.info("Starting background timelapse video generation thread")
    threading.Thread(target=run_video_loop, daemon=True).start()
    
    # Start Flask server
    logger.info("Starting Flask web server on http://0.0.0.0:10000")
    logger.info("Visit the website to view real-time logs and download the timelapse video")
    app.run(host="0.0.0.0", port=10000)
