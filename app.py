import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import time
import threading
from flask import Flask, send_file

import os

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
interval = 1800  # 30 minutes
fps = 8

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
    data_array = []
    for y in range(TstartY, TendY+1):
        for x in range(TstartX, TendX+1):
            url = URL.format(x, y)
            print(f"Downloading tile {x},{y}")
            res = requests.get(url)
            if res.status_code == 200:
                data_array.append(BytesIO(res.content))
            else:
                # Create a placeholder image as BytesIO instead of PIL Image
                placeholder = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0,0,0,0))
                placeholder_bytes = BytesIO()
                placeholder.save(placeholder_bytes, format='PNG')
                placeholder_bytes.seek(0)
                data_array.append(placeholder_bytes)

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

    try:
        while True:
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

            video.write(frame)
            print(f"✅ Added frame to {output_file}")

            time.sleep(interval)  # wait 30 minutes
    finally:
        if video:
            video.release()
            print("🎬 Video finalized.")

# ---------------- FLASK APP ----------------
app = Flask(__name__)

@app.route("/download")
def download_video():
    return send_file(output_file, as_attachment=True)

if __name__ == "__main__":
    # Start video loop in background
    threading.Thread(target=run_video_loop, daemon=True).start()
    # Start Flask server
    app.run(host="0.0.0.0", port=10000)
