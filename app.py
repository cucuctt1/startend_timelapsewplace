import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import time
import threading
from flask import Flask, send_file, render_template_string, jsonify
from flask_sqlalchemy import SQLAlchemy
import logging
from datetime import datetime
import base64
import json

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

# ---------------- DATABASE SETUP ----------------
# Initialize Flask app early for database setup
app = Flask(__name__)

# Database configuration for PostgreSQL (Render compatible)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///timelapse.db')
# Fix for Render PostgreSQL URL format
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

try:
    db = SQLAlchemy(app)
    logger.info(f"Database connected: {DATABASE_URL.split('@')[0].split('//')[1] if '@' in DATABASE_URL else 'SQLite'}")
    
    # Database Models
    class FrameData(db.Model):
        __tablename__ = 'frame_data'
        
        id = db.Column(db.Integer, primary_key=True)
        timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        frame_number = db.Column(db.Integer, unique=True, nullable=False, index=True)
        image_data = db.Column(db.LargeBinary, nullable=False)  # Compressed PNG data
        width = db.Column(db.Integer, nullable=False)
        height = db.Column(db.Integer, nullable=False)
        file_size = db.Column(db.Integer, nullable=False)  # Size in bytes
        created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        
        def __repr__(self):
            return f'<FrameData {self.frame_number} at {self.timestamp}>'

    class VideoMetadata(db.Model):
        __tablename__ = 'video_metadata'
        
        id = db.Column(db.Integer, primary_key=True)
        video_name = db.Column(db.String(255), nullable=False)
        total_frames = db.Column(db.Integer, default=0)
        fps = db.Column(db.Integer, nullable=False)
        width = db.Column(db.Integer, nullable=False)
        height = db.Column(db.Integer, nullable=False)
        interval_seconds = db.Column(db.Integer, nullable=False)
        created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
        updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
        is_complete = db.Column(db.Boolean, default=False)
        file_path = db.Column(db.String(500))
        
        def __repr__(self):
            return f'<VideoMetadata {self.video_name} - {self.total_frames} frames>'
    
except Exception as e:
    logger.error(f"Database setup failed: {e}")
    db = None
    FrameData = None
    VideoMetadata = None

# Database helper functions
def save_frame_to_db(frame_image, frame_number):
    """Save frame image to database as compressed PNG"""
    try:
        if db is None or FrameData is None:
            logger.warning("Database not available, skipping frame save")
            return None
            
        # Convert frame to PNG bytes
        img_buffer = BytesIO()
        frame_pil = Image.fromarray(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        frame_pil.save(img_buffer, format='PNG', optimize=True, compress_level=6)
        img_data = img_buffer.getvalue()
        
        height, width = frame_image.shape[:2]
        
        # Check if frame already exists
        existing_frame = FrameData.query.filter_by(frame_number=frame_number).first()
        if existing_frame:
            logger.warning(f"Frame {frame_number} already exists in database, skipping")
            return existing_frame
        
        # Create new frame record
        frame_record = FrameData(
            frame_number=frame_number,
            image_data=img_data,
            width=width,
            height=height,
            file_size=len(img_data),
            timestamp=datetime.utcnow()
        )
        
        db.session.add(frame_record)
        db.session.commit()
        
        logger.info(f"💾 Saved frame {frame_number} to database ({len(img_data) // 1024} KB)")
        return frame_record
        
    except Exception as e:
        logger.error(f"Failed to save frame {frame_number} to database: {e}")
        try:
            if db:
                db.session.rollback()
        except:
            pass  # Ignore rollback errors
        return None

def get_frame_from_db(frame_number):
    """Retrieve frame image from database"""
    try:
        if db is None or FrameData is None:
            return None
            
        frame_record = FrameData.query.filter_by(frame_number=frame_number).first()
        if not frame_record:
            return None
        
        # Convert PNG bytes back to OpenCV format
        img_buffer = BytesIO(frame_record.image_data)
        pil_image = Image.open(img_buffer)
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return frame
    except Exception as e:
        logger.error(f"Failed to retrieve frame {frame_number} from database: {e}")
        return None

def get_all_frames_count():
    """Get total number of frames in database"""
    try:
        if db is None or FrameData is None:
            return 0
        return FrameData.query.count()
    except Exception as e:
        logger.error(f"Failed to count frames: {e}")
        return 0

def create_video_from_db_frames(output_filename=None):
    """Create video from all frames stored in database"""
    if not output_filename:
        output_filename = f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        if db is None or FrameData is None:
            logger.warning("Database not available, cannot create video from frames")
            return None
            
        # Get all frames ordered by frame number
        frames = FrameData.query.order_by(FrameData.frame_number).all()
        
        if not frames:
            logger.warning("No frames found in database to create video")
            return None
        
        logger.info(f"🎬 Creating video from {len(frames)} database frames")
        
        # Get video parameters from first frame
        first_frame_img = get_frame_from_db(frames[0].frame_number)
        if first_frame_img is None:
            logger.error("Failed to load first frame from database")
            return None
        
        height, width = first_frame_img.shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        # Write all frames to video
        for i, frame_record in enumerate(frames):
            frame_img = get_frame_from_db(frame_record.frame_number)
            if frame_img is not None:
                video_writer.write(frame_img)
                if (i + 1) % 10 == 0:
                    logger.info(f"📹 Processed frame {i + 1}/{len(frames)}")
            else:
                logger.warning(f"Skipped corrupted frame {frame_record.frame_number}")
        
        video_writer.release()
        
        # Update video metadata if available
        if VideoMetadata is not None:
            try:
                video_meta = VideoMetadata.query.first()
                if not video_meta:
                    video_meta = VideoMetadata(
                        video_name=output_filename,
                        total_frames=len(frames),
                        fps=fps,
                        width=width,
                        height=height,
                        interval_seconds=interval,
                        file_path=output_filename,
                        is_complete=True
                    )
                    db.session.add(video_meta)
                else:
                    video_meta.total_frames = len(frames)
                    video_meta.updated_at = datetime.utcnow()
                    video_meta.is_complete = True
                    video_meta.file_path = output_filename
                
                db.session.commit()
            except Exception as meta_error:
                logger.warning(f"Failed to update video metadata: {meta_error}")
        
        logger.info(f"🎉 Video created successfully: {output_filename}")
        return output_filename
        
    except Exception as e:
        logger.error(f"Failed to create video from database frames: {e}")
        return None

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
            res = requests.get(url)
            if res.status_code == 200:
                data_array.append(BytesIO(res.content))
            else:
                failed_tiles += 1
                logger.warning(f"Failed to download tile - status code: {res.status_code}")
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
    
    # Initialize frame count with app context
    with app.app_context():
        try:
            frame_count = get_all_frames_count()  # Resume from database count
        except Exception as e:
            logger.warning(f"Could not get frame count from database: {e}")
            frame_count = 0

    logger.info(f"Starting timelapse video loop - interval: {interval}s, fps: {fps}")
    logger.info(f"Resuming from frame count: {frame_count}")
    
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
                logger.info(f"Frame dimensions: {width}x{height}, fps: {fps}")

            # Save frame to database (primary storage) with app context
            frame_count += 1
            with app.app_context():
                try:
                    saved_frame = save_frame_to_db(frame, frame_count)
                    
                    if saved_frame:
                        logger.info(f"✅ Saved frame #{frame_count} to database")
                        print(f"✅ Added frame #{frame_count} to database")
                        
                        # Optionally create/update video file every N frames or on schedule
                        if frame_count % 10 == 0:  # Every 10 frames, recreate video
                            logger.info("Recreating video from database frames...")
                            video_file = create_video_from_db_frames(output_file)
                            if video_file:
                                logger.info(f"📹 Video updated: {video_file}")
                    else:
                        logger.error(f"Failed to save frame #{frame_count}")
                except Exception as db_error:
                    logger.error(f"Database error for frame #{frame_count}: {db_error}")
                    # Continue without database if there's an error
                    logger.info(f"✅ Captured frame #{frame_count} (database unavailable)")
                    print(f"✅ Captured frame #{frame_count} (database unavailable)")

            logger.info(f"Waiting {interval} seconds until next frame")
            time.sleep(interval)  # wait for the specified interval
            
    except Exception as e:
        logger.error(f"Error in video loop: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
    finally:
        logger.info(f"🎬 Video loop stopped after {frame_count} frames")
        print("🎬 Video loop finalized.")

# ---------------- FLASK APP ROUTES ----------------

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
                    <li>Grid size: {{ numx }} × {{ numy }} tiles</li>
                    <li>Capture interval: {{ interval }} seconds ({{ interval_min }} minutes)</li>
                    <li>Video FPS: {{ fps }}</li>
                    <li>Database: {{ database_type }}</li>
                </ul>
                
                <p><strong>Database Statistics:</strong></p>
                <ul>
                    <li>Total frames stored: <span id="total-frames">{{ db_stats.total_frames }}</span></li>
                    <li>Latest frame: #<span id="latest-frame">{{ db_stats.latest_frame }}</span></li>
                    <li>Database size: <span id="db-size">{{ db_stats.db_size_mb }}</span> MB</li>
                    <li>Video status: <span id="video-status">{{ "✅ Ready" if db_stats.video_exists else "⏳ Generating" }}</span></li>
                </ul>
            </div>
            
            <div class="controls">
                <a href="/download" class="btn">📥 Download Video</a>
                <button onclick="rebuildVideo()" class="btn">🔨 Rebuild Video</button>
                <button onclick="refreshStats()" class="btn">📊 Refresh Stats</button>
                <button onclick="location.reload()" class="btn">🔄 Refresh Logs</button>
            </div>
            
            <div id="rebuild-status" style="margin: 10px 0; padding: 10px; border-radius: 5px; display: none;"></div>
            
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
            
            // Refresh database statistics
            function refreshStats() {
                fetch('/api/stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('total-frames').textContent = data.total_frames;
                        document.getElementById('latest-frame').textContent = data.latest_frame_number;
                        document.getElementById('db-size').textContent = data.database_size_mb;
                        document.getElementById('video-status').textContent = data.video_exists ? '✅ Ready' : '⏳ Generating';
                    })
                    .catch(error => console.error('Error refreshing stats:', error));
            }
            
            // Rebuild video from database
            function rebuildVideo() {
                const statusDiv = document.getElementById('rebuild-status');
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#fff3cd';
                statusDiv.style.border = '1px solid #ffeaa7';
                statusDiv.style.color = '#856404';
                statusDiv.textContent = '🔨 Rebuilding video from database frames...';
                
                fetch('/api/rebuild_video')
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            statusDiv.style.background = '#d4edda';
                            statusDiv.style.border = '1px solid #c3e6cb';
                            statusDiv.style.color = '#155724';
                            statusDiv.textContent = `✅ ${data.message} (${data.total_frames} frames)`;
                            refreshStats();
                        } else {
                            statusDiv.style.background = '#f8d7da';
                            statusDiv.style.border = '1px solid #f5c6cb';
                            statusDiv.style.color = '#721c24';
                            statusDiv.textContent = `❌ Error: ${data.message}`;
                        }
                        
                        setTimeout(() => {
                            statusDiv.style.display = 'none';
                        }, 5000);
                    })
                    .catch(error => {
                        statusDiv.style.background = '#f8d7da';
                        statusDiv.style.border = '1px solid #f5c6cb';
                        statusDiv.style.color = '#721c24';
                        statusDiv.textContent = `❌ Network error: ${error.message}`;
                        
                        setTimeout(() => {
                            statusDiv.style.display = 'none';
                        }, 5000);
                    });
            }
            
            // Auto-refresh stats every 15 seconds
            setInterval(refreshStats, 15000);
            
            // Auto-refresh page every 60 seconds
            setTimeout(function() {
                location.reload();
            }, 60000);
        </script>
    </body>
    </html>
    """
    
    # Get database statistics
    try:
        db_stats = {
            'total_frames': get_all_frames_count(),
            'latest_frame': 0,
            'db_size_mb': get_database_size_mb(),
            'video_exists': os.path.exists(output_file)
        }
        
        if FrameData is not None:
            latest_frame = FrameData.query.order_by(FrameData.frame_number.desc()).first()
            if latest_frame:
                db_stats['latest_frame'] = latest_frame.frame_number
            
        database_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        db_stats = {
            'total_frames': 0,
            'latest_frame': 0,
            'db_size_mb': 0,
            'video_exists': os.path.exists(output_file)
        }
        database_type = "Unknown"
    
    return render_template_string(template, 
                                log_entries=log_entries,
                                max_entries=MAX_LOG_ENTRIES,
                                cord1=cord1,
                                cord2=cord2,
                                numx=numx,
                                numy=numy,
                                interval=interval,
                                interval_min=interval // 60,
                                fps=fps,
                                db_stats=db_stats,
                                database_type=database_type)

@app.route("/download")
def download_video():
    """Download the generated timelapse video"""
    logger.info("User requested video download")
    try:
        # First try to create fresh video from database
        fresh_video = create_video_from_db_frames(output_file)
        if fresh_video and os.path.exists(fresh_video):
            return send_file(fresh_video, as_attachment=True)
        elif os.path.exists(output_file):
            return send_file(output_file, as_attachment=True)
        else:
            return "Video file not found. The timelapse may still be generating.", 404
    except Exception as e:
        logger.error(f"Error serving video file: {e}")
        return f"Error generating video: {str(e)}", 500

@app.route("/api/stats")
def api_stats():
    """API endpoint for database statistics"""
    try:
        total_frames = get_all_frames_count()
        
        latest_frame = None
        if FrameData is not None:
            latest_frame = FrameData.query.order_by(FrameData.frame_number.desc()).first()
            
        video_meta = None
        if VideoMetadata is not None:
            video_meta = VideoMetadata.query.first()
        
        stats = {
            'total_frames': total_frames,
            'latest_frame_number': latest_frame.frame_number if latest_frame else 0,
            'latest_frame_time': latest_frame.timestamp.isoformat() if latest_frame else None,
            'database_size_mb': get_database_size_mb(),
            'video_exists': os.path.exists(output_file),
            'video_metadata': {
                'total_frames': video_meta.total_frames if video_meta else 0,
                'fps': video_meta.fps if video_meta else fps,
                'is_complete': video_meta.is_complete if video_meta else False,
                'last_updated': video_meta.updated_at.isoformat() if video_meta else None
            } if video_meta else None
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/api/rebuild_video")
def rebuild_video():
    """Force rebuild video from database frames"""
    logger.info("User requested video rebuild from database")
    try:
        video_file = create_video_from_db_frames(output_file)
        if video_file:
            return jsonify({
                'success': True, 
                'message': f'Video rebuilt successfully: {video_file}',
                'total_frames': get_all_frames_count()
            })
        else:
            return jsonify({'success': False, 'message': 'Failed to rebuild video'}), 500
    except Exception as e:
        logger.error(f"Error rebuilding video: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

def get_database_size_mb():
    """Get approximate database size in MB"""
    try:
        if db is None:
            return 0
            
        if 'sqlite' in DATABASE_URL.lower():
            db_path = DATABASE_URL.split(':///')[-1]
            if os.path.exists(db_path):
                return round(os.path.getsize(db_path) / (1024 * 1024), 2)
        else:
            # For PostgreSQL, estimate from frame data
            if FrameData is not None:
                total_size = db.session.query(db.func.sum(FrameData.file_size)).scalar() or 0
                return round(total_size / (1024 * 1024), 2)
        return 0
    except Exception:
        return 0

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🚀 Starting Timelapse Generator Application")
    logger.info(f"Configuration loaded:")
    logger.info(f"  - Grid size: {numx} × {numy} tiles")
    logger.info(f"  - Capture interval: {interval} seconds ({interval // 60} minutes)")
    logger.info(f"  - Video FPS: {fps}")
    logger.info(f"  - Output file: {output_file}")
    logger.info(f"  - Database: {DATABASE_URL.split('@')[0].split('//')[1] if '@' in DATABASE_URL else 'SQLite'}")
    logger.info("=" * 50)
    
    # Initialize database
    try:
        with app.app_context():
            db.create_all()
            existing_frames = get_all_frames_count()
            logger.info(f"📊 Database initialized - {existing_frames} existing frames found")
            
            # Create initial video metadata if not exists
            if VideoMetadata is not None:
                video_meta = VideoMetadata.query.first()
                if not video_meta:
                    video_meta = VideoMetadata(
                        video_name=output_file,
                        total_frames=existing_frames,
                        fps=fps,
                        width=0,
                        height=0,
                        interval_seconds=interval,
                        is_complete=False
                    )
                    db.session.add(video_meta)
                    db.session.commit()
                    logger.info("📝 Created initial video metadata")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    # Start video loop in background
    logger.info("Starting background timelapse video generation thread")
    threading.Thread(target=run_video_loop, daemon=True).start()
    
    # Start Flask server
    logger.info("Starting Flask web server on http://0.0.0.0:10000")
    logger.info("🌐 Visit the website to view real-time logs, database stats, and download videos")
    logger.info("🔄 Application will survive Render hosting restarts with database persistence")
    
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port)
