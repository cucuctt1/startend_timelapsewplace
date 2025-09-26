import requests
from PIL import Image
import numpy as np
from io import BytesIO
import cv2
import time
import threading
from flask import Flask, send_file, render_template_string, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import logging
from datetime import datetime
import base64
import json
import psycopg2
from psycopg2 import OperationalError
import sqlite3
from collections import OrderedDict
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
import queue

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

# ---------------- RAM STORAGE FALLBACK ----------------
class RAMFrameStorage:
    """In-memory frame storage as fallback when database is unavailable"""
    
    def __init__(self, max_frames=1000):
        self.frames = OrderedDict()  # frame_number -> {data, metadata}
        self.max_frames = max_frames
        self.total_size = 0
        self.lock = threading.Lock()
        
    def add_frame(self, frame_number, image_data, width, height):
        """Add frame to RAM storage with size management"""
        with self.lock:
            # Remove oldest frames if at capacity
            while len(self.frames) >= self.max_frames:
                oldest_frame = self.frames.popitem(last=False)
                self.total_size -= len(oldest_frame[1]['image_data'])
                logger.info(f"🗑️ Removed oldest frame #{oldest_frame[0]} from RAM storage")
            
            # Add new frame
            frame_data = {
                'image_data': image_data,
                'width': width,
                'height': height,
                'timestamp': datetime.utcnow(),
                'file_size': len(image_data)
            }
            
            self.frames[frame_number] = frame_data
            self.total_size += len(image_data)
            
            logger.info(f"💾 Stored frame #{frame_number} in RAM ({len(image_data) // 1024} KB)")
            return True
    
    def get_frame(self, frame_number):
        """Retrieve frame from RAM storage"""
        with self.lock:
            return self.frames.get(frame_number)
    
    def get_all_frames(self):
        """Get all frames in order"""
        with self.lock:
            return list(self.frames.items())
    
    def count_frames(self):
        """Get total frame count"""
        with self.lock:
            return len(self.frames)
    
    def get_size_mb(self):
        """Get total storage size in MB"""
        with self.lock:
            return round(self.total_size / (1024 * 1024), 2)
    
    def clear_old_frames(self, keep_last_n=100):
        """Clear old frames to free memory"""
        with self.lock:
            while len(self.frames) > keep_last_n:
                oldest_frame = self.frames.popitem(last=False)
                self.total_size -= len(oldest_frame[1]['image_data'])

# Global RAM storage instance
ram_storage = RAMFrameStorage(max_frames=500)  # Keep last 500 frames in RAM

# ---------------- DATABASE SETUP ----------------
# Initialize Flask app early for database setup
app = Flask(__name__)

# Database configuration with multiple fallback options
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///timelapse.db')
USE_RAM_STORAGE = False

# Fix for Render PostgreSQL URL format
if DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'connect_args': {
        'connect_timeout': 10,
        'options': '-c statement_timeout=30000'
    } if 'postgresql' in DATABASE_URL else {}
}

def test_database_connection():
    """Test database connection with multiple fallback strategies"""
    global USE_RAM_STORAGE
    
    try:
        # Test PostgreSQL connection directly
        if 'postgresql' in DATABASE_URL:
            logger.info("🔄 Testing PostgreSQL connection...")
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            conn.close()
            logger.info("✅ PostgreSQL connection successful")
            return True
            
        # Test SQLite connection
        elif 'sqlite' in DATABASE_URL:
            logger.info("🔄 Testing SQLite connection...")
            db_path = DATABASE_URL.split(':///')[-1]
            conn = sqlite3.connect(db_path)
            conn.close()
            logger.info("✅ SQLite connection successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.warning("🔄 Switching to RAM storage mode...")
        USE_RAM_STORAGE = True
        return False

# Test database connection
db_available = test_database_connection()

try:
    if db_available and not USE_RAM_STORAGE:
        db = SQLAlchemy(app)
        logger.info(f"✅ Database connected: {DATABASE_URL.split('@')[0].split('//')[1] if '@' in DATABASE_URL else 'SQLite'}")
        
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
        
        logger.info("📊 Database models initialized")
        
    else:
        logger.warning("⚠️ Database unavailable - using RAM storage")
        db = None
        FrameData = None
        VideoMetadata = None
        USE_RAM_STORAGE = True
    
except Exception as e:
    logger.error(f"❌ Database setup failed: {e}")
    logger.warning("🔄 Falling back to RAM storage")
    db = None
    FrameData = None
    VideoMetadata = None
    USE_RAM_STORAGE = True

# Database helper functions
def save_frame_to_storage(frame_image, frame_number):
    """Save frame to database or RAM storage with automatic fallback"""
    try:
        # Convert frame to PNG bytes
        img_buffer = BytesIO()
        frame_pil = Image.fromarray(cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB))
        frame_pil.save(img_buffer, format='PNG', optimize=True, compress_level=6)
        img_data = img_buffer.getvalue()
        
        height, width = frame_image.shape[:2]
        
        # Try database first if available
        if not USE_RAM_STORAGE and db is not None and FrameData is not None:
            try:
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
                
            except Exception as db_error:
                logger.error(f"Database save failed: {db_error}")
                logger.warning("🔄 Falling back to RAM storage for this frame")
                
                # Fallback to RAM storage
                if ram_storage.add_frame(frame_number, img_data, width, height):
                    return {'frame_number': frame_number, 'source': 'ram'}
        
        # Use RAM storage (either by design or as fallback)
        else:
            if ram_storage.add_frame(frame_number, img_data, width, height):
                return {'frame_number': frame_number, 'source': 'ram'}
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to save frame {frame_number}: {e}")
        return None

# Keep backward compatibility
def save_frame_to_db(frame_image, frame_number):
    """Backward compatibility wrapper"""
    return save_frame_to_storage(frame_image, frame_number)

def get_frame_from_storage(frame_number):
    """Retrieve frame from database or RAM storage"""
    try:
        # Try database first if available
        if not USE_RAM_STORAGE and db is not None and FrameData is not None:
            try:
                frame_record = FrameData.query.filter_by(frame_number=frame_number).first()
                if frame_record:
                    # Convert PNG bytes back to OpenCV format
                    img_buffer = BytesIO(frame_record.image_data)
                    pil_image = Image.open(img_buffer)
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    return frame
            except Exception as db_error:
                logger.warning(f"Database read failed: {db_error}, trying RAM storage")
        
        # Try RAM storage
        frame_data = ram_storage.get_frame(frame_number)
        if frame_data:
            img_buffer = BytesIO(frame_data['image_data'])
            pil_image = Image.open(img_buffer)
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame
        
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve frame {frame_number}: {e}")
        return None

def get_all_frames_count():
    """Get total number of frames from database or RAM storage"""
    try:
        # Try database first if available
        if not USE_RAM_STORAGE and db is not None and FrameData is not None:
            try:
                return FrameData.query.count()
            except Exception as db_error:
                logger.warning(f"Database count failed: {db_error}, using RAM storage count")
        
        # Use RAM storage count
        return ram_storage.count_frames()
    except Exception as e:
        logger.error(f"Failed to count frames: {e}")
        return 0

# Keep backward compatibility
def get_frame_from_db(frame_number):
    """Backward compatibility wrapper"""
    return get_frame_from_storage(frame_number)

class LazyVideoCreator:
    """Efficient video creator with lazy loading and streaming capabilities"""
    
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.frame_queue = queue.Queue(maxsize=batch_size * 2)
        self.processing = False
        
    def get_frame_iterator(self):
        """Get iterator for frames from storage with lazy loading"""
        if not USE_RAM_STORAGE and db is not None and FrameData is not None:
            try:
                # Database iterator with batching
                total_frames = FrameData.query.count()
                for offset in range(0, total_frames, self.batch_size):
                    batch = FrameData.query.order_by(FrameData.frame_number).offset(offset).limit(self.batch_size).all()
                    for frame_record in batch:
                        yield frame_record.frame_number, frame_record.image_data
                        
                    # Allow garbage collection of processed batch
                    gc.collect()
                return
            except Exception as e:
                logger.warning(f"Database iteration failed: {e}, using RAM storage")
        
        # RAM storage iterator
        for frame_number, frame_data in ram_storage.get_all_frames():
            yield frame_number, frame_data['image_data']
    
    def create_video_streaming(self, output_filename=None):
        """Create video with streaming and efficient memory usage"""
        if not output_filename:
            output_filename = f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        try:
            logger.info("🎬 Starting efficient video creation with lazy loading...")
            
            # Get frame count
            total_frames = get_all_frames_count()
            if total_frames == 0:
                logger.warning("No frames available for video creation")
                return None
            
            logger.info(f"📊 Processing {total_frames} frames in batches of {self.batch_size}")
            
            video_writer = None
            processed_count = 0
            
            # Process frames in batches
            for frame_number, image_data in self.get_frame_iterator():
                try:
                    # Convert PNG bytes to OpenCV format
                    img_buffer = BytesIO(image_data)
                    pil_image = Image.open(img_buffer)
                    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    # Initialize video writer with first frame
                    if video_writer is None:
                        height, width = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
                        logger.info(f"📹 Video writer initialized: {width}x{height} @ {fps} fps")
                    
                    # Write frame
                    video_writer.write(frame)
                    processed_count += 1
                    
                    # Progress logging
                    if processed_count % 25 == 0:
                        progress = (processed_count / total_frames) * 100
                        logger.info(f"📹 Progress: {processed_count}/{total_frames} ({progress:.1f}%)")
                    
                    # Memory cleanup every batch
                    if processed_count % self.batch_size == 0:
                        gc.collect()
                        
                except Exception as frame_error:
                    logger.warning(f"Skipped corrupted frame {frame_number}: {frame_error}")
                    continue
            
            if video_writer:
                video_writer.release()
                logger.info(f"✅ Video created successfully: {output_filename} ({processed_count} frames)")
                
                # Update metadata if available
                self.update_video_metadata(output_filename, processed_count)
                
                return output_filename
            else:
                logger.error("Failed to initialize video writer")
                return None
                
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return None
    
    def update_video_metadata(self, filename, frame_count):
        """Update video metadata in database if available"""
        if not USE_RAM_STORAGE and VideoMetadata is not None:
            try:
                video_meta = VideoMetadata.query.first()
                if not video_meta:
                    video_meta = VideoMetadata(
                        video_name=filename,
                        total_frames=frame_count,
                        fps=fps,
                        width=0,
                        height=0,
                        interval_seconds=interval,
                        file_path=filename,
                        is_complete=True
                    )
                    db.session.add(video_meta)
                else:
                    video_meta.total_frames = frame_count
                    video_meta.updated_at = datetime.utcnow()
                    video_meta.is_complete = True
                    video_meta.file_path = filename
                
                db.session.commit()
                logger.info("📝 Video metadata updated")
            except Exception as e:
                logger.warning(f"Failed to update video metadata: {e}")

# Global video creator instance
video_creator = LazyVideoCreator(batch_size=15)

def create_video_from_storage(output_filename=None):
    """Create video from storage with efficient streaming"""
    return video_creator.create_video_streaming(output_filename)

def create_video_from_db_frames(output_filename=None):
    """Backward compatibility wrapper"""
    return create_video_from_storage(output_filename)

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

            # Save frame to storage (database or RAM) with app context
            frame_count += 1
            with app.app_context():
                try:
                    saved_frame = save_frame_to_storage(frame, frame_count)
                    
                    if saved_frame:
                        storage_type = "database" if not USE_RAM_STORAGE else "RAM"
                        logger.info(f"✅ Saved frame #{frame_count} to {storage_type}")
                        print(f"✅ Added frame #{frame_count} to {storage_type}")
                        
                        # Create/update video file every N frames (less frequent for efficiency)
                        if frame_count % 20 == 0:  # Every 20 frames
                            logger.info("🔄 Updating video with new frames...")
                            video_file = create_video_from_storage(output_file)
                            if video_file:
                                logger.info(f"📹 Video updated: {video_file}")
                        
                        # Memory management for RAM storage
                        if USE_RAM_STORAGE and frame_count % 50 == 0:
                            ram_storage.clear_old_frames(keep_last_n=300)
                            gc.collect()
                            logger.info(f"🧹 Memory cleanup completed (RAM: {ram_storage.get_size_mb()} MB)")
                            
                    else:
                        logger.error(f"Failed to save frame #{frame_count}")
                        
                except Exception as storage_error:
                    logger.error(f"Storage error for frame #{frame_count}: {storage_error}")
                    logger.info(f"✅ Captured frame #{frame_count} (storage error)")
                    print(f"✅ Captured frame #{frame_count} (storage error)")

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
            .loading { 
                display: inline-block; 
                width: 20px; 
                height: 20px; 
                border: 3px solid #f3f3f3; 
                border-top: 3px solid #007acc; 
                border-radius: 50%; 
                animation: spin 1s linear infinite; 
                margin-right: 10px; 
            }
            @keyframes spin { 
                0% { transform: rotate(0deg); } 
                100% { transform: rotate(360deg); } 
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-online { background-color: #28a745; }
            .status-offline { background-color: #dc3545; }
            .status-warning { background-color: #ffc107; }
            .error-banner {
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                display: none;
            }
            .success-banner {
                background: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
                display: none;
            }
            .db-url {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                background: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                word-break: break-all;
            }
            .status-good { color: #28a745; font-weight: bold; }
            .status-bad { color: #dc3545; font-weight: bold; }
            .status-warning { color: #ffc107; font-weight: bold; }
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
                </ul>
                
                <p><strong>Database Status:</strong></p>
                <ul>
                    <li>Database Type: <span id="db-type">{{ db_info.type }}</span></li>
                    <li>Connection Status: <span id="db-connection">{{ db_info.status }}</span></li>
                    <li>Connection URL: <span id="db-url" class="db-url">{{ db_info.url_display }}</span></li>
                    <li>Last Test: <span id="db-last-test">{{ db_info.last_test }}</span></li>
                </ul>
                
                <p><strong>Storage Statistics:</strong></p>
                <ul>
                    <li>Storage mode: <span id="storage-mode">{{ storage_mode }}</span></li>
                    <li>Total frames stored: <span id="total-frames">{{ db_stats.total_frames }}</span></li>
                    <li>Latest frame: #<span id="latest-frame">{{ db_stats.latest_frame }}</span></li>
                    <li>Storage size: <span id="db-size">{{ db_stats.db_size_mb }}</span> MB</li>
                    <li>Video status: <span id="video-status">{{ "✅ Ready" if db_stats.video_exists else "⏳ Generating" }}</span></li>
                    <li>Connection status: <span id="connection-status">{{ connection_status }}</span></li>
                </ul>
            </div>
            
            <div class="controls">
                <a href="/download" class="btn" id="download-btn">📥 Download Video</a>
                <button onclick="rebuildVideo()" class="btn" id="rebuild-btn">🔨 Rebuild Video</button>
                <button onclick="refreshStats()" class="btn" id="refresh-btn">📊 Refresh Stats</button>
                <button onclick="location.reload()" class="btn">🔄 Refresh Logs</button>
                <button onclick="testConnection()" class="btn">🔗 Test Connection</button>
            </div>
            
            <div id="error-banner" class="error-banner"></div>
            <div id="success-banner" class="success-banner"></div>
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
            
            // Utility functions
            function showLoading(buttonId) {
                const btn = document.getElementById(buttonId);
                const originalText = btn.innerHTML;
                btn.innerHTML = '<span class="loading"></span>' + originalText;
                btn.disabled = true;
                return originalText;
            }
            
            function hideLoading(buttonId, originalText) {
                const btn = document.getElementById(buttonId);
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
            
            function showError(message) {
                const banner = document.getElementById('error-banner');
                banner.textContent = message;
                banner.style.display = 'block';
                setTimeout(() => banner.style.display = 'none', 8000);
            }
            
            function showSuccess(message) {
                const banner = document.getElementById('success-banner');
                banner.textContent = message;
                banner.style.display = 'block';
                setTimeout(() => banner.style.display = 'none', 5000);
            }
            
            // Refresh database statistics with loading
            function refreshStats() {
                const originalText = showLoading('refresh-btn');
                
                fetch('/api/stats')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        document.getElementById('total-frames').textContent = data.total_frames;
                        document.getElementById('latest-frame').textContent = data.latest_frame_number;
                        document.getElementById('db-size').textContent = data.database_size_mb;
                        document.getElementById('video-status').textContent = data.video_exists ? '✅ Ready' : '⏳ Generating';
                        document.getElementById('storage-mode').textContent = data.storage_mode || 'Unknown';
                        document.getElementById('connection-status').textContent = data.connection_status || 'Unknown';
                        
                        // Update database info if available
                        if (data.database_info) {
                            document.getElementById('db-type').textContent = data.database_info.type;
                            document.getElementById('db-connection').textContent = data.database_info.status;
                            document.getElementById('db-url').textContent = data.database_info.url_display;
                            document.getElementById('db-last-test').textContent = data.database_info.last_test;
                        }
                        
                        showSuccess('Statistics refreshed successfully');
                    })
                    .catch(error => {
                        console.error('Error refreshing stats:', error);
                        showError(`Failed to refresh statistics: ${error.message}`);
                    })
                    .finally(() => {
                        hideLoading('refresh-btn', originalText);
                    });
            }
            
            // Rebuild video with enhanced loading and error handling
            function rebuildVideo() {
                const originalText = showLoading('rebuild-btn');
                const statusDiv = document.getElementById('rebuild-status');
                
                statusDiv.style.display = 'block';
                statusDiv.style.background = '#fff3cd';
                statusDiv.style.border = '1px solid #ffeaa7';
                statusDiv.style.color = '#856404';
                statusDiv.innerHTML = '<span class="loading"></span>🔨 Rebuilding video from frames... This may take a few minutes.';
                
                fetch('/api/rebuild_video')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        if (data.success) {
                            statusDiv.style.background = '#d4edda';
                            statusDiv.style.border = '1px solid #c3e6cb';
                            statusDiv.style.color = '#155724';
                            statusDiv.innerHTML = `✅ ${data.message} (${data.total_frames} frames)`;
                            refreshStats();
                            showSuccess('Video rebuilt successfully!');
                        } else {
                            statusDiv.style.background = '#f8d7da';
                            statusDiv.style.border = '1px solid #f5c6cb';
                            statusDiv.style.color = '#721c24';
                            statusDiv.innerHTML = `❌ Error: ${data.message}`;
                            showError(`Video rebuild failed: ${data.message}`);
                        }
                        
                        setTimeout(() => statusDiv.style.display = 'none', 8000);
                    })
                    .catch(error => {
                        console.error('Rebuild error:', error);
                        statusDiv.style.background = '#f8d7da';
                        statusDiv.style.border = '1px solid #f5c6cb';
                        statusDiv.style.color = '#721c24';
                        statusDiv.innerHTML = `❌ Network error: ${error.message}`;
                        showError(`Network error during rebuild: ${error.message}`);
                        
                        setTimeout(() => statusDiv.style.display = 'none', 8000);
                    })
                    .finally(() => {
                        hideLoading('rebuild-btn', originalText);
                    });
            }
            
            // Test database/storage connection
            function testConnection() {
                showSuccess('Testing connection...');
                
                // Update last test time immediately
                document.getElementById('db-last-test').textContent = 'Testing...';
                
                fetch('/api/test_connection')
                    .then(response => {
                        if (!response.ok) throw new Error(`HTTP ${response.status}`);
                        return response.json();
                    })
                    .then(data => {
                        if (data.success) {
                            showSuccess(`✅ ${data.message}`);
                            // Update connection status immediately
                            document.getElementById('db-connection').textContent = '🟢 Connected';
                        } else {
                            showError(`❌ ${data.message}`);
                            document.getElementById('db-connection').textContent = '❌ Failed';
                        }
                        
                        // Update last test time
                        const now = new Date();
                        document.getElementById('db-last-test').textContent = now.toLocaleTimeString();
                        
                        // Refresh all stats to get updated info
                        setTimeout(refreshStats, 1000);
                    })
                    .catch(error => {
                        console.error('Connection test error:', error);
                        showError(`Connection test failed: ${error.message}`);
                        document.getElementById('db-connection').textContent = '❌ Network Error';
                        document.getElementById('db-last-test').textContent = 'Network Error';
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
    
    # Get storage statistics
    try:
        total_frames = get_all_frames_count()
        
        db_stats = {
            'total_frames': total_frames,
            'latest_frame': 0,
            'db_size_mb': 0,
            'video_exists': os.path.exists(output_file)
        }
        
        # Get database information
        db_info = get_database_info()
        
        # Get latest frame and storage size based on storage type
        if USE_RAM_STORAGE:
            frames = ram_storage.get_all_frames()
            if frames:
                db_stats['latest_frame'] = max([f[0] for f in frames])
            db_stats['db_size_mb'] = ram_storage.get_size_mb()
            storage_mode = "🟡 RAM Storage (Database Offline)"
            connection_status = "🔴 Database Offline - Using RAM Fallback"
        else:
            if FrameData is not None:
                try:
                    latest_frame = FrameData.query.order_by(FrameData.frame_number.desc()).first()
                    if latest_frame:
                        db_stats['latest_frame'] = latest_frame.frame_number
                except:
                    pass
            db_stats['db_size_mb'] = get_database_size_mb()
            db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
            storage_mode = f"🟢 {db_type} Database"
            connection_status = "🟢 Database Online"
            
    except Exception as e:
        logger.error(f"Error getting storage stats: {e}")
        db_stats = {
            'total_frames': 0,
            'latest_frame': 0,
            'db_size_mb': 0,
            'video_exists': os.path.exists(output_file)
        }
        storage_mode = "❌ Unknown"
        connection_status = "❌ Error"
        db_info = {
            'type': '❌ Unknown',
            'status': '❌ Error',
            'url_display': 'Error retrieving info',
            'last_test': 'Error'
        }
    
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
                                storage_mode=storage_mode,
                                connection_status=connection_status,
                                db_info=db_info)

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
    """API endpoint for storage statistics with enhanced information"""
    try:
        total_frames = get_all_frames_count()
        storage_mode = "RAM Storage" if USE_RAM_STORAGE else "Database"
        connection_status = "🔴 Offline" if USE_RAM_STORAGE else "🟢 Online"
        
        # Get latest frame info
        latest_frame_number = 0
        latest_frame_time = None
        
        if not USE_RAM_STORAGE and FrameData is not None:
            try:
                latest_frame = FrameData.query.order_by(FrameData.frame_number.desc()).first()
                if latest_frame:
                    latest_frame_number = latest_frame.frame_number
                    latest_frame_time = latest_frame.timestamp.isoformat()
            except Exception:
                pass
        elif USE_RAM_STORAGE:
            frames = ram_storage.get_all_frames()
            if frames:
                latest_frame_number = max([f[0] for f in frames])
                latest_frame_time = datetime.utcnow().isoformat()
        
        # Get video metadata
        video_meta = None
        if not USE_RAM_STORAGE and VideoMetadata is not None:
            try:
                video_meta = VideoMetadata.query.first()
            except Exception:
                pass
        
        # Calculate storage size
        if USE_RAM_STORAGE:
            storage_size = ram_storage.get_size_mb()
        else:
            storage_size = get_database_size_mb()
        
        # Get database info
        db_info = get_database_info()
        
        stats = {
            'total_frames': total_frames,
            'latest_frame_number': latest_frame_number,
            'latest_frame_time': latest_frame_time,
            'database_size_mb': storage_size,
            'storage_mode': storage_mode,
            'connection_status': connection_status,
            'video_exists': os.path.exists(output_file),
            'using_ram_storage': USE_RAM_STORAGE,
            'database_info': db_info,
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
    """Force rebuild video from frames with enhanced error handling"""
    logger.info("User requested video rebuild from storage")
    try:
        frame_count = get_all_frames_count()
        if frame_count == 0:
            return jsonify({
                'success': False, 
                'message': 'No frames available for video creation'
            }), 400
        
        logger.info(f"Starting video rebuild with {frame_count} frames")
        video_file = create_video_from_storage(output_file)
        
        if video_file:
            storage_type = "RAM storage" if USE_RAM_STORAGE else "database"
            return jsonify({
                'success': True, 
                'message': f'Video rebuilt from {storage_type}: {video_file}',
                'total_frames': frame_count,
                'storage_mode': storage_type
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Video creation failed - check logs for details'
            }), 500
            
    except Exception as e:
        logger.error(f"Error rebuilding video: {e}")
        return jsonify({
            'success': False, 
            'message': f'Video rebuild error: {str(e)}'
        }), 500

@app.route("/api/test_connection")
def test_connection():
    """Test database/storage connection"""
    try:
        if USE_RAM_STORAGE:
            # Test RAM storage
            frame_count = ram_storage.count_frames()
            size_mb = ram_storage.get_size_mb()
            return jsonify({
                'success': True,
                'message': f'RAM storage OK - {frame_count} frames ({size_mb} MB)',
                'storage_type': 'RAM',
                'frames': frame_count,
                'size_mb': size_mb
            })
        else:
            # Test database connection
            if db is None:
                return jsonify({
                    'success': False,
                    'message': 'Database not initialized'
                }), 500
            
            # Try a simple query
            frame_count = get_all_frames_count()
            db_size = get_database_size_mb()
            
            # Test database write capability
            with app.app_context():
                db.session.execute('SELECT 1')
                
            db_type = "PostgreSQL" if "postgresql" in DATABASE_URL else "SQLite"
            return jsonify({
                'success': True,
                'message': f'{db_type} connection OK - {frame_count} frames ({db_size} MB)',
                'storage_type': db_type,
                'frames': frame_count,
                'size_mb': db_size
            })
            
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return jsonify({
            'success': False,
            'message': f'Connection test failed: {str(e)}'
        }), 500

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

def get_database_info():
    """Get detailed database information for status display"""
    try:
        if USE_RAM_STORAGE:
            return {
                'type': '🟡 RAM Storage (Fallback)',
                'status': '🔴 Database Connection Failed',
                'url_display': 'Using in-memory storage',
                'last_test': 'Connection failed at startup'
            }
        
        # Determine database type
        if 'postgresql' in DATABASE_URL.lower():
            db_type = '🐘 PostgreSQL'
        elif 'sqlite' in DATABASE_URL.lower():
            db_type = '📁 SQLite'
        else:
            db_type = '❓ Unknown'
        
        # Create display URL (hide sensitive info)
        url_display = DATABASE_URL
        if '@' in DATABASE_URL:
            # Hide password in URL
            parts = DATABASE_URL.split('@')
            if '://' in parts[0]:
                protocol_user = parts[0].split('://')
                if ':' in protocol_user[1]:
                    user_pass = protocol_user[1].split(':')
                    url_display = f"{protocol_user[0]}://{user_pass[0]}:***@{parts[1]}"
        
        # Test current connection status
        try:
            if db is not None:
                with app.app_context():
                    db.session.execute('SELECT 1')
                status = '🟢 Connected'
                last_test = datetime.now().strftime('%H:%M:%S')
            else:
                status = '🔴 Not Initialized'
                last_test = 'Never'
        except Exception as test_error:
            status = f'❌ Connection Error'
            last_test = f'Failed: {str(test_error)[:50]}...'
        
        return {
            'type': db_type,
            'status': status,
            'url_display': url_display,
            'last_test': last_test
        }
        
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {
            'type': '❌ Error',
            'status': '❌ Info retrieval failed',
            'url_display': str(e)[:100],
            'last_test': 'Error'
        }

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
    
    # Initialize storage system
    try:
        if not USE_RAM_STORAGE and db is not None:
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
        else:
            logger.info(f"📊 RAM storage initialized - capacity: {ram_storage.max_frames} frames")
            logger.info("⚠️ Note: RAM storage will lose data on restart - frames are temporary")
            
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        logger.warning("🔄 Switching to RAM storage due to initialization failure")
        USE_RAM_STORAGE = True
    
    # Start video loop in background
    logger.info("Starting background timelapse video generation thread")
    threading.Thread(target=run_video_loop, daemon=True).start()
    
    # Start Flask server
    logger.info("Starting Flask web server on http://0.0.0.0:10000")
    logger.info("🌐 Visit the website to view real-time logs, database stats, and download videos")
    logger.info("🔄 Application will survive Render hosting restarts with database persistence")
    
    port = int(os.getenv('PORT', 10000))
    app.run(host="0.0.0.0", port=port)
