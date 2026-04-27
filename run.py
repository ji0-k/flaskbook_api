import os
from dotenv import load_dotenv
from flask import Flask, render_template
from flask_socketio import SocketIO
from api import api
from api.config import config
from api.service.AiStreamService import run_rtsp_stream, set_target
from api.service.UsbCamService import run_usb_stream, set_usb_detection_active
from api.extensions import db, migrate

# .env 파일 로드
load_dotenv()

config_name = os.environ.get("CONFIG", "local")

app = Flask(__name__)
app.config.from_object(config[config_name])

# DB 설정
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    f"?charset=utf8mb4"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
migrate.init_app(app, db)

from api.models import detection  # noqa

app.register_blueprint(api)

# SocketIO 설정
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# RTSP URL (.env에서 로드)
RTSP_URL = os.getenv("RTSP_URL")
print(f"[DEBUG] RTSP_URL = {RTSP_URL}")

_ai_task_started = False
_usb_task_started = False

# --- 라우트 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ai-detect/aistream')
def ai_stream():
    return render_template('ai_detect/ai_stream.html')

@app.route('/ai-detect/usbstream')
def usb_stream():
    return render_template('ai_detect/usb_stream.html')

# --- AI (IP CAM) ---
def run_ai_logic():
    print(f"[SYSTEM] AI Background Task Start")
    run_rtsp_stream(socketio, RTSP_URL, app)

@socketio.on('connect')
def handle_connect():
    global _ai_task_started
    if not _ai_task_started:
        _ai_task_started = True
        socketio.start_background_task(run_ai_logic)

@socketio.on('set_detection_target')
def handle_target(data):
    target = data.get('target', '')
    set_target(target)

@socketio.on('start_detection')
def handle_start_detection():
    from api.service.AiStreamService import set_detection_active
    set_detection_active(True)

@socketio.on('stop_detection')
def handle_stop_detection():
    from api.service.AiStreamService import set_detection_active
    set_detection_active(False)

# --- USB CAM ---
@socketio.on('usb_connect')
def handle_usb_connect():
    global _usb_task_started
    if not _usb_task_started:
        _usb_task_started = True
        socketio.start_background_task(run_usb_stream, socketio)

@socketio.on('usb_start_detection')
def handle_usb_start():
    set_usb_detection_active(True)

@socketio.on('usb_stop_detection')
def handle_usb_stop():
    set_usb_detection_active(False)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)