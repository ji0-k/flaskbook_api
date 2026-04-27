import cv2
import base64
import os
import time
import threading
from pathlib import Path
from ultralytics import YOLO
from api.extensions import db
from api.models.detection import DetectionSession, DetectionLog

_detection_target = None
_detection_active = False
basedir = Path(__file__).parent.parent.parent

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (255, 128, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128),
]

def set_target(target: str):
    global _detection_target
    _detection_target = target

def set_detection_active(active: bool):
    global _detection_active
    _detection_active = active

def load_model():
    model = YOLO('yolov8n.pt')
    return model

def detect_objects(model, frame):
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        score = float(box.conf)
        if score < 0.8:
            continue
        label_id = int(box.cls)
        label_name = model.names[label_id]
        if _detection_target and label_name != _detection_target:
            continue
        color = COLORS[label_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_name} {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detections.append({
            "class_name": label_name,
            "confidence": score,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    return frame, detections

def open_capture(rtsp_url: str):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp"
        "|fflags;+discardcorrupt"
        "|analyzeduration;2000000"
        "|probesize;1000000"
    )
    return cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)


class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def write(self, frame):
        with self.lock:
            self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


def capture_thread(rtsp_url: str, buffer: FrameBuffer, stop_event: threading.Event):
    """RTSP 캡처 전용 스레드"""
    cap = open_capture(rtsp_url)
    fail_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            cap.release()
            time.sleep(min(fail_count * 2, 10))
            cap = open_capture(rtsp_url)
            if cap.isOpened():
                fail_count = 0
            continue

        fail_count = 0
        buffer.write(frame)

    cap.release()


def run_rtsp_stream(socketio, rtsp_url: str, app):
    print("[INFO] 모델 로딩 중...")
    model = load_model()
    print("[INFO] 모델 로딩 완료!")

    buffer = FrameBuffer()
    stop_event = threading.Event()

    t = threading.Thread(target=capture_thread, args=(rtsp_url, buffer, stop_event), daemon=True)
    t.start()

    DETECT_EVERY = 3
    SEND_WIDTH = 640
    frame_count = 0
    last_detected_frame = None
    last_detections = []
    session_id = None

    try:
        while True:
            frame = buffer.read()
            if frame is None:
                socketio.sleep(0.05)
                continue

            frame_count += 1

            h, w = frame.shape[:2]
            if w > SEND_WIDTH:
                scale = SEND_WIDTH / w
                frame = cv2.resize(frame, (SEND_WIDTH, int(h * scale)))

            if _detection_active:
                if session_id is None:
                    with app.app_context():
                        session_obj = DetectionSession(camera_url=rtsp_url)
                        db.session.add(session_obj)
                        db.session.commit()
                        session_id = session_obj.id
                        print(f"[DB] 세션 시작: session_id={session_id}")

                if frame_count % DETECT_EVERY == 0:
                    last_detected_frame, last_detections = detect_objects(model, frame)

                    if last_detections:
                        socketio.emit('detection_result', {'detections': last_detections})
                        with app.app_context():
                            logs = [
                                DetectionLog(session_id=session_id, **d)
                                for d in last_detections
                            ]
                            db.session.bulk_save_objects(logs)
                            db.session.commit()

                send_frame = last_detected_frame if last_detected_frame is not None else frame

            else:
                if session_id is not None:
                    with app.app_context():
                        session_obj = DetectionSession.query.get(session_id)
                        if session_obj:
                            session_obj.end()
                            db.session.commit()
                            print(f"[DB] 세션 종료: session_id={session_id}")
                    session_id = None
                    last_detected_frame = None
                    last_detections = []

                send_frame = frame

            _, buffer_jpg = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer_jpg).decode('utf-8')

            socketio.emit('video_frame', {'frame': frame_b64})
            socketio.sleep(0.05)

    finally:
        stop_event.set()
        if session_id is not None:
            with app.app_context():
                session_obj = DetectionSession.query.get(session_id)
                if session_obj:
                    session_obj.end()
                    db.session.commit()
                    print(f"[DB] 세션 종료: session_id={session_id}")