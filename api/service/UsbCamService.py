import cv2
import base64
import threading
from ultralytics import YOLO

_detection_active = False

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (255, 128, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128),
]

def set_usb_detection_active(active: bool):
    global _detection_active
    _detection_active = active

def detect_objects(model, frame):
    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        score = float(box.conf)
        if score < 0.8:
            continue
        label_id = int(box.cls)
        label_name = model.names[label_id]
        color = COLORS[label_id % len(COLORS)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_name} {score:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        detections.append({
            "class_name": label_name,
            "confidence": score,
        })

    return frame, detections


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


def capture_thread(buffer: FrameBuffer, stop_event: threading.Event):
    cap = cv2.VideoCapture(0)
    print(f"[DEBUG] 카메라 오픈 상태: {cap.isOpened()}")
    if not cap.isOpened():
        print("[ERROR] USB 카메라를 열 수 없습니다!") # 로그 추가
        return
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    while not stop_event.is_set():
        print("[THREAD] cap.read() 시도 중...") # ⬅️ 여기 진입 확인
        ret, frame = cap.read()
        print(f"[THREAD] cap.read() 완료, 결과: {ret}") # ⬅️ 여기가 출력 안 된다면 프리징
        if ret:
            buffer.write(frame)

    cap.release()


def run_usb_stream(socketio):
    print("[USB] 모델 로딩 중...")
    model = YOLO('yolov8n.pt')
    print("[USB] 모델 로딩 완료!")

    buffer = FrameBuffer()
    stop_event = threading.Event()

    t = threading.Thread(target=capture_thread, args=(buffer, stop_event), daemon=True)
    t.start()

    DETECT_EVERY = 3
    frame_count = 0
    last_detected_frame = None

    try:
        while True:
            frame = buffer.read()
            if frame is None:
                socketio.sleep(0.05)
                continue
            
            frame_count += 1

            if _detection_active:
                if frame_count % DETECT_EVERY == 0:
                    last_detected_frame, detections = detect_objects(model, frame)
                    if detections:
                        socketio.emit('usb_detection_result', {'detections': detections})
                send_frame = last_detected_frame if last_detected_frame is not None else frame
            else:
                last_detected_frame = None
                send_frame = frame

            _, buffer_jpg = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer_jpg).decode('utf-8')
            socketio.emit('usb_frame', {'frame': frame_b64})
            socketio.sleep(0.05)

    finally:
        stop_event.set()