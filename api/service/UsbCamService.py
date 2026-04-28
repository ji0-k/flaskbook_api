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

        cv2.putText(
            frame,
            f"{label_name} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

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
            if self.frame is None:
                return None

            return self.frame.copy()


def open_camera():
    """
    Windows 노트북 내장 웹캠 연결용.
    보통 내장 웹캠은 index=0 이지만,
    환경에 따라 1 또는 2일 수도 있어서 순서대로 탐색한다.
    """

    for index in [0, 1, 2]:
        print(f"[USB] 카메라 index={index} 연결 시도")

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if cap.isOpened():
            print(f"[USB] 카메라 연결 성공: index={index}")

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            return cap

        cap.release()

    print("[USB ERROR] 사용 가능한 웹캠을 찾지 못했습니다.")
    return None


def capture_thread(buffer: FrameBuffer, stop_event: threading.Event, socketio):
    cap = open_camera()

    if cap is None:
        socketio.emit("usb_error", {
            "message": "웹캠을 열 수 없습니다. 카메라 권한 또는 장치 사용 여부를 확인하세요."
        })
        return

    frame_count = 0
    fail_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            fail_count += 1

            if fail_count % 30 == 0:
                print(f"[USB WARNING] 프레임 읽기 실패 누적: {fail_count}")

            continue

        fail_count = 0
        frame_count += 1

        # 콘솔 지옥 방지: 100프레임마다 한 번만 출력
        if frame_count % 100 == 0:
            print(f"[USB] 프레임 수신 중... frame_count={frame_count}")

        buffer.write(frame)

    cap.release()
    print("[USB] 카메라 연결 종료")


def run_usb_stream(socketio):
    print("[USB] 모델 로딩 중...")
    model = YOLO("yolov8n.pt")
    print("[USB] 모델 로딩 완료!")

    buffer = FrameBuffer()
    stop_event = threading.Event()

    capture = threading.Thread(
        target=capture_thread,
        args=(buffer, stop_event, socketio),
        daemon=True
    )
    capture.start()

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
                    detected_frame, detections = detect_objects(
                        model,
                        frame.copy()
                    )

                    last_detected_frame = detected_frame

                    if detections:
                        socketio.emit("usb_detection_result", {
                            "detections": detections
                        })

                send_frame = (
                    last_detected_frame
                    if last_detected_frame is not None
                    else frame
                )

            else:
                last_detected_frame = None
                send_frame = frame

            success, buffer_jpg = cv2.imencode(
                ".jpg",
                send_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            if not success:
                continue

            frame_b64 = base64.b64encode(buffer_jpg).decode("utf-8")

            socketio.emit("usb_frame", {
                "frame": frame_b64
            })

            socketio.sleep(0.05)

    finally:
        stop_event.set()
        print("[USB] 스트림 종료")