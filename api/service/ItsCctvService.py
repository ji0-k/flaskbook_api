import cv2
import base64
import requests
import threading
import time
import os
from ultralytics import YOLO

_detection_active = False
_current_stop_event = None
_stream_lock = threading.Lock()

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128),
    (255, 128, 0), (0, 255, 128), (128, 0, 255), (255, 0, 128),
]

_model = None


def get_model():
    global _model

    if _model is None:
        print("[ITS] 모델 로딩 중...")
        _model = YOLO("yolov8n.pt")
        print("[ITS] 모델 로딩 완료!")

    return _model


def set_its_detection_active(active: bool):
    global _detection_active
    _detection_active = active
    print(f"[ITS] 탐지 활성화 상태 변경: {_detection_active}")


def fetch_cctv_list(
    api_key: str,
    api_url: str,
    cctv_type: int = 1,
    min_x: float = 126.0,
    max_x: float = 129.0,
    min_y: float = 34.0,
    max_y: float = 38.5
) -> list:
    """
    ITS API에서 CCTV 목록 가져오기
    """

    try:
        params = {
            "apiKey": api_key,
            "type": "its",
            "cctvType": cctv_type,
            "minX": min_x,
            "maxX": max_x,
            "minY": min_y,
            "maxY": max_y,
            "getType": "json",
        }

        print("[ITS DEBUG] CCTV 목록 요청 시작")
        print("[ITS DEBUG] api_url =", api_url)
        print("[ITS DEBUG] api_key =", api_key[:5] + "..." if api_key else None)
        print("[ITS DEBUG] params =", {k: v for k, v in params.items() if k != "apiKey"})

        resp = requests.get(
            api_url,
            params=params,
            timeout=10,
            verify=False
        )

        print("[ITS DEBUG] status_code =", resp.status_code)
        print("[ITS DEBUG] request_url =", resp.url)
        print("[ITS DEBUG] response preview =", resp.text[:500])

        resp.raise_for_status()
        data = resp.json()

        items = data.get("response", {}).get("data", [])

        print(f"[ITS DEBUG] CCTV 목록 개수 = {len(items)}")

        result = []

        for item in items:
            url = item.get("cctvurl", "")

            if not url:
                continue

            name = item.get("cctvname", "이름없음")

            print("[ITS DEBUG] CCTV item =", name, "/", url)

            result.append({
                "name": name,
                "url": url,
                "x": item.get("coordx", ""),
                "y": item.get("coordy", ""),
            })

        return result

    except Exception as e:
        print(f"[ITS ERROR] CCTV 목록 조회 실패: {e}")
        return []


class FrameBuffer:
    """
    캡처 스레드가 읽은 최신 프레임을 저장하는 버퍼
    """

    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def write(self, frame):
        with self.lock:
            self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


def _open_capture(url: str):
    """
    CCTV URL을 OpenCV VideoCapture로 여는 함수

    - RTSP URL이면 RTSP 전용 옵션 적용
    - HTTP / HTTPS / m3u8 URL이면 RTSP 옵션을 적용하지 않음
    """

    print("[ITS DEBUG] OpenCV open url =", url)

    if url.startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp"
            "|fflags;+discardcorrupt"
            "|analyzeduration;2000000"
            "|probesize;1000000"
        )
        print("[ITS DEBUG] RTSP 옵션 적용")
    else:
        # 이전에 RTSP 옵션이 남아 있으면 HTTP/HLS 재생에 방해될 수 있어서 제거
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        print("[ITS DEBUG] RTSP 옵션 미적용")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    print("[ITS DEBUG] cap opened =", cap.isOpened())

    return cap


def _capture_thread(url: str, buffer: FrameBuffer, stop_event: threading.Event):
    """
    CCTV 프레임을 계속 읽어서 FrameBuffer에 저장하는 백그라운드 스레드
    """

    cap = _open_capture(url)
    fail_count = 0

    if not cap.isOpened():
        print("[ITS ERROR] CCTV 캡처 열기 실패:", url)

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            fail_count += 1

            print(
                f"[ITS WARN] 프레임 읽기 실패 ({fail_count}회) "
                f"/ opened={cap.isOpened()}"
            )

            cap.release()

            # 실패할수록 재시도 간격 증가, 최대 10초
            time.sleep(min(fail_count * 2, 10))

            cap = _open_capture(url)

            if cap.isOpened():
                print("[ITS INFO] CCTV 재연결 성공")
                fail_count = 0

            continue

        fail_count = 0
        buffer.write(frame)

    cap.release()
    print("[ITS] 캡처 스레드 종료")


def detect_objects(model, frame):
    """
    YOLO 모델로 객체 탐지 후 박스 그리기
    """

    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        score = float(box.conf)

        if score < 0.5:
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
            "confidence": score
        })

    return frame, detections


def run_its_stream(socketio, cctv_url: str):
    """
    선택된 ITS CCTV URL로 스트리밍 + YOLO 탐지
    """

    global _current_stop_event

    with _stream_lock:
        if _current_stop_event is not None:
            print("[ITS] 기존 스트림 중단 요청")
            _current_stop_event.set()
            time.sleep(0.5)

        stop_event = threading.Event()
        _current_stop_event = stop_event

    model = get_model()
    buffer = FrameBuffer()

    t = threading.Thread(
        target=_capture_thread,
        args=(cctv_url, buffer, stop_event),
        daemon=True
    )
    t.start()

    DETECT_EVERY = 3
    SEND_WIDTH = 640

    frame_count = 0
    last_detected_frame = None

    print(f"[ITS] 스트리밍 시작: {cctv_url}")

    try:
        while not stop_event.is_set():
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
                if frame_count % DETECT_EVERY == 0:
                    last_detected_frame, detections = detect_objects(model, frame)

                    if detections:
                        socketio.emit(
                            "its_detection_result",
                            {"detections": detections}
                        )

                send_frame = (
                    last_detected_frame
                    if last_detected_frame is not None
                    else frame
                )

            else:
                last_detected_frame = None
                send_frame = frame

            ok, encoded = cv2.imencode(
                ".jpg",
                send_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            if not ok:
                print("[ITS WARN] JPG 인코딩 실패")
                socketio.sleep(0.05)
                continue

            frame_b64 = base64.b64encode(encoded).decode("utf-8")

            socketio.emit("its_frame", {"frame": frame_b64})

            socketio.sleep(0.05)

    finally:
        stop_event.set()
        print("[ITS] 스트리밍 종료")


def stop_its_stream():
    """
    현재 실행 중인 ITS 스트림 중단
    """

    global _current_stop_event

    with _stream_lock:
        if _current_stop_event is not None:
            print("[ITS] 스트림 중단")
            _current_stop_event.set()
            _current_stop_event = None