import os
import re
import time
import base64
import requests
import cv2

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO

from api import api
from api.config import config
from api.extensions import db, migrate

# 기존 IP CAM RTSP 스트리밍 서비스
from api.service.AiStreamService import run_rtsp_stream, set_target

# USB 카메라 스트리밍 서비스
from api.service.UsbCamService import run_usb_stream, set_usb_detection_active


# --------------------------------------------------
# 1. .env 환경 변수 로드
# --------------------------------------------------
# .env 안에 있는 DB 정보, RTSP_URL, ITS_API_KEY 등을 불러온다.
load_dotenv()

# CONFIG 값이 없으면 local 설정을 기본으로 사용한다.
config_name = os.environ.get("CONFIG", "local")


# --------------------------------------------------
# 2. Flask 앱 생성 및 config 적용
# --------------------------------------------------
app = Flask(__name__)
app.config.from_object(config[config_name])


# --------------------------------------------------
# 3. DB 설정
# --------------------------------------------------
# .env에 있는 DB 접속 정보를 이용해서 SQLAlchemy DB URI 생성
app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{os.environ['DB_USERNAME']}:{os.environ['DB_PASSWORD']}"
    f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    f"?charset=utf8mb4"
)

# SQLAlchemy 변경 추적 기능 비활성화
# 불필요한 메모리 사용을 줄이기 위함
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# DB와 Flask-Migrate 초기화
db.init_app(app)
migrate.init_app(app, db)

# Flask-Migrate가 detection 모델을 인식하도록 import
from api.models import detection  # noqa


# --------------------------------------------------
# 4. Blueprint 등록
# --------------------------------------------------
# api 패키지에 등록된 라우트들을 Flask 앱에 연결한다.
app.register_blueprint(api)


# --------------------------------------------------
# 5. SocketIO 설정
# --------------------------------------------------
# eventlet은 deprecated 경고가 발생하므로 threading 사용
# 현재 프로젝트 구조에서는 threading이 디버깅과 실행 안정성 면에서 더 단순함
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)


# --------------------------------------------------
# 6. 환경 변수 / API 주소 설정
# --------------------------------------------------
# 기존 IP 카메라 RTSP 주소
RTSP_URL = os.getenv("RTSP_URL")

# 국가교통정보센터 API 키
ITS_API_KEY = os.getenv("ITS_API_KEY")

# 국가교통정보센터 CCTV 목록 조회 API
# 주의:
# 이 주소는 영상 주소가 아니라 "CCTV 목록을 조회하는 API 주소"이다.
# 실제 영상 주소는 API 응답 데이터 안의 cctvurl 필드에 들어있다.
ITS_CCTV_API_URL = "https://openapi.its.go.kr:9443/cctvInfo"

print(f"[DEBUG] RTSP_URL = {RTSP_URL}")

# API 키는 보안상 실제 값을 출력하지 않고 로드 여부만 출력한다.
print(f"[DEBUG] ITS_API_KEY loaded = {bool(ITS_API_KEY)}")


# --------------------------------------------------
# 7. API 키 마스킹 함수
# --------------------------------------------------
def mask_api_key_in_url(url):
    """
    콘솔 로그에 API 키가 그대로 노출되지 않도록 마스킹하는 함수.

    예:
        https://openapi.its.go.kr:9443/cctvInfo?apiKey=abcdef&type=its

    출력:
        https://openapi.its.go.kr:9443/cctvInfo?apiKey=****MASKED****&type=its

    필요한 이유:
    - 발표 화면 캡처 시 API 키 노출 방지
    - 팀원에게 콘솔 로그 공유 시 API 키 유출 방지
    - GitHub, Notion 등에 로그를 붙여넣을 때 보안 사고 방지
    """
    if not url:
        return url

    return re.sub(
        r"(apiKey=)[^&]+",
        r"\1****MASKED****",
        url
    )


# --------------------------------------------------
# 8. 백그라운드 작업 상태값
# --------------------------------------------------
# Socket.IO는 페이지 접속, 새로고침, 재연결 때 connect 이벤트가 여러 번 발생할 수 있다.
# 그래서 같은 스트리밍 작업이 중복 실행되지 않도록 플래그를 둔다.
_ai_task_started = False
_usb_task_started = False

# ITS CCTV 스트리밍 상태값
_its_selected_url = None        # 사용자가 선택한 ITS CCTV 영상 URL
_its_stream_active = False      # ITS 스트리밍 실행 여부
_its_task_started = False       # ITS 백그라운드 작업 최초 실행 여부


# --------------------------------------------------
# 9. 화면 라우트
# --------------------------------------------------
@app.route("/")
def index():
    """
    메인 페이지
    """
    return render_template("index.html")


@app.route("/ai-detect/aistream")
def ai_stream():
    """
    기존 IP CAM RTSP 기반 AI 스트리밍 페이지
    """
    return render_template("ai_detect/ai_stream.html")


@app.route("/ai-detect/usbstream")
def usb_stream():
    """
    USB 카메라 스트리밍 페이지
    """
    return render_template("ai_detect/usb_stream.html")


@app.route("/ai-detect/itsstream")
def its_stream():
    """
    ITS 국가교통 CCTV 선택형 스트리밍 페이지
    """
    return render_template("ai_detect/its_stream.html")


# --------------------------------------------------
# 10. 기존 IP CAM RTSP 스트리밍 처리
# --------------------------------------------------
def run_ai_logic():
    """
    기존 IP CAM RTSP 스트림을 읽고,
    AI 탐지를 수행한 뒤 SocketIO로 프레임을 전송하는 백그라운드 작업.

    중요한 점:
    이 함수는 모든 Socket 연결에서 자동 실행하면 안 된다.

    이유:
    - ai_stream.html도 Socket.IO를 사용한다.
    - usb_stream.html도 Socket.IO를 사용한다.
    - its_stream.html도 Socket.IO를 사용한다.

    만약 connect 이벤트에서 바로 run_ai_logic()을 실행하면,
    ITS 페이지에 들어갔을 때도 기존 RTSP 카메라를 열려고 하면서
    [WARN] 재연결 실패 로그가 계속 발생할 수 있다.
    """
    print("[SYSTEM] AI Background Task Start")
    run_rtsp_stream(socketio, RTSP_URL, app)


@socketio.on("connect")
def handle_connect():
    """
    모든 Socket.IO 연결에서 실행되는 공통 이벤트.

    여기서는 절대 특정 스트리밍 작업을 자동 실행하지 않는다.

    각 페이지별 스트리밍 시작은 전용 이벤트에서 처리한다.
    - AI RTSP: ai_stream_connect
    - USB CAM: usb_connect
    - ITS CCTV: its_select_cctv
    """
    print("[SOCKET] client connected")


@socketio.on("ai_stream_connect")
def handle_ai_stream_connect():
    """
    ai_stream.html 페이지에서만 호출하는 이벤트.

    프론트에서:
        socket.emit("ai_stream_connect");

    를 보내면 이 함수가 실행되고,
    기존 IP CAM RTSP 스트리밍 작업이 시작된다.
    """
    global _ai_task_started

    print("[SOCKET] AI stream page connected")

    if not _ai_task_started:
        print("[SYSTEM] AI Task 최초 실행")
        _ai_task_started = True
        socketio.start_background_task(run_ai_logic)


@socketio.on("set_detection_target")
def handle_target(data):
    """
    프론트에서 탐지 대상 객체를 지정할 때 호출된다.

    예:
        person
        car
        bus
    """
    target = data.get("target", "")
    print(f"[SOCKET] target 변경: {target}")
    set_target(target)


@socketio.on("start_detection")
def handle_start_detection():
    """
    기존 IP CAM AI 탐지 시작 이벤트.

    스트리밍 자체를 새로 여는 것이 아니라,
    AiStreamService 내부의 탐지 활성화 플래그를 True로 바꾸는 역할이다.
    """
    from api.service.AiStreamService import set_detection_active

    print("[SOCKET] AI 탐지 시작")
    set_detection_active(True)


@socketio.on("stop_detection")
def handle_stop_detection():
    """
    기존 IP CAM AI 탐지 중지 이벤트.

    스트림을 완전히 끄는 것이 아니라,
    탐지 활성화 플래그만 False로 바꾼다.
    """
    from api.service.AiStreamService import set_detection_active

    print("[SOCKET] AI 탐지 중지")
    set_detection_active(False)


# --------------------------------------------------
# 11. USB 카메라 처리
# --------------------------------------------------
@socketio.on("usb_connect")
def handle_usb_connect():
    """
    usb_stream.html 페이지에서만 호출하는 이벤트.

    USB 카메라 스트리밍 백그라운드 작업을 최초 1회만 실행한다.
    """
    global _usb_task_started

    print("[SOCKET] USB client connected")

    if not _usb_task_started:
        print("[SYSTEM] USB Task 최초 실행")
        _usb_task_started = True
        socketio.start_background_task(run_usb_stream, socketio)


@socketio.on("usb_start_detection")
def handle_usb_start():
    """
    USB 카메라 탐지 시작 이벤트.
    """
    print("[SOCKET] USB 탐지 시작")
    set_usb_detection_active(True)


@socketio.on("usb_stop_detection")
def handle_usb_stop():
    """
    USB 카메라 탐지 중지 이벤트.
    """
    print("[SOCKET] USB 탐지 중지")
    set_usb_detection_active(False)


# --------------------------------------------------
# 12. ITS CCTV 목록 조회 API
# --------------------------------------------------
@app.route("/api/its/cctv-list")
def its_cctv_list():
    """
    its_stream.html에서 CCTV 목록을 불러올 때 호출하는 내부 API.

    전체 흐름:
    1. 프론트에서 /api/its/cctv-list?cctvType=2 형태로 요청
    2. Flask 서버가 국가교통정보센터 OpenAPI를 호출
    3. 응답 데이터에서 실제 영상 주소인 cctvurl을 추출
    4. 프론트가 사용하기 쉬운 형태로 변환해서 반환

    주의:
    ITS_CCTV_API_URL 자체는 영상 주소가 아니다.
    실제 영상 주소는 API 응답 안의 cctvurl 필드다.
    """
    cctv_type = request.args.get("cctvType", "1")

    if not ITS_API_KEY:
        print("[ITS ERROR] ITS_API_KEY 없음")
        return jsonify({
            "success": False,
            "message": "ITS_API_KEY가 .env에 설정되어 있지 않습니다.",
            "data": []
        }), 500

    # cctvType 기준:
    # 1: 도시고속도로
    # 2: 일반국도
    # 3: 고속도로
    #
    # 국가교통정보센터 API에서는 고속도로 계열은 type=ex,
    # 일반 ITS 계열은 type=its로 요청한다.
    road_type = "ex" if cctv_type == "3" else "its"

    params = {
        "apiKey": ITS_API_KEY,
        "type": road_type,
        "cctvType": cctv_type,

        # 전국 범위 좌표
        # 현재는 테스트 편의상 전국 범위로 설정
        # 추후 성능 개선 시 지역 단위로 좁히는 것이 좋음
        "minX": "124.0",
        "maxX": "132.0",
        "minY": "33.0",
        "maxY": "39.0",

        "getType": "json"
    }

    try:
        res = requests.get(
            ITS_CCTV_API_URL,
            params=params,
            timeout=10
        )

        # API 키가 포함된 실제 URL은 그대로 출력하지 않는다.
        safe_url = mask_api_key_in_url(res.url)

        print("[ITS] request url:", safe_url)
        print("[ITS] status code:", res.status_code)
        print("[ITS] response preview:", res.text[:500])

        res.raise_for_status()
        json_data = res.json()

        raw_items = (
            json_data
            .get("response", {})
            .get("data", [])
        )

        data = []

        for item in raw_items:
            cctv_url = item.get("cctvurl")

            # 실제 영상 주소가 없는 CCTV는 화면에 보여줘도 재생할 수 없으므로 제외
            if not cctv_url:
                continue

            data.append({
                "name": item.get("cctvname", "이름 없는 CCTV"),
                "url": cctv_url,
                "x": item.get("coordx", ""),
                "y": item.get("coordy", "")
            })

        print(f"[ITS] CCTV count: {len(data)}")

        return jsonify({
            "success": True,
            "message": "CCTV 목록 조회 성공",
            "data": data
        })

    except Exception as e:
        print("[ITS ERROR]", e)

        return jsonify({
            "success": False,
            "message": str(e),
            "data": []
        }), 500


# --------------------------------------------------
# 13. ITS CCTV 스트리밍 처리
# --------------------------------------------------
def run_its_stream():
    """
    ITS CCTV 영상 URL을 OpenCV로 열고,
    프레임을 base64 문자열로 변환해서 프론트로 전송하는 백그라운드 작업.

    프론트 its_stream.html은 아래 이벤트를 기다린다.
        socket.on("its_frame", ...)

    서버는 프레임을 읽을 때마다 아래 형태로 전송한다.
        socketio.emit("its_frame", {"frame": frame_base64})

    현재 버전:
    - ITS CCTV 영상 스트리밍 중심
    - YOLO 탐지는 아직 연결하지 않음
    - 추후 YOLO 탐지를 붙이면 its_detection_result에 탐지 결과를 넣으면 됨
    """
    global _its_stream_active, _its_selected_url

    print("[ITS STREAM] background task started")

    cap = None
    last_url = None

    while True:
        # 스트리밍이 꺼져 있으면 루프를 과도하게 돌지 않도록 대기
        if not _its_stream_active:
            time.sleep(0.2)
            continue

        # 선택된 CCTV URL이 없으면 대기
        if not _its_selected_url:
            time.sleep(0.2)
            continue

        # 새 CCTV를 선택했거나 기존 VideoCapture가 없으면 새로 연결
        if cap is None or last_url != _its_selected_url:
            if cap is not None:
                cap.release()

            last_url = _its_selected_url
            print(f"[ITS STREAM] CCTV 연결 시도: {last_url}")

            cap = cv2.VideoCapture(last_url)

            if not cap.isOpened():
                print("[ITS STREAM WARN] CCTV 열기 실패")
                socketio.emit("its_detection_result", {
                    "detections": []
                })
                time.sleep(1)
                continue

            print("[ITS STREAM] CCTV 연결 성공")

        ret, frame = cap.read()

        # 프레임 읽기에 실패하면 연결을 끊고 다음 루프에서 재연결 시도
        if not ret or frame is None:
            print("[ITS STREAM WARN] 프레임 읽기 실패 - 재연결 예정")

            if cap is not None:
                cap.release()
                cap = None

            time.sleep(1)
            continue

        # 전송량 감소를 위해 프레임 크기 축소
        frame = cv2.resize(frame, (960, 540))

        # OpenCV 프레임을 JPEG로 인코딩
        ok, buffer = cv2.imencode(".jpg", frame)

        if not ok:
            print("[ITS STREAM WARN] JPEG 인코딩 실패")
            continue

        # JPEG 바이너리를 base64 문자열로 변환
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        # 프론트로 영상 프레임 전송
        socketio.emit("its_frame", {
            "frame": frame_base64
        })

        # 현재는 ITS 영상에 YOLO 탐지를 연결하지 않았으므로 빈 결과 전송
        socketio.emit("its_detection_result", {
            "detections": []
        })

        # CPU 과부하 방지
        time.sleep(0.03)


@socketio.on("its_select_cctv")
def handle_its_select_cctv(data):
    """
    ITS CCTV 목록에서 특정 CCTV를 클릭했을 때 실행되는 이벤트.

    its_stream.html 내부에서:
        socket.emit("its_select_cctv", { url });

    형태로 선택한 CCTV 영상 URL을 서버로 보낸다.

    서버 역할:
    1. 선택한 URL을 전역 상태에 저장
    2. ITS 스트리밍 활성화
    3. ITS 백그라운드 작업이 아직 없으면 최초 1회 실행
    """
    global _its_selected_url, _its_task_started, _its_stream_active

    selected_url = data.get("url")

    if not selected_url:
        print("[ITS STREAM ERROR] 선택된 CCTV URL 없음")
        return

    print(f"[ITS STREAM] 선택된 CCTV URL: {selected_url}")

    _its_selected_url = selected_url
    _its_stream_active = True

    if not _its_task_started:
        print("[ITS STREAM] Task 최초 실행")
        _its_task_started = True
        socketio.start_background_task(run_its_stream)


@socketio.on("its_start_detection")
def handle_its_start_detection():
    """
    ITS 페이지의 '탐지 시작' 버튼 이벤트.

    현재 버전에서는 YOLO 탐지를 아직 붙이지 않았기 때문에
    스트리밍 활성화 역할로 사용한다.

    추후 확장:
    - YOLO 모델 연결
    - 탐지 플래그 추가
    - its_detection_result로 탐지 결과 전송
    """
    global _its_stream_active

    print("[ITS STREAM] 탐지/스트리밍 시작")
    _its_stream_active = True


@socketio.on("its_stop_detection")
def handle_its_stop_detection():
    """
    ITS 페이지의 '탐지 중지' 버튼 이벤트.

    현재는 스트리밍 루프를 대기 상태로 전환한다.
    """
    global _its_stream_active

    print("[ITS STREAM] 탐지/스트리밍 중지")
    _its_stream_active = False


@socketio.on("its_disconnect_stream")
def handle_its_disconnect_stream():
    """
    ITS 페이지의 '연결 끊기' 버튼 이벤트.

    선택된 CCTV URL을 비우고 스트리밍을 중지한다.
    """
    global _its_selected_url, _its_stream_active

    print("[ITS STREAM] 연결 끊기")
    _its_selected_url = None
    _its_stream_active = False


# --------------------------------------------------
# 14. 서버 실행
# --------------------------------------------------
if __name__ == "__main__":
    socketio.run(
        app,
        host="0.0.0.0",
        port=5001,
        debug=False,

        # Flask-SocketIO에서 Werkzeug 개발 서버를 사용할 때 필요한 옵션
        allow_unsafe_werkzeug=True
    )