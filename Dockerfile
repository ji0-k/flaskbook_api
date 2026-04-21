# 베이스 이미지 - Python 3.11 슬림 버전
FROM python:3.11-slim-bullseye

# 작업 디렉터리 설정
WORKDIR /app

# OpenCV 실행에 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 먼저 복사 후 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch torchvision torchaudio && \
    pip install --no-cache-dir -r requirements.txt

# 앱 전체 복사
COPY . .

# 환경변수 설정
ENV FLASK_APP=run.py
ENV CONFIG=local
ENV PYTHONPATH=/app

# 포트 설정
EXPOSE 5000

# 컨테이너 실행 시 Flask 서버 시작
CMD ["flask", "run", "--host=0.0.0.0"]