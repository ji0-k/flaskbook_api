from datetime import datetime
from api.extensions import db


class DetectionSession(db.Model):
    """스트리밍 탐지 세션"""
    __tablename__ = "detection_sessions"

    id         = db.Column(db.Integer, primary_key=True)
    camera_url = db.Column(db.String(512), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at   = db.Column(db.DateTime, nullable=True)

    logs = db.relationship("DetectionLog", back_populates="session",
                           cascade="all, delete-orphan")

    def end(self):
        self.ended_at = datetime.utcnow()


class DetectionLog(db.Model):
    """프레임별 탐지된 객체 1건"""
    __tablename__ = "detection_logs"

    id          = db.Column(db.Integer, primary_key=True)
    session_id  = db.Column(db.Integer, db.ForeignKey("detection_sessions.id"),
                            nullable=False)
    class_name  = db.Column(db.String(100), nullable=False)   # 예: "person"
    confidence  = db.Column(db.Float, nullable=False)          # 0.0 ~ 1.0
    x1          = db.Column(db.Integer, nullable=False)
    y1          = db.Column(db.Integer, nullable=False)
    x2          = db.Column(db.Integer, nullable=False)
    y2          = db.Column(db.Integer, nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    session = db.relationship("DetectionSession", back_populates="logs")