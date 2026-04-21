from flask import Blueprint, jsonify, request

from api import calculation

api = Blueprint("api", __name__)

@api.route("/detect", methods=["POST"])
def detect():
    """물체 감지 API"""
    return calculation.detection(request)