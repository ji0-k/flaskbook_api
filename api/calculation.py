from pathlib import Path

import cv2
import numpy as np
import torch
from flask import current_app, jsonify

from flaskbook_api.api.postprocess import draw_lines, draw_texts, make_color, make_line
from flaskbook_api.api.preparation import load_image
from flaskbook_api.api.preprocess import image_to_tensor

basedir = Path(__file__).parent.parent

def detection(request):
    dict_results = {}
    # 라벨 읽어 들이기
    labels = current_app.config["LABELS"]
    # 이미지 읽어 들이기
    image, filename = load_image(request)
    # 이미지 데이터를 텐서 타입의 수치 데이터로 변경
    image_tensor = image_to_tensor(image)

    # 학습 완료 모델의 읽어 들이기
    try:
        model = torch.load(str(basedir / "model.pt"), weights_only=False)
    except FileNotFoundError:
        return jsonify("The model is not found"), 404

    # 모델의 실행
    model.eval()
    output = model([image_tensor])

    # 출력 데이터의 취득
    boxes = output[0]["boxes"].detach().numpy()
    labels_idx = output[0]["labels"].detach().numpy()
    scores = output[0]["scores"].detach().numpy()

    # 결과 이미지 작성 준비
    result_image = np.array(image)

    # 감지 결과를 이미지에 덧붙여 씀
    for box, label_idx, score in zip(boxes, labels_idx, scores):
        if score > 0.8:
            color = make_color(labels)
            line = make_line(result_image)
            c1 = (int(box[0]), int(box[1]))
            c2 = (int(box[2]), int(box[3]))
            draw_lines(c1, c2, result_image, line, color)
            display_txt = f"{labels[label_idx]}: {score:.1f}"
            draw_texts(result_image, line, c1, color, display_txt)
            dict_results[labels[label_idx]] = score.item()

    # 결과 이미지 저장
    dir_output = basedir / "data" / "output" / filename
    cv2.imwrite(str(dir_output), result_image)

    return jsonify(dict_results)