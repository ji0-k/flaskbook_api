from pathlib import Path

import PIL

basedir = Path(__file__).parent.parent

def load_image(request, reshaped_size=(256, 256)):
    """이미지 읽어 들이기"""
    filename = request.json["filename"]
    dir_image = str(basedir / "data" / "original" / filename)
    # 이미지 데이터의 객체를 생성
    image_obj = PIL.Image.open(dir_image).convert("RGB")
    # 이미지 데이터의 크기 변경
    image = image_obj.resize(reshaped_size)
    return image, filename