import base64
import json

def load_parameters_from_json():
    with open('config.json', 'r') as f:
        data = json.load(f)
        return data

def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("./yolo/inference/images/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
