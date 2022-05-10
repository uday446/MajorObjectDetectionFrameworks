from flask import Flask, request, jsonify, render_template,Response
import os
from flask_cors import CORS, cross_origin
from utils.com_ineuron_utils.utils import decodeImage
from yolo.detect import Detector
from Detect2.ObjectDetector import Detector2
from tf2od.detect import Predictor
import threading
from model_threder import thredd
import time
# import sys
# sys.path.insert(0, './com_ineuron_apparel')

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')


app = Flask(__name__)
CORS(app)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.yolov5weights = "yolo/My_model/Yolov5s/best.pt"
        self.yolov5output_name = "yolov5s.jpg"
        self.yolov5moutput_name = "yolov5m.jpg"
        self.objectDetectionYolo = Detector(self.yolov5weights,self.yolov5output_name)
        self.yolov5mweights = "yolo/My_model/Yolov5m/best.pt"
        self.objectDetectionYolov5m = Detector(self.yolov5mweights,self.yolov5moutput_name)
        self.img = 'yolo/inference/images/inputImage.jpg'
        self.modelretina = 'retinanet_R_50_FPN_3x.yaml'
        self.yamlpathretina = 'Detect2/Retinanet_r50_3x/retinanet_R_50_FPN_3x.yaml'
        self.weightretina = 'Detect2/Retinanet_r50_3x/model_final_5bd44e.pkl'
        self.outputretina = 'retinanet.jpg'
        self.objectDetectionDetectron2retina = Detector2(self.modelretina, self.yamlpathretina, self.weightretina,self.outputretina,self.img)
        self.modelrcnn = 'faster_rcnn_R_50_FPN_1x.yaml'
        self.yamlpathrcnn = 'Detect2/Faster_rcnn_r50/faster_rcnn_R_50_FPN_1x.yaml'
        self.weightrcnn = 'Detect2/Faster_rcnn_r50/model_final_b275ba.pkl'
        self.outputrcnn = 'fasterrcnn.jpg'
        self.objectDetectionDetectron2rcnn = Detector2(self.modelrcnn, self.yamlpathrcnn, self.weightrcnn, self.outputrcnn, self.img)
        self.modelresnet = 'tf2od\my_model\Resnet50v1\saved_model'
        self.outputresnet = 'tf2od\output\Resnet.jpg'
        self.objectDetectionTfod2resnet = Predictor(self.modelresnet, self.outputresnet)
        self.modelefficient = 'tf2od\my_model\Efficientdet_d0\saved_model'
        self.outputefficient = 'tf2od\output\Efficientnet.jpg'
        self.objectDetectionTfod2efficient = Predictor(self.modelefficient, self.outputefficient)
        self. th = thredd(self.objectDetectionYolo,self.objectDetectionYolov5m,self.objectDetectionDetectron2retina,self.objectDetectionDetectron2rcnn,self.objectDetectionTfod2resnet,self.objectDetectionTfod2efficient)

@app.route("/predictall")
def predictAllRoute():
    #Create new threads
    start = time.time()
    threading.Thread(target=thredd.yolov5s,args=(clApp.th,)).start()
    threading.Thread(target=thredd.yolov5m,args=(clApp.th,)).start()
    threading.Thread(target=thredd.tfod2_resnet,args=(clApp.th,)).start()
    threading.Thread(target=thredd.tfod2_efficient,args=(clApp.th,)).start()
    threading.Thread(target=thredd.detectron2_fasterrcnn,args=(clApp.th,)).start()
    threading.Thread(target=thredd.detectron2_retinanet,args=(clApp.th,)).start()
    print(time.time()-start)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    try:
        image = request.json['image']
        decodeImage(image, clApp.filename)
        #cv2.imwrite("yolo/inference/images/inputImage.jpg",image)
        framework = request.json['frame']
        model = request.json['model']
        result = choose(framework,model)
        return jsonify(result)
    except ValueError as val:
        print(val)
        return Response("Value not found inside  json data")
    except KeyError:
        return Response("Key value error incorrect key passed")
    except Exception as e:
        print(e)
        result = "Invalid input"

    return jsonify(result)

def choose(framework,model):
    mod = model
    if framework == "yolo" and mod == "yolov5s":
        result = clApp.objectDetectionYolo.detect_action()
    elif framework == "yolo" and mod == "yolov5m":
        result = clApp.objectDetectionYolov5m.detect_action()
    elif framework == "detectron2" and mod == "Retinanet_r50_3x":
        result = clApp.objectDetectionDetectron2retina.inference()
    elif framework == "detectron2" and mod == "Faster_rcnn_r50":
        result = clApp.objectDetectionDetectron2rcnn.inference()
    elif framework == "tfod2" and mod == "Efficientdet_d0":
        result = clApp.objectDetectionTfod2efficient.run_inference()
    else:
        result = clApp.objectDetectionTfod2resnet.run_inference()
    return result

#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    port = 9500
    app.run(host='127.0.0.1', port=port)