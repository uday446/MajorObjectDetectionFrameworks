from flask import Flask, request, jsonify, render_template,Response
from yolo.detect import Detector
from Detect2.ObjectDetector import Detector2
from tf2od.detect import Predictor

class thredd:
    def __init__(self,yolov5s,yolov5m,retina,rcnn,resnet,efficient):
        self.objectDetectionYolo = yolov5s
        self.objectDetectionYolov5m = yolov5m
        self.objectDetectionDetectron2retina = retina
        self.objectDetectionDetectron2rcnn = rcnn
        self.objectDetectionTfod2resnet = resnet
        self.objectDetectionTfod2efficient = efficient
    def yolov5s(self):
        try:
            result = self.objectDetectionYolo.detect_action()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)

    def tfod2_efficient(self):
        try:
            result = self.objectDetectionTfod2efficient.run_inference()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)

    def tfod2_resnet(self):
        try:
            result = self.objectDetectionTfod2resnet.run_inference()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)

    def yolov5m(self):
        try:
            result = self.objectDetectionYolov5m.detect_action()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)

    def detectron2_retinanet(self):
        try:
            result = self.objectDetectionDetectron2retina.inference()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)

    def detectron2_fasterrcnn(self):
        try:
            result = self.objectDetectionDetectron2rcnn.inference()
            return "done"
        except ValueError as val:
            print(val)
            return Response("Value not found inside  json data")
        except KeyError:
            return Response("Key value error incorrect key passed")
        except Exception as e:
            print(e)
            result = "Invalid input"
        #return jsonify(result)
