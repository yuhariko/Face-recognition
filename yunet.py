from itertools import product

import numpy as np
import cv2 as cv

class Yunet:
    def __init__(self, modelPath = "face_detection_yunet_2023mar_int8.onnx", inputSize=[320,320], 
                 confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        """
        model:              the path to the requested model
        config:             the path to the config file for compability, which is not requested for ONNX models
        input_size:         the size of the input image
        score_threshold:    the threshold to filter out bounding boxes of score smaller than the given value
        nms_threshold:      the threshold to suppress bounding boxes of IoU bigger than the given value
        top_k:              keep top K bboxes before NMS
        backend_id:         the id of backend
        target_id:          the id of target device
        """
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize) #w, h
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId
        )
    
    def name(self):
        return self.__class__.__name__
    
    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId
        )

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        """
        0-1: x, y of bbox top left corner
        2-3: width, height of bbox
        4-5: x, y of right eye (blue point in the example image)
        6-7: x, y of left eye (red point in the example image)
        8-9: x, y of nose tip (green point in the example image)
        10-11: x, y of right corner of mouth (pink point in the example image)
        12-13: x, y of left corner of mouth (yellow point in the example image)
        14: face score
        """
          
        #forward
        faces = self._model.detect(image)
        return faces[1]

        