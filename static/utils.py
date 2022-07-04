import io
import cv2
import math
import onnx
import pickle
import base64
import numpy as np
import onnxruntime as ort

from PIL import Image

import warnings
warnings.filterwarnings("ignore")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


#########################################################################################################

class CFG(object):
    def __init__(self, mode: str=None):
        self.mode = mode
        self.ort_session = None

        if self.mode == "brain-abnormality": 
            self.path = "static/brain-abnormality-segmentation-model.onnx"
            self.size = 256
        elif self.mode == "pneumonia": 
            self.path = "static/pneumonia-prediction-model.onnx"
            self.size = 384
        elif self.mode == "tuberculosis": 
            self.path = "static/tuberculosis-prediction-model.onnx"
            self.size = 384

    def setup(self):
        model = onnx.load(self.path)
        onnx.checker.check_model(model)
        self.ort_session = ort.InferenceSession(self.path)
    
    def infer(self, image: np.ndarray):
        im_h, im_w, im_c = image.shape

        if im_c != 3: image = np.concatenate((np.expand_dims(image, axis=2), 
                                            np.expand_dims(image, axis=2), 
                                            np.expand_dims(image, axis=2)), axis=2)
        # image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)

        if im_h != self.size or im_w != self.size: image = cv2.resize(src=image, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        image = image / 255

        if self.mode == "brain-abnormality":
            for i in range(image.shape[0]): image[i, :, :] = (image[i, :, :] - image[i, :, :].mean()) / image[i, :, :].std()
            
            image = np.expand_dims(image, axis=0)

            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
            result = self.ort_session.run(None, input)

            return np.round(result[0].squeeze())
        
        elif self.mode == "pneumonia":
            for i in range(image.shape[0]): image[i, :, :] = (image[i, :, :] - image[i, :, :].mean()) / image[i, :, :].std()

            image = np.expand_dims(image, axis=0)

            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
            result = self.ort_session.run(None, input)

            return sigmoid(result[0].squeeze())
        
        elif self.mode == "tuberculosis":
            for i in range(image.shape[0]): image[i, :, :] = (image[i, :, :] - image[i, :, :].mean()) / image[i, :, :].std()

            image = np.expand_dims(image, axis=0)

            input = {self.ort_session.get_inputs()[0].name : image.astype("float32")}
            result = self.ort_session.run(None, input)

            return sigmoid(result[0].squeeze())

#########################################################################################################

def clean_data(old_data: str) -> np.ndarray:
    old_data = [data for data in old_data.split(",")]
    new_data = []
    for d in old_data:
        if "[" in d: d = d.replace("[", "")
        if "]" in d: d = d.replace("]", "")
        new_data.append(int(d))
    return np.array(new_data).reshape(1, -1)


def infer_diabetes_model(data: np.ndarray) -> tuple:
    model = pickle.load(open("static/diabetes-model.pkl", "rb"))
    y_pred = model.predict(data)[0]
    y_pred_proba = model.predict_proba(data)[0][1]

    if y_pred == 0: 
        return "No Diabetes", y_pred_proba
    else: 
        return "Diabeties", y_pred_proba


def infer_liver_disease_model(data: np.ndarray) -> tuple:
    model = pickle.load(open("static/liver-disease-model.pkl", "rb"))
    y_pred = model.predict(data)[0]
    y_pred_proba = model.predict_proba(data)[0][1]

    if y_pred == 0: 
        return "No Liver Disease", y_pred_proba
    else: 
        return "Liver Disease", y_pred_proba


def infer_cardiovascular_disease_model(data: np.ndarray) -> tuple:
    model = pickle.load(open("static/cardiovascular-disease-model.pkl", "rb"))
    y_pred = model.predict(data)[0]
    y_pred_proba = model.predict_proba(data)[0][1]

    if y_pred == 0: 
        return "No Cardiovascular Disease", y_pred_proba
    else: 
        return "Cardiovasculer Disease", y_pred_proba


#########################################################################################################

def decode_data(data: str) -> np.ndarray:
    encoded_image = np.array(bytearray(data), dtype="uint8")
    return cv2.cvtColor(src=cv2.imdecode(encoded_image, cv2.IMREAD_COLOR), code=cv2.COLOR_BGRA2RGB)


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    return header, image


def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".png", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'","").replace("'","")
    imageData = header + "," + imageData
    return imageData

#########################################################################################################
