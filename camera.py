from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
from PIL import Image
import time

video = 0


class VideoCamera(object):
    def __init__(self, model,graph):
        global video
        self.video = cv2.VideoCapture(0)
        self.gender_model=model
        self.graph = graph
        self.detector = MTCNN()

    def __del__(self):
        global video
        self.video.release()

    def process_img(self,face):
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        face_array = face_array.reshape(1,224,224,3)
        with self.graph.as_default():
            gen = self.gender_model.predict(face_array)
            if gen[0][0] == 1:
                text = "MALE"
            else:
                text = "FEMALE"
        return text

    def get_frame(self):
        global video
        # grabs webcam image
        ret, frame = self.video.read()
        # DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
        # print(frame.shape)
        # detects faces with mtcnn. If no face detected. Except catches error and returns normal webcam image
        try:
            results = self.detector.detect_faces(frame)
        except:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
        # getting outline of face
        imshape = frame.shape
        x1, y1, width, height = results[0]['box']
        a,b,c,d = x1,y1,(x1+width),(y1+height) #Box for display only
        x1 = int(0.5*x1)
        y1 = int(0.5*y1)
        x2, y2 = x1 + width, y1 + height
        x2 = int(x2+0.5*(imshape[1]-x2))
        y2 = int(y2+0.5*(imshape[0]-y2))
        # cropping to face
        face = frame[y1:y2, x1:x2]
        # about bounding box
        text = self.process_img(face)
        cv2.rectangle(frame, (a, b), (c, d), (0, 255, 0), 2)
        cv2.putText(frame, text, (5,25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
        # ret, jpeg = cv2.imencode('.jpg', frame)
        # return jpeg.tobytes()
