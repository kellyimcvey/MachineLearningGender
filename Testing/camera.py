import keras_vggface
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

video = 0
detector = MTCNN()
class VideoCamera(object):
    def __init__(self):
    	global video
    	self.video = cv2.VideoCapture(0)

    def __del__(self):
    	global video
    	self.video.release()

    def get_frame(self):
    	global video
    	# grabs webcam image
    	ret, frame = self.video.read()
    	# DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
    	# print(frame.shape)
    	# detects faces with mtcnn. If no face detected. Except catches error and returns normal webcam image
    	results = detector.detect_faces(frame) #errors out if no face
    	#getting outline of face
    	x1, y1, width, height = results[0]['box']
    	x2, y2 = x1 + width, y1 + height
    	# cropping to face
    	face = frame[y1:y2, x1:x2]
    	# creates PIL image object
    	image = Image.fromarray(face)
    	image = image.resize((224, 224))
	    # run face array into keras model
	    face_array = np.asarray(image)
	    # MORE HERE
	    ret, jpeg = cv2.imencode('.jpg', face_array)
	    return jpeg.tobytes()
	    # ret, jpeg = cv2.imencode('.jpg', frame)
	    # return jpeg.tobytes()