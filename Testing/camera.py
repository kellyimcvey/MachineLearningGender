import cv2
video = 0

class VideoCamera(object):
    def __init__(self):
    	global video
    	self.video = cv2.VideoCapture(0)

    def __del__(self):
    	global video
    	self.video.release()

    def get_frame(self):
    	global video
    	ret, frame = self.video.read()
    	# DO WHAT YOU WANT WITH TENSORFLOW / KERAS AND OPENCV
    	ret, jpeg = cv2.imencode('.jpg', frame)
    	return jpeg.tobytes()