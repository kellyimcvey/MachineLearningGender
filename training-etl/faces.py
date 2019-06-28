import pandas as pd
import keras_vggface
from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

names = pd.read_csv("male_names.txt",header=None)

names = names[0][3500:4000]
detector = MTCNN()
for name in tqdm(names):
	try:
		pixels = plt.imread(f"all_originals/{name}")
		imshape = pixels.shape
	except:
		continue
	# create the detector, using default weights
	
	# detect faces in the image
	try:
		results = detector.detect_faces(pixels)
	except:
		continue
	# extract the bounding box from the first face
	if results[0]["confidence"] < 0.98:
		continue
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	x1 = int(0.5*x1)
	y1 = int(0.5*y1)
	x2 = int(x2+0.5*(imshape[1]-x2))
	y2 = int(y2+0.5*(imshape[0]-y2))
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	try:
		image = Image.fromarray(face)
	except:
		continue
	# print(type(image))
	image = image.resize((224, 224))
	image.save(f"Train/male/{name}")
	face_array = np.asarray(image)