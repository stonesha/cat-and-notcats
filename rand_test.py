from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required = True, help = "directory of dataset")
ap.add_argument("-m", "--model", required = True, help = "path to trained model")
args = vars(ap.parse_args())

path = args["directory"]
if random.uniform(0, 1) == 1:
    path += "/cats"
else:
    path += "/notcats"


files = os.listdir(path)
index = random.randrange(0, len(files))
print("Image selected: ", files[index])
path += ("/" + files[index])

#load image
img = cv2.imread(path)
orig = img.copy()

#preprocess
img = cv2.resize(img, (256, 256))
img = img.astype("float") / 255.0
img = img_to_array(img).reshape((-1, 256, 256, 1))

#load model
print("Loading model")
model = load_model(args["model"])

#classify
(notcats, cats) = model.predict(img)[0]

#build label
label = "Cat" if cats > notcats else "Not Cat"
proba = cats if cats > notcats else notcats
label = "{}: {:.2f}%".format(label, proba * 100)

#draw label on image
output = cv2.resize(orig, (256, 256))
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255, 0), 2)

#shows output
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()