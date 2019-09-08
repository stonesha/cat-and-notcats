from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import numpy as np
import argparse
import cv2

#construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True, help = "path to trained model")
ap.add_argument("-i", "--image", required = True, help = "path to input image")
args = vars(ap.parse_args())

#load the image
image = cv2.imread(args["image"])
orig = image.copy()
orig_height, orig_width, orig_channels = orig.shape

#pre-process the image for classification
image = cv2.resize(image, (256, 256))
image = image.astype("float") / 255.0
image = img_to_array(image).reshape((-1, 256, 256, 1))
#image = np.expand_dims(image, axis = 0)

#load  the trained model
print("Loading model...")
model = load_model(args["model"])

#classify the input image
(notcats, cats) = model.predict(image)[0]

#build label
label = "Cat" if cats > notcats else "Not Cat"
proba = cats if cats > notcats else notcats
label = "{}: {:.2f}%".format(label, proba * 100)

#draw the label on the image
output = cv2.resize(orig, (256, 256))
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()