#TESTING
# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import pytesseract

# load the image
image = cv2.imread('C:/Users/Agnij/Desktop/BE_LP/Example/Examples (1).PNG')
orig = image.copy()
z = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (64, 64))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model('C:/Users/Agnij/Desktop/BE_LP/LP_NON_LP.model')

# classify the input image
(NONLP, LP) = model.predict(image)[0]

# build the label
label = "LP" if LP > NONLP else "NONLP"
proba = LP if LP > NONLP else NONLP
label = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = imutils.resize(orig, width=400)
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)


#PRE-PROCESSING BEFORE PYTESSERACT:
#BINARIZING
gray = cv2.cvtColor(z, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

Extracted_plate = pytesseract.image_to_string(thresh)
'''cv2.imshow("A", thresh)
cv2.waitKey(0)'''
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)











#DESKEWING PROCESS 
'''
#ROTATED BOUNDING BOX
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
    angle = -angle

(h, w) = thresh.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)	

cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
'''