# Import the modules
import cv2
import numpy as np

import imutils
from imutils.video import WebcamVideoStream

# Custom functions
# define rotate function
def rotate(image, angle, center=None, scale=1.0):
    # get image size
    (h, w) = image.shape[:2]
 
    # if dosen't assign image center, set image center point as center
    if center is None:
        center = (w / 2, h / 2)
 
    # Do rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # return rotated image
    return rotated

# Load the model
net = cv2.dnn.readNet('../model/inference_graph.xml', '../model/inference_graph.bin') 

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Read the Camera
vs = WebcamVideoStream(src=0).start()

while true:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = rotate(frame, 180)

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    # For each rectangular region, use mnist cnn model to predict.
    for rect in rects:
        # Draw the rectangles
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        # detect empty roi
        if(roi.size == 0): break
        # Resize the image
        if(roi.shape > (10, 10)):
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))
            # Inference
            blob = cv2.dnn.blobFromImage(roi, size=(28, 28), ddepth=cv2.CV_32F)
            net.setInput(blob) 
            out = net.forward() 
            if out[0][int(np.argmax(out[0]))] > 0.5:
                #cv2.putText(image, text, coordinate, font, size, color, width of line, type of line) 
                #cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)            
                cv2.putText(frame, str(np.argmax(out[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()