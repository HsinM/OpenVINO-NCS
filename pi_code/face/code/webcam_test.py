# Reference
# https://docs.opencv.org/4.0.1/d6/d0f/group__dnn.html

import cv2
import imutils
from imutils.video import WebcamVideoStream

# Load the model
net = cv2.dnn.readNet('../model/face-detection-adas-0001.xml', '../model/face-detection-adas-0001.bin') 

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Read the Camera
vs = WebcamVideoStream(src=0).start()

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)


    # Prepare input blob and perform an inference
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U) 

    net.setInput(blob) 

    out = net.forward()

    # Draw detected faces on the frame
    for detection in out.reshape(-1, 7): 

        confidence = float(detection[2]) 

        xmin = int(detection[3] * frame.shape[1]) 
        ymin = int(detection[4] * frame.shape[0]) 

        xmax = int(detection[5] * frame.shape[1]) 
        ymax = int(detection[6] * frame.shape[0])

        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()