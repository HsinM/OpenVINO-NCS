# Reference
# https://docs.opencv.org/4.0.1/d6/d0f/group__dnn.html
# https://chtseng.wordpress.com/2019/01/21/intel-openvino%E4%BB%8B%E7%B4%B9%E5%8F%8A%E6%A8%B9%E8%8E%93%E6%B4%BE%E3%80%81linux%E7%9A%84%E5%AE%89%E8%A3%9D/

# os for 檔案列舉
import os

# cv2 for 圖像處理 & 推論(inference)
import cv2

# np for processing array
import numpy as np

# Load the model
net = cv2.dnn.readNet('../model/inference_graph.xml', '../model/inference_graph.bin') 

# Specify target device
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Load images
data_dir = "../data/"
images = []

for dirPath, dirNames, fileNames in os.walk(data_dir):
    #print(dirPath)
        print('檔案清單:')
        for f in fileNames:
            print(os.path.join(dirPath, f))
            
            img = cv2.imread(os.path.join(dirPath, f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.astype('float32') / 255
            
            images.append(img)
            
print('---共{}張---\n'.format(len(images)))
#print(images[0])
#print(img == images[9])

for i in range(len(images)):
    # Prepare input blob and perform an inference 
    blob = cv2.dnn.blobFromImage(images[i], size=(28, 28), ddepth=cv2.CV_32F)
    net.setInput(blob) 

    # Inference
    out = net.forward()

    # Print output
    #print(out[0])
    print('圖像中的數字為:{} (機率:{}%)'.format(np.argmax(out), out[0][np.argmax(out)]*100))
    
