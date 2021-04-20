import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import tf_slim as slim

options = {
 'model': 'yolov2-obj-helemt.cfg',
 'load': 'bin/yolo-obj_best-helmet.weights',
 'threshold': 0.6,
 'gpu': 0
    
}
tfnet = TFNet(options)
cap = cv2.VideoCapture('h25.mp4')
path = 'HelmetDataset/'
colors=[tuple(255 * np.random.rand(3)) for i in range(5)]
while(cap.isOpened()):
    stime= time.time()
    ret, frame = cap.read()
    results = tfnet.return_predict(frame)
    if ret:
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            confidence = result['confidence']
            frame= cv2.rectangle(frame, tl, br, color, 7)
            frame= cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
        cv2.imshow('frame', frame)
        image = np.array(frame)
        if confidence > 0.8 and label == "Helmet" :    
          cv2.imwrite(os.path.join(path , 'helmet.png'), image)
        print('FPS {:1f}'.format(1/(time.time() -stime)))
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
