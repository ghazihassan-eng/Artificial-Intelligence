from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import tf_slim as slim
import time
import numpy as np
import os
import cv2

options = {
    'model': 'yolov2-obj-helemt.cfg',
    'load':'bin/yolo-obj_best-helmet.weights',
    'threshold': 0.8,
    'gpu': 0
}

tfnet = TFNet(options)

filename = 'video125.avi'
frames_per_second = 244.0
res = '720p'

# Set resolution for the video capture
# Function adapted from https://kirr.co/0l6qmh
def change_res(capture, width, height):
    capture.set(3, width)
    capture.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


# grab resolution dimensions and set video capture to it.
def get_dims(capture, res='1080p'):
    width, height = STD_DIMENSIONS["480p"]
    if res in STD_DIMENSIONS:
        width,height = STD_DIMENSIONS[res]
    ## change the current caputre device
    ## to the resulting resolution
    change_res(capture, width, height)
    return width, height

# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']



capture = cv2.VideoCapture(0)
path = 'Helmet-DetectingImage/'
output = cv2.VideoWriter(filename, get_video_type(filename), 25, get_dims(capture, res))

colors = [tuple(255 * np.random.rand(3)) for _ in range(5)]

i = 0
frame_rate_divider = 3
while(capture.isOpened()) :
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        if i % frame_rate_divider == 0:
            results = tfnet.return_predict(frame)
        
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                label = result['label']
                confidence = result['confidence']
                text = '{}: {:.0f}%'.format(label, confidence * 100)
                frame = cv2.rectangle(frame, tl, br, color, 5)
                frame = cv2.putText(frame, text, tl,cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                image = np.array(frame)
            if confidence > 0.9 and label == "Helmet" :    
               cv2.imwrite(os.path.join(path , 'helmet.png'), image)   
            output.write(frame)
            cv2.imshow('frame', frame)
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            i +=1 
        else:
            i +=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        


#while True:
#    ret, frame = cap.read()
#    out.write(frame)
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break


capture.release()
output.release()
cv2.destroyAllWindows()
