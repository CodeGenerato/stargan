import numpy as np
import argparse
import cv2
import scipy.io

# construct the argument parse and parse the arguments
#install cv2 and caffe 
#conda install -c conda-forge opencv
#conda install -c conda-forge caffe
# run in python

def transformframe(frame) :
    # bring to range [0,1]
    frame = frame / 255.0
    # normalise with assignment mean and std dev
    frame[:,:,0] = (frame[:,:,0] - 0.5) / 0.5
    frame[:,:,1] = (frame[:,:,1] - 0.5) / 0.5
    frame[:,:,2] = (frame[:,:,2] - 0.5) / 0.5
    
    return frame

def classify(imageloc, net, mat) :
    frame = cv2.imread(imageloc)
    u1 = frame[:,:,0]
    u2 = frame[:,:,1]
    u3 = frame[:,:,2]
    
    v1 = np.sum(u1)/(256 * 256)
    v2 = np.sum(u2)/(256*256)
    v3 = np.sum(u3)/(256*256)

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224), (v1, v2, v3))
    net.setInput(blob)
    detections = net.forward()

    res = np.where(detections[0] == np.amax(detections[0]))
    print("confidence " + str(np.amax(detections[0])) + " make " + str(mat['make_model_names'][res]))



ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = 'deploy.prototxt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = 'googlenet_finetune_web_car_iter_10000.caffemodel' ,
	help="path to Caffe pre-trained model")

args = vars(ap.parse_args())
mat = scipy.io.loadmat('make_model_names_cls.mat')

# take the model
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
classify("m.jpg", net, mat)