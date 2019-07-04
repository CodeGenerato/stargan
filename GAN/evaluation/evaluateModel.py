import numpy as np
import argparse
import cv2
import scipy.io

# construct the argument parse and parse the arguments
#install cv2 and caffe 
#conda install -c conda-forge opencv
#conda install -c conda-forge caffe
# run in python

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = 'deploy.prototxt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = 'googlenet_finetune_web_car_iter_10000.caffemodel' ,
	help="path to Caffe pre-trained model")

args = vars(ap.parse_args())
mat = scipy.io.loadmat('make_model_names_cls.mat')

# take the model
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

frame = cv2.imread("m.jpg")
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)),
	1, (224, 224))

#feed input
net.setInput(blob)
detections = net.forward()

res = np.where(detections[0] == np.amax(detections[0]))
print("confidence " + str(np.amax(detections[0])) + " make " + str(mat['make_model_names'][res]))
