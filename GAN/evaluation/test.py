from os.path import abspath
import sys
import numpy as np
import argparse
import cv2
import scipy.io

trainLoc = '/media/shadowwalker/DATA/comp-cars/dataset/data/train_test_split/classification/test.txt'
ImageLoc = "/media/shadowwalker/DATA/comp-cars/dataset/data/image/"

NameDict =	{
  "4" : "Citroen",
  "78" : "Audi",
  "81" : "BWM",
  "77" : "Benz",
  "73" : "Volkswagen",
  "95" : "Skoda",
  "111" : "Volvo"
}

arr = ['4', '78', '81', '77', '73', '95', '111']


total = 0
correct = 0

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default = 'deploy.prototxt',
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default = 'googlenet_finetune_web_car_iter_10000.caffemodel' ,
	help="path to Caffe pre-trained model")

args = vars(ap.parse_args())
mat = scipy.io.loadmat('make_model_names_cls.mat')

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
r = "/media/shadowwalker/DATA/comp-cars/dataset/data/image/78/1/2014/3ac218c0c6c378.jpg"
frame = cv2.imread(r)

with open(trainLoc,'rb') as f:
    img = [line.strip() for line in f]

for x in img :
    check = x.split('/')
    for i in range(len(arr)) :
        if (arr[i] == check[0]) :
            fileloc = ImageLoc + x
            frame = cv2.imread(fileloc)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224))
            net.setInput(blob)
            detections = net.forward()
            res = np.where(detections[0] == np.amax(detections[0]))
            out = str(mat['make_model_names'][res])
            total = total + 1
            if (NameDict[arr[i]] in out) :
                correct = correct + 1
            elif (NameDict[arr[i]] == "Citroen" and "DS" in out) :
                correct = correct + 1

print(correct, total)
  
