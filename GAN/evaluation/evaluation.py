import sys
import numpy as np
import argparse
import cv2
import scipy.io
import os
import operator

ImageLoc = "/media/shadowwalker/DATA/projects/stargan/GAN/evaluation/evaluation_set"

'''
NameDict =	{
  "158" : "Citroen"
  "78" : "Audi",
  "81" : "BWM",
  "77" : "Benz", 
  "73" : "Volkswagen",
  "95" : "Skoda",
  "111" : "Volvo"
}

'''
#make a map of name and sum of prob

dict = {}

NameDict =	{
    "111" : "Volvo"
}

arri = ['111']

def func(arr, out) :
    i = 0
    k = 0
    j = i + 1
    score = 0.0
    
    while (i < len(arr)) :
        score = 0.0
        while ((j < len(arr)) and (arr[j][0][0] == arr[i][0][0])) :
            score = score + out[i]
            i = i + 1
            j = j + 1            
            #print(arr[i][0][0], score)
        if (j < len(arr)) :    
            i = j
            j = j + 1
        if (i == 430) :
            break
        
        dict.update({arr[i - 1][0][0]:score})
    dict['Citroen'] = dict['Citroen'] + dict['DS']
    model = max(dict.iteritems(), key=operator.itemgetter(1))[0]
    return (dict[model], model)
    
             

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

k = 0
for subdir, dirs, files in os.walk(ImageLoc):

    for file in files:
        fileloc = os.path.join(subdir, file)
        sp = fileloc.split("/")
        arr = sp[9].split("-")
        arr[0] = str(arr[0])
        if (arr[0] == arri[0]) :
            frame = cv2.imread(fileloc)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (224, 224)), 1, (224, 224))
            net.setInput(blob)
            detections = net.forward()
            
            #k = k + 1
            
            x, out = func(mat['make_model_names'], detections[0])
            #res = np.where(detections[0] == np.amax(detections[0]))
            #out = mat['make_model_names'][res]
            #print(x, out, file)
            total = total + 1
            if (NameDict[arr[0]] in out) :
                correct = correct + 1
            elif (NameDict[arr[0]] == "Citroen" and "DS" in out) :
                correct = correct + 1
        
print(correct, total)
