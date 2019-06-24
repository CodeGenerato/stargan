###################################################
#  This file does the preprocessing of the input  #
#  compCar Dataset. It iteratively selects each   #
#  image, crops the bounding box and saves the    #
#  result in the required folder.                 #
###################################################

import os
import sys
from PIL import Image


sys.path.append("../proc_stanford_car_dataset")
from  get_img_for_make import resize_aspec

rootdir = '/media/shadowwalkers/DATA/comp-cars/dataset/data'
finalDir = '/media/shadowwalkers/DATA/comp-cars/dataset/data/EnhancedDataset/'

# create a folder "Dataset" which would  contain the required images

for subdir, dirs, files in os.walk(rootdir):
   
    for file in files:
        location = os.path.join(subdir, file)
        if ".jpg" in location and "image" in location :
            labelLocation = location
            loc1 = labelLocation.find("image")
            if loc1 != -1 :
                labelLocation = labelLocation[0:loc1] + "label/" + labelLocation[loc1 + 6: len(labelLocation) - 3] + "txt"
                f = open(labelLocation, "r")
                content = f.read().split("\n")
                bBox = [(int(s) - 1) for s in content[2].split() if s.isdigit()]
                if content[0] == "1" :
                    try :
                        newFileName = location[55:].replace("/", "-")
                        imageObject = Image.open(location)

                        cropped = resize_aspec(imageObject, (bBox[0],bBox[1],bBox[2],bBox[3]), 256, True)
                        cropped.save(finalDir + newFileName, "JPEG")
                    except :
                        print("some error")
                f.close()
