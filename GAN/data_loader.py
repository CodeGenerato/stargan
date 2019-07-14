from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import utils
import numpy as np
from matplotlib import colors


class CarDataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, domains, image_dir, transform, hold_out_size):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.domains=domains
        self.sample_count = {}
        self.img_cache = {}

      
        self.valid_set_size=hold_out_size
        if self.valid_set_size >0:
            self.valid_set_dir="./validation_set"
            utils.create_empty_dir(self.valid_set_dir)
            
        self.preprocess()
        
        self.num_images = len(self.dataset)
      
            

    def preprocess(self):
        """Preprocess the attribute file."""
      
        file_name_list = os.listdir(self.image_dir)
        random.seed(1234)
        random.shuffle(file_name_list)
      
        for i,d in enumerate(self.domains):
              self.attr2idx[d]=i          

        for i, file_name in enumerate(file_name_list):
            if (file_name.startswith('X_')):
                continue
            
            parts = file_name.split("-")
            label = int(parts[0])
            if label not in self.domains:
                continue
            img_name = file_name

            count=self.get_sample_count(label)
            if count<self.valid_set_size:
                # create holdout set on the fly
                utils.copy_file(self.image_dir,self.valid_set_dir,img_name)
            else:
                self.dataset.append([img_name, self.attr2idx[label]])
                
            self.increment_sample_count(label)

        print("Sample count per domain: "+str(self.sample_count)+" (including holdout set, holdout size per domain is: "+str(self.valid_set_size)+")")
        print('Finished preprocessing the dataset...')

    def increment_sample_count(self,label):
        if label not in self.sample_count:
            self.sample_count[label]=0
        
        self.sample_count[label]=self.sample_count[label]+1

    def get_sample_count(self,label):
        if label not in self.sample_count:
            return 0
        else:
            return self.sample_count[label]

    def hsv_color_change(self,img,hshift):
        color_shift=random.uniform(-hshift,hshift)
        
        np_img=np.array(img)
        hsv=colors.rgb_to_hsv(np_img/255)
        hsv=np.add(hsv,np.array([color_shift,0,0]))
        hsv=np.mod(hsv,1+np.finfo(float).eps)
        rgb=colors.hsv_to_rgb(hsv)*255
        rgb=np.around(rgb)
       
        return Image.fromarray(rgb.astype('uint8'))

        # easy but slow:
        #color_shift=random.randint(-hshift,hshift)
        #hsv_img= img.convert('HSV')
        # for x in range(0, hsv_img.width - 1):
        #     for y in range(0,  hsv_img.height - 1):
        #         color=list(hsv_img.getpixel( (x,y) ))
        #         color[0]+=color_shift
        #         if(color[0] > 255):
        #             color[0]-=255
        #         if(color[0] <0):
        #             color[0]+=255
        #         hsv_img.putpixel( (x,y), tuple(color))

        #return hsv_img.convert('RGB')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset= self.dataset
        filename, label = dataset[index]
        
        path=os.path.join(self.image_dir, filename)
        if path not in self.img_cache:
            image = Image.open(path)
            image.load()
            self.img_cache[path]=image
        else:
            image=self.img_cache[path]
        
     
        encoded_lab=torch.zeros(len(self.domains), dtype=torch.float32)
        encoded_lab[label]=1
        #image=self.hsv_color_change(image,0.5)
        #im.save(self.image_dir+"/testimg.jpg")
        #image.save(self.image_dir+"/testimg2.jpg")
        return self.transform(image), encoded_lab

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(domains,image_dir, crop_size=178, image_size=128,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
   
    if mode == 'train':
       transform.extend([T.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1), T.RandomHorizontalFlip()])

    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size, interpolation=Image.LANCZOS))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),inplace=True))
    transform = T.Compose(transform)

    hold_out_size= 0 if mode == 'train' else 0
    dataset = CarDataset(domains,image_dir, transform, hold_out_size=hold_out_size)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader


