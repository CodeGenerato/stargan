from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import utils


class CarDataset(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, domains, image_dir, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.domains=domains
        self.sample_count = {}

        self.valid_set_size=2
        self.valid_set_dir="./validation_set"
        utils.create_empty_dir(self.valid_set_dir)

        self.preprocess()
        

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

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
                self.train_dataset.append([img_name, self.attr2idx[label]])
                
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

   

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        
        encoded_lab=torch.zeros(len(self.domains), dtype=torch.float32)
        encoded_lab[label]=1
        return self.transform(image), encoded_lab

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(domains,image_dir, crop_size=178, image_size=128,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CarDataset(domains,image_dir, transform, mode)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader