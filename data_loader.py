from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        # lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        # all_attr_names = lines[1].split()
        # for i, attr_name in enumerate(all_attr_names):
        #     self.attr2idx[attr_name] = i
        #     self.idx2attr[i] = attr_name

        # read all file names, shuffle and create filename, label pairs
        file_name_list = os.listdir(self.image_dir)
        random.seed(1234)
        random.shuffle(file_name_list)

        
        #112 ferrari x
        #117 maserati y
        #139 ??     y
        #141 lotus -
        #142 landrover y
        #77 merceds y
        #81 bmw y
        self.attr2idx[77]=0
        self.attr2idx[81]=1
        self.attr2idx[117]=2
        self.attr2idx[139]=3
        self.attr2idx[142]=4

        for i, file_name in enumerate(file_name_list):
            parts = file_name.split("-")
            label = int(parts[0])
            if label not in (77,81,117,139,142):
                continue
            img_name = file_name
           
            self.train_dataset.append([img_name, self.attr2idx[label]])

        # lines = lines[2:]

        # for i, line in enumerate(lines):
        #     split = line.split()
        #     filename = split[0]
        #     values = split[1:]

        #     label = []
        #     for attr_name in self.selected_attrs:
        #         idx = self.attr2idx[attr_name]
        #         label.append(values[idx] == '1')

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))

        a=torch.zeros(5, dtype=torch.float32)
        a[label]=1
        return self.transform(image), a

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128,
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    # transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        raise AttributeError("RaFD not supported")

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader
