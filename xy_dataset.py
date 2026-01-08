import torch
import os
import glob
import uuid
import PIL.Image
import torch.utils.data
import subprocess
import cv2
import numpy as np

# EDITED by George Gorospe, george.gorospe@nmaia.net
# Updated: 1/3/2026 - updated for 1-1000 label
# This file was originally written by Nvidia and included in their Jetracer github repo.
# Source: https://github.com/NVIDIA-AI-IOT/jetracer

# About the changes: I've modified this custom dataset class to permit the addition of datasets collected
# and labeled through some of my own tools.
# There is one new method, update_directory() that checks for the datapoints in the given directory and includes in the dataset.

# More information: A custom Dataset class must implement three core functions: __init__, __len__, and __getitem__.
# For this dataset, categories have been removed.

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, random_hflip=False):
        super(XYDataset, self).__init__()
        self.directory = directory
        self.create_directory()
        self.transform = transform
        self.refresh()
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.annotations)
    
    # Get Item: 
    # From the list of annotations retreive indexed file and read
    # PIL.Image.fromarray(image)
    # Transform image: if the transform flag is positive, applytransform, set new x and y.
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = cv2.imread(ann['image_path'], cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        
        x = 2.0 * (ann['x'] / 1000 - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        #x = ann['x']
        #y = ann['y']

        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
        return image, ann, torch.Tensor([x,y])   
    
    def _parse(self, path):
        basename = os.path.basename(path)
        items = basename.split('_')
        x = items[0]
        y = items[1]
        return int(x), int(y)
    
    def create_directory(self):
        if not os.path.exists(self.directory):
            print("Creating New Dataset Directory")
            subprocess.call(['mkdir', '-p', self.directory])
        else:
            print("Dataset directory exists.")
            num_files =  len(glob.glob(os.path.join(self.directory, '*.jpg')))
            print("Number of files in datadset: " + str(num_files))
        
        
    def refresh(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,

                'x': x,
                'y': y
            }]
    
    def save_entry(self, image, x, y):
        dataset_dir = os.path.join(self.directory)
        if not os.path.exists(dataset_dir):
            subprocess.call(['mkdir', '-p', dataset_dir])
            
        filename = '%d_%d_%s.jpg' % (x, y, str(uuid.uuid1()))
        
        image_path = os.path.join(dataset_dir, filename)
        cv2.imwrite(image_path, image)
        self.refresh()
    

    
    # Add directory of images (G. Gorospe 2021)
    def update_directory(self):
        self.annotations = []
        for image_path in glob.glob(os.path.join(self.directory, '*.jpg')):
            x, y = self._parse(image_path)
            self.annotations += [{
                'image_path': image_path,
                'x': x,
                'y': y
            }]
    
    def get_count(self):
        i = 0
        for a in self.annotations:
            #if a['category'] == category:
            i += 1
        return i


class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap