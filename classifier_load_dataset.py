import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os 
import json
import matplotlib.pyplot as plt
import glob
import numpy as np
import random
import json


class my_dataset(Dataset):
    def __init__(self, transform=None, num_samples=5000, dataset="", configs="", training=True, n_class_color=None, n_class_size=None, test_size=1.3, alpha=1.0, flag_four=False):
        self.training = training
        self.test_size = test_size
        self.flag_four = flag_four

        self.dataset = dataset
        self.n_class_size = n_class_size
        self.n_class_color = n_class_color

        if training:
            self.train_image_paths = []
            for config in configs:
                new_paths = glob.glob(dataset+"/*/CLEVR_"+config+"_*.png")
                self.train_image_paths += new_paths
        else:
            self.test_image_paths = glob.glob(dataset+"/test/CLEVR_"+configs+"_*.png")

        if self.training: 
           self.len_data = len(self.train_image_paths) - 1
        else:
           self.len_data = len(self.test_image_paths) - 1

        self.num_samples = num_samples
        self.transform = transform


    def __getitem__(self, index):
       if self.training:
           ipath = random.randint(0, len(self.train_image_paths)-1)
           img_path = self.train_image_paths[ipath]
       else:
           ipath = random.randint(0, len(self.test_image_paths)-1)
           img_path = self.test_image_paths[ipath]
            
       img = Image.open(img_path) #.convert('RGB')
       if self.transform is not None:
           img = self.transform(img)
        
       json_path = img_path.replace(".png", ".json")
       with open(json_path, 'r') as f: 
           features = json.load(f)
           if self.dataset == "single-body_2d_3classes": 
               size = features[0]
               color = features[1][0]
           if self.dataset == "single-body_3d_3classes": 
               size = features[0]
               color = features[1][0]
           if "single-body_2d_4classes" in self.dataset:
               size = features[0]
               color = features[1][0]
               position = features[-1]

       name_labels = img_path.split("_")[-2]
       if "single-body_2d_3classes" in self.dataset or "single-body_3d_3classes" in self.dataset:
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: int(name_labels[2])}
       elif "single-body_2d_4classes" in self.dataset:
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: int(name_labels[2]), 3: int(name_labels[3])}

       return img, label 

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = my_dataset(transform, dataset="single-body_2d_color_0.05", n_class_size=1, n_class_color=1, configs=["000","010","100","001"])
    dataloader = DataLoader(dataset, batch_size=4)

