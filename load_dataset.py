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
    def __init__(self, transform=None, num_samples=5000, dataset="", configs="", training=True, test_size=None, alpha=1.0, beta=2.0, remove_node=None, flag_double=1):
        self.training = training
        self.test_size = test_size

        self.dataset = dataset

        if "celeba" in dataset: 
            prefix = "celeba"
            ext = ".jpg"
        else:
            prefix = "CLEVR"
            ext = ".png"
        if training:
            self.train_image_paths = []
            for config in configs:
                if config=="000" and alpha!=1500 and remove_node!="100": 
                    new_paths = glob.glob("input/"+dataset+"/train_"+remove_node+"/"+prefix+"_000_*"+ext)
                else: 
                   new_paths = glob.glob("working/"+dataset+"/train/"+prefix+"_"+config+"_*"+ext)
                if remove_node==config: 
                    new_paths = new_paths[:alpha]
                self.train_image_paths += new_paths
        else:
            self.test_image_paths = glob.glob("input/"+dataset+"/test/"+prefix+"_"+configs+"_*"+ext)

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

       name_labels = img_path.split("_")[-2]
       if self.dataset=="single-body_2d_3classes": 
           with open(img_path.replace(".png", ".json"), 'r') as f:
               my_dict = json.loads(f.read())
               _size = my_dict[0]
               _color = my_dict[1][:3]

           if self.training:
               size = _size
               color = _color
           else:
              if int(name_labels[2])==0: size = 2.6
              if int(name_labels[2])==1: size = self.test_size
              if int(name_labels[1])==0: color = [ 0.9 , 0.1 , 0.1 ] 
              if int(name_labels[1])==1: color = [ 0.1 , 0.1 , 0.9 ] 
              if int(name_labels[1])==2: color = [ 0.1 , 0.9 , 0.1 ] 
           size = np.array(size, dtype=np.float32)
           color = np.array(color, dtype=np.float32)

       if self.dataset=="single-body_2d_3classes": 
           label = {0: int(name_labels[0]), 1: color, 2: size}
       elif "celeba-3classes" in self.dataset: 
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: int(name_labels[2])}

       return img, label 

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    #transform = transforms.Compose([transforms.Resize((54,54)), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = my_dataset(transform, dataset="single-body_2d_3classes", n_class_size=1, n_class_color=1, configs=["000","010","100","001"])
    dataloader = DataLoader(dataset, batch_size=4)

    for img, label in dataloader:
        print('label=',label)
        print(img.shape)
        plt.imshow(np.transpose(img[0].numpy(), (2,1,0)))
        plt.show()
        print('img.shape=',img.shape)
        exit()

