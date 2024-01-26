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

def hamming_distance(train_config):
   target = "000"
   distance = 0
   for ic in range(len(target)):
       distance += np.abs(int(train_config[ic])-int(target[ic]))
   return distance


class my_dataset(Dataset):
    def __init__(self, transform=None, num_samples=5000, dataset="", configs="", training=True, test_size=None, alpha=1.0, beta=2.0, remove_node=None, flag_zipf=2, flag_double=1):
        self.training = training
        self.test_size = test_size

        self.dataset = dataset
        #self.n_class_size = n_class_size
        #self.n_class_color = n_class_color

        if flag_zipf==0: 
            ratios = {0: beta**3, 1: beta**2, 2: beta}
            sum_ratio = ratios[0] + 3 * ratios[1] + 3 * ratios[2] 
        elif flag_zipf==1: 
            ratios = {0: beta**2, 1: beta, 2: beta**2}
            sum_ratio = ratios[0] + 3 * ratios[1] + 3 * ratios[2] 

        #if dataset=="celeba-2classes":
        if "celeba" in dataset: 
            prefix = "celeba"
            ext = ".jpg"
        elif dataset=="fairface-3classes":
            prefix = "fairface"
            ext = ".jpg"
        else:
            prefix = "CLEVR"
            ext = ".png"
        if training:
            self.train_image_paths = []
            for config in configs:
                if flag_zipf==2:
                    if config=="000" and alpha!=1500 and remove_node!="100": 
                        new_paths = glob.glob("input/"+dataset+"/train_"+remove_node+"/"+prefix+"_000_*"+ext)
                    else: 
                       new_paths = glob.glob("input/"+dataset+"/train/"+prefix+"_"+config+"_*"+ext)
                    if remove_node==config: 
                        new_paths = new_paths[:alpha]
                    if flag_double==0 and config=="010": 
                        new_paths = new_paths + new_paths
                else: 
                    distance = hamming_distance(config)
                    new_paths = glob.glob("input/"+dataset+"/train/"+prefix+"_"+config+"_*"+ext)
                    new_paths = new_paths[:int(num_samples*ratios[distance]/sum_ratio)]
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
       if not ("celeba" in self.dataset or self.dataset=="fairface-3classes" ):
           with open(img_path.replace(".png", ".json"), 'r') as f:
               my_dict = json.loads(f.read())
               if "single-body_2d_3classes" in self.dataset:
                   _size = my_dict[0]
                   _color = my_dict[1][:3]
               if "single-body_3d_3classes" in self.dataset:
                   _size = my_dict[0]
                   _color = my_dict[1][:3]
               if "single-body_2d_4classes" in self.dataset:
                   _size = my_dict[0]
                   _color = my_dict[1][:3]
                   _position = my_dict[-1]

           if self.training:
               size = _size
               if "single-body_2d_3classes" in self.dataset or "single-body_3d_3classes" in self.dataset:
                   color = _color
               if "single-body_2d_4classes" in self.dataset:
                   color = _color
                   position = _position
           else:
              if int(name_labels[2])==0: size = 2.6
              if int(name_labels[2])==1: size = self.test_size
              if "single-body_2d_3classes" in self.dataset or "single-body_3d_3classes" in self.dataset:
                  if int(name_labels[1])==0: color = [ 0.9 , 0.1 , 0.1 ] 
                  if int(name_labels[1])==1: color = [ 0.1 , 0.1 , 0.9 ] 
                  if int(name_labels[1])==2: color = [ 0.1 , 0.9 , 0.1 ] 
              if "single-body_2d_4classes" in self.dataset:
                  if int(name_labels[1])==0: color = [ 0.9 , 0.1 , 0.1 ] 
                  if int(name_labels[1])==1: color = [ 0.1 , 0.1 , 0.9 ] 
                  if int(name_labels[3])==0: position = 0.1
                  if int(name_labels[3])==1: position = 0.9

           size = np.array(size, dtype=np.float32)
           if "single-body_2d_3classes" in self.dataset or "single-body_3d_3classes" in self.dataset:
               color = np.array(color, dtype=np.float32)

       if "single-body_2d_3classes" in self.dataset or "single-body_3d_3classes" in self.dataset:
           label = {0: int(name_labels[0]), 1: color, 2: size}
       elif "single-body_2d_4classes" in self.dataset: 
           label = {0: int(name_labels[0]), 1: np.array(color, dtype=np.float32), 2: size, 3: np.array(position, dtype=np.float32)}
       elif "celeba-2classes" in self.dataset: 
           label = {0: int(name_labels[0]), 1: int(name_labels[1])}
       elif "celeba-3classes" in self.dataset: 
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: int(name_labels[2])}
       elif "fairface-3classes" in self.dataset: 
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: int(name_labels[2])}
       else:
           label = {0: int(name_labels[0]), 1: int(name_labels[1]), 2: size}

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

