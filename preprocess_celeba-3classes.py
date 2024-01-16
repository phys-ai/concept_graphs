import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from PIL import Image
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import random
import json

attr_df = pd.read_csv("input/celeba-dataset/list_attr_celeba.csv")
bbox_df = pd.read_csv("input/celeba-dataset/list_bbox_celeba.csv")
bbox_df = pd.DataFrame(bbox_df).set_index('image_id') 
#print(pearsonr(attr_df['race'], attr_df['gender']))

#ntest_sample = 200
#ntrain_sample = 10000
ntest_sample = 100
ntrain_sample = 10000
out_dir = "working/celeba-3classes-smiling-"+str(ntrain_sample)+"_"+str(ntest_sample)+"/"
if not os.path.exists(out_dir): os.mkdir(out_dir)
if not os.path.exists(out_dir + "train/"): os.mkdir(out_dir + "train/")
if not os.path.exists(out_dir + "test/"): os.mkdir(out_dir + "test/")
attr_df = attr_df[["image_id","Male","Pale_Skin","Black_Hair","Blond_Hair","Brown_Hair","Young","Smiling"]] #"Young"

"""
pixel_size = 96
tf = transforms.Compose([
     transforms.Resize((pixel_size,pixel_size)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
for img_path in attr_df["image_id"].tolist()[:5]: 
    img = Image.open("input/celeba-dataset/img_align_celeba/img_align_celeba/"+img_path)
    tf_img = tf(img)
    arr_img = tf_img.detach().cpu().numpy()
    print(arr_img)
    plt.imshow(np.transpose(arr_img,(1,2,0)))
    plt.show()
"""

with open("config_category.json", 'r') as f:
     configs = json.load(f)
experiment = "H32-train1"
test_configs = configs[experiment]["test"]


counts = {}
for att0 in [1,-1]: 
    for att1 in [1,-1]: 
        for att2 in [1,-1]: 
            cond0 = attr_df["Male"]==att0
            #if "smiling" in out_dir: 
            cond1 = attr_df["Smiling"]==att1
            #else: 
            #    cond1 = attr_df["Young"]==att1
            if att2==1: cond2 = (attr_df["Black_Hair"]==1) #| (attr_df["Brown_Hair"]==1)
            else: cond2 = (attr_df["Blond_Hair"]==1)
            if att0==-1: str_att0 = "0"
            else: str_att0 = "1"
            if att1==1: str_att1 = "0"
            else: str_att1 = "1"
            if att2==1: str_att2 = "0"
            else: str_att2 = "1"
            label = str_att0 + str_att1 + str_att2
            image_files = attr_df[cond0&cond1&cond2]["image_id"]
            if label in test_configs: nmax = ntest_sample 
            else: nmax = ntrain_sample 
            print(label, nmax, len(image_files))
            for image_file in list(image_files)[:nmax]: 
                tmpbbox = bbox_df.loc[image_file]
                source_file = "input/celeba-dataset/img_align_celeba/img_align_celeba/" + image_file 
                destination_file = out_dir + "train/celeba_" + label + "_" + image_file
                img = Image.open(source_file)
                destination_file = out_dir + "train/celeba_" + label + "_" + image_file.split("/")[-1]
                shutil.copy2(source_file, destination_file)

            for image_file in list(image_files)[nmax:nmax+250]: 
                tmpbbox = bbox_df.loc[image_file]
                source_file = "input/celeba-dataset/img_align_celeba/img_align_celeba/" + image_file 
                destination_file = out_dir + "train/celeba_" + label + "_" + image_file
                img = Image.open(source_file)
                destination_file = out_dir + "test/celeba_" + label + "_" + image_file.split("/")[-1]
                shutil.copy2(source_file, destination_file)

