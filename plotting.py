import matplotlib.pyplot as plt
import json 
import numpy as np
from PIL import Image
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision import models, transforms
from matplotlib import cm
import glob
import matplotlib
import linear_classifier_3classes
import mlp_classifier_4classes
import torch
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
import matplotlib.image as mpimg
from torchvision import models, transforms
import torch.nn as nn
import pandas as pd
from sklearn.metrics import log_loss
font = {'size': 15}
matplotlib.rc('font', **font)
cp = sns.color_palette("colorblind")
criterion = nn.CrossEntropyLoss()


pixel_size = 28
INPUT_DIM = pixel_size * pixel_size * 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_class_color = 2

scale_factor = 78
tf = transforms.Compose([transforms.Resize((pixel_size,pixel_size)), transforms.ToTensor()])



def hamming_distance(target, train_configs):
   distance = 999
   for train_config in train_configs: 
       tmpdist = 0
       for ic in range(len(target)): 
           tmpdist += np.abs(int(train_config[ic])-int(target[ic]))
       if tmpdist<distance: distance = tmpdist
   return distance

def rescaled_conv(arr, N):
    conv_result = np.convolve(arr, np.ones(N) / N, mode='full')
    conv_result[:N-1] = conv_result[:N-1] * N/np.arange(1,N)
    return conv_result[0: len(conv_result)-N+1]

def calc_loss(preds, obs, classifier, nclasses=3):
    y_pred = classifier(torch.from_numpy(preds).to(device))
    loss = []
    for ii in range(nclasses): 
        y_obs = torch.tensor([int(obs[ii])]).repeat(len(y_pred[ii]))
        tmploss = criterion(y_pred[ii],y_obs).detach().numpy()
        #top_pred = 1.0 / (1.0 + np.exp(-y_pred[ii].detach().numpy()))
        #tmploss = top_pred[:,int(obs[ii])]
        loss.append( tmploss ) 
    return np.array(loss)


def calc_acc(preds, obs, classifier, nclasses=3):
    y_pred = classifier(torch.from_numpy(preds).to(device))
    accs = []
    preds = []
    for ii in range(nclasses): 
        top_pred = y_pred[ii].argmax(1, keepdim=True).detach().numpy()
        acc = np.array(top_pred[:,0]==int(obs[ii]), dtype=np.int) 
        accs.append( acc ) 
        preds.append( top_pred[:,0] ) 
    return np.array(preds), np.array(accs)


def learning_dynamics(dataset, experiment, param, test_size="", prefix_dir="output/"):

    properties_json = "properties_"+dataset+".json"
    with open(properties_json, 'r') as f: 
        properties = json.load(f)

    classifier_linear = linear_classifier_3classes.MLP(INPUT_DIM, [2,n_class_color,2,2])
    classifier_linear.load_state_dict(torch.load("working/linear-classifier_"+dataset+"_multi-class.pt", map_location=torch.device(device)))
    

    in_dir = prefix_dir+dataset+"/"+experiment+"/"+param+"/"
    out_dir = prefix_dir+dataset+"/"+experiment+"/"+param+"_"
    
    eps = range(100) 
    accs = {test_config: [] for test_config in configs[experiment]["test"] + configs[experiment]["train"]}
    losses = {test_config: [] for test_config in configs[experiment]["test"] + configs[experiment]["train"]}
    preds = {test_config: [] for test_config in configs[experiment]["test"] + configs[experiment]["train"]}
    gen_plots = []

    for ep in eps: 
        x_real_plot = {}
        x_gen_plot = {}
        for itest_config, test_config in enumerate(configs[experiment]["test"] + configs[experiment]["train"]): 
            template_path = glob.glob('../image_generation/output/'+dataset+'/template/CLEVR_'+test_config+'_00*.png')
            x_real = Image.open(template_path[0])
            x_real = tf(x_real).detach().numpy()
            img_path = in_dir+"/image_"+test_config+"_ep"+str(ep)+".npz" 
            x_gen = np.load(img_path)["x_gen"]
            x_gen = np.clip(x_gen, 0, 1)
            pred, acc = calc_acc(x_gen, test_config, classifier_linear)
            loss = calc_loss(x_gen, test_config, classifier_linear)
            accs[test_config].append( acc )
            preds[test_config].append( pred )
            losses[test_config].append( loss )
            x_gen = np.transpose(x_gen,(0,2,3,1))
            x_gen_plot[test_config] = np.mean(x_gen, axis=0)
            x_real_plot[test_config] = np.transpose(x_real, (1,2,0))
        gen_plots.append(x_gen_plot)
    #print(preds["111"])

    sampled_eps = np.arage(0, 100, 10)
    fig, axes = plt.subplots(ncols=len(sampled_eps)+1, nrows=len(configs[experiment]["test"]), sharex=True,sharey=True,
                             figsize=(1.5*len(sampled_eps), 1.5*len(configs[experiment]["test"])), gridspec_kw = {'wspace':0., 'hspace':0.1})
    for itest_config, test_config in enumerate(configs[experiment]["test"]): 
        axes[itest_config,0].imshow(x_real_plot[test_config], vmin=0, vmax=1)
    for iep, ep in enumerate(sampled_eps): 
        for itest_config, test_config in enumerate(configs[experiment]["test"]): 
            axes[itest_config,iep+1].imshow(gen_plots[iep][test_config], vmin=0, vmax=1)
        for ax in fig.axes: 
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['bottom'].set_color('#dddddd')
            ax.spines['top'].set_color('#dddddd') 
            ax.spines['right'].set_color('#dddddd') 
            ax.spines['left'].set_color('#dddddd') 
    plt.savefig(in_dir+"gen_learning.png", bbox_inches='tight', pad_inches=0.03), plt.close()
    print(in_dir+"gen_learning.png")



    lss = ['-', '--', '-.', ':', (5, (10, 3)), '-', '--', '-']
    colors = {0: "#56C1FF", 1: "#FF95CA", 2: "#ED220D", 3: "#ff42a1"}
    option = "" 
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.spines["top"].set_linewidth(0)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(axis="both", which="both", bottom=True, top=False,
                   labelbottom=True, left=True, right=False,
                   labelleft=True,direction='out',length=5,width=1.0,pad=8,labelsize=15)
    for itest_config, test_config in enumerate(configs[experiment]["train"]+configs[experiment]["test"]): 
        distance = hamming_distance(test_config, configs[experiment]["train"])
        color = colors[distance]
        _accs = np.array(accs[test_config])
        mult = np.mean(_accs[:, 0, :] * _accs[:, 1, :] * _accs[:, 2, :], axis=1)
        index = test_config[0] + test_config[2] + test_config[1] + "     "
        x_values = np.arange(len(mult)) * scale_factor
        plt.plot(x_values[:-1], rescaled_conv(mult, N=10)[:-1]*100, c=color, lw=3, label=index, ls=lss[itest_config])
    ax.set_xscale('log')
    plt.xlabel("Optimization Steps", fontsize=20)
    plt.ylabel("Accuracy", fontsize=20)
    plt.tight_layout()
    plt.xlim(0,10**3.8)
    plt.legend(loc="lower right", frameon=False, labelspacing=0.2, borderpad=0.2, borderaxespad=0.3, handlelength=3, fontsize=15) #, bbox_to_anchor=(1.08, 0.))
    plt.ylim(-1,101)
    plt.savefig(in_dir+"Full_learning_dynamics_multi-class_"+option+".pdf"), plt.close()
    print(in_dir+"Full_learning_dynamics_multi-class_"+option+".pdf")


if __name__ == '__main__': 

    learning_dynamics("single-body_2d_3classes", "H32-train1", "5000_1.4_1_1_256_500_100_0.0001_010_1500_2.0_0_1_2_1_1")


