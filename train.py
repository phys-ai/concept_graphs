from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST, ImageFolder
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import load_dataset
import os
import glob
from PIL import Image
import json
import argparse
from einops import rearrange, repeat, reduce, pack, unpack
import math


parser = argparse.ArgumentParser()
parser.add_argument('--lrate', default=1e-4, type=float)
parser.add_argument('--test_size', default=1.4, type=float)
parser.add_argument('--alpha', default=1500, type=int)
parser.add_argument('--beta', default=2.0, type=float)
parser.add_argument('--num_samples', default=5000, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--n_T', default=500, type=int)
parser.add_argument('--n_feat', default=256, type=int)
parser.add_argument('--n_sample', default=64, type=int)
parser.add_argument('--n_epoch', default=100, type=int)
parser.add_argument('--experiment', default="H32-train1", type=str)
parser.add_argument('--remove_node', default="010", type=str)
parser.add_argument('--flag_concat', default=2, type=str) #, action=argparse.BooleanOptionalAction)
parser.add_argument('--flag_eval', default=1, type=int)
parser.add_argument('--flag_online', default=1, type=int)
parser.add_argument('--flag_zipf', default=2, type=int)
parser.add_argument('--flag_att', default=1, type=int)
parser.add_argument('--flag_double', default=1, type=int)
parser.add_argument('--pixel_size', default=28, type=int)
#parser.add_argument('--n_class_size', default=1, type=int)
#parser.add_argument('--n_class_color', default=1, type=int)
parser.add_argument('--save_model', default=1, type=int)
parser.add_argument('--dataset', default="single-body_2d_3classes", type=str)




class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in experiment channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


def l2norm(t):
    return F.normalize(t, dim = -1)

def exists(val):
    return val is not None

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class RearrangeToSequence(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = rearrange(x, 'b c ... -> b ... c')
        x, ps = pack([x], 'b * c')

        x = self.fn(x)

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b ... c -> b c ...')
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, fp16_eps = 1e-3, stable = False):
        super().__init__()
        self.eps = eps
        self.fp16_eps = fp16_eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = self.eps if x.dtype == torch.float32 else self.fp16_eps

        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        rotary_emb = None,
        cosine_sim = True,
        cosine_sim_scale = 16
    ):
        super().__init__()
        self.scale = cosine_sim_scale if cosine_sim else (dim_head ** -0.5)
        self.cosine_sim = cosine_sim

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # rotary embeddings
        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net
        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # whether to use cosine sim
        if self.cosine_sim:
            q, k = map(l2norm, (q, k))

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities
        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values
        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels, flag_att):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        attention = nn.Identity()
        if flag_att==0:  
            create_self_attn = lambda dim: RearrangeToSequence(Residual(Attention(dim)))
            attention = create_self_attn(out_channels)
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), attention, nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, flag_att):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        attention = nn.Identity()
        if flag_att==0:  
            create_self_attn = lambda dim: RearrangeToSequence(Residual(Attention(dim)))
            attention = create_self_attn(out_channels)
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            attention, 
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10, flag_concat=0, dataset="", flag_att=1):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_contexts = len(n_classes)
        self.n_feat = 2 * n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat, flag_att)
        self.down2 = UnetDown(n_feat, 2 * n_feat, flag_att)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)

        ### embedding shape
        self.dataset = dataset
        self.flag_concat = flag_concat
        if flag_concat==1: 
            n_out1 = int(2*n_feat/self.n_contexts)
            n_out2 = int(n_feat/self.n_contexts)
        else:
            n_out1 = 2*n_feat
            n_out2 = n_feat
        self.n_out1 = n_out1
        self.n_out2 = n_out2

        self.contextembed1 = []
        self.contextembed2 = []
        for iclass in range(len(self.n_classes)):
            self.contextembed1.append( EmbedFC(self.n_classes[iclass], n_out1).to("cuda:0") )
            self.contextembed2.append( EmbedFC(self.n_classes[iclass], n_out2).to("cuda:0") )


        n_conv = 7
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, n_conv, n_conv), 
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat, flag_att)
        self.up2 = UnetUp(2 * n_feat, n_feat, flag_att)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        temb1 = self.timeembed1(t).view(-1, int(self.n_feat), 1, 1)
        temb2 = self.timeembed2(t).view(-1, int(self.n_feat/2), 1, 1)

        # embed context, time step
        cemb1 = 0
        cemb2 = 0
        for ic in range(len(self.n_classes)):
            tmpc = c[ic]
            if tmpc.dtype==torch.int64: 
                tmpc = nn.functional.one_hot(tmpc, num_classes=self.n_classes[ic]).type(torch.float)
            cemb1 += self.contextembed1[ic](tmpc).view(-1, int(self.n_out1/1.), 1, 1)
            cemb2 += self.contextembed2[ic](tmpc).view(-1, int(self.n_out2/1.), 1, 1)


        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * torch.mean((input - target) ** 2, (1,2,3)))

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1, n_classes=None, flag_weight=0):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.n_classes = n_classes

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.flag_weight = flag_weight
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c[0])+self.drop_prob).to(self.device)
        
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, c_gen, size, device, guide_w = 0.0):

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
         
        _c_gen = [tmpc_gen[:n_sample].to(device) for tmpc_gen in c_gen.values()] 

        context_mask = torch.zeros_like(_c_gen[0]).to(device)

        x_i_store = [] 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.nn_model(x_i, _c_gen, t_is, context_mask)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i%20==0:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store


def train_mnist(args):

    n_epoch = args.n_epoch 
    batch_size = args.batch_size 
    n_T = args.n_T 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat = args.n_feat 
    lrate = args.lrate 
    alpha = args.alpha
    beta = args.beta
    test_size = args.test_size
    save_model = args.save_model 
    dataset = args.dataset 
    num_samples = args.num_samples 
    pixel_size = args.pixel_size
    if dataset=="celeba-2classes" or "fairface" in dataset:
        in_channels = 3
    else:
        in_channels = 4
    experiment = args.experiment 
    n_sample = args.n_sample 
    if n_sample > batch_size: 
        n_sample = batch_size
    flag_eval = args.flag_eval 
    flag_online = args.flag_online 
    flag_att = args.flag_att 
    flag_double = args.flag_double 
    flag_zipf = args.flag_zipf 
    remove_node = args.remove_node 
    #n_class_size = args.n_class_size
    #n_class_color = args.n_class_color

    with open("config_category.json", 'r') as f:
         configs = json.load(f)[experiment]

    print("### Number of samples", num_samples)
    if experiment=="H42-train1": 
        n_classes = [2,3,1,1]
    elif experiment=="H22-train1": 
        n_classes = [2,2]
    else:
        n_classes = [3,3,1]

    if dataset=="celeba-2classes":
       tf = transforms.Compose([
             transforms.Resize((pixel_size,pixel_size)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
           ])
    else: 
       tf = transforms.Compose([transforms.Resize((pixel_size,pixel_size)), transforms.ToTensor()])

    save_dir = './output/'+dataset+'/'+experiment+'/'
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    save_dir = save_dir + str(num_samples)+"_"+str(test_size)+"_"+str(n_feat)+"_"+str(n_T)+"_"+str(n_epoch)+"_"+str(lrate)+"_"+remove_node+"_"+str(alpha)+"_"+str(beta)+"_"+str(flag_att)+"_"+str(flag_double)+"_"+str(flag_zipf)+"_"+str(flag_online)+"/"
    if not os.path.isdir(save_dir): os.makedirs(save_dir)

    ddpm = DDPM(nn_model=ContextUnet(in_channels=in_channels, n_feat=n_feat, n_classes=n_classes, dataset=dataset, flag_att=flag_att), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1, n_classes=n_classes)
    ddpm.to(device)

    # optionally load a model
    if flag_eval==0: 
        ddpm.load_state_dict(torch.load(model_save_dir + f"model_"+str(int(n_epoch-1))+".pth", map_location=torch.device(device)))

    train_dataset = load_dataset.my_dataset(tf, num_samples, dataset, configs=configs["train"], training=True, alpha=alpha, remove_node=remove_node, flag_zipf=flag_zipf, flag_double=flag_double)
    train_dataloader = DataLoader([train_dataset[isample] for isample in range(2500)], batch_size=batch_size, shuffle=True, num_workers=1)
    train_dataloader_online = DataLoader([train_dataset[isample] for isample in range(2500,num_samples)], batch_size=batch_size, shuffle=True, num_workers=1)


    test_dataloaders = {}
    log_dict = {'train_loss_per_batch': [],
                'test_loss_per_batch': {key: [] for key in configs["test"]}}
    _configs = list(set(configs["test"] + configs["train"])) 
    for config in _configs: 
        test_dataset = load_dataset.my_dataset(tf, n_sample, dataset, configs=config, training=False, test_size=test_size) 
        test_dataloaders[config] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')

        if not flag_eval==0: 
            ddpm.train()

            # linear lrate decay
            optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

            loss_ema = None
            pbar = tqdm(train_dataloader)
            for x, c in pbar:
                optim.zero_grad()
                x = x.to(device)
                _c = [tmpc.to(device) for tmpc in c.values()]
                loss = ddpm(x, _c)
                log_dict['train_loss_per_batch'].append(loss.item())
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optim.step()
        
            if flag_online==0: 
                optim.zero_grad()
                x, c = next(iter(train_dataloader_online))
                x = x.to(device)
                _c = [tmpc.to(device) for tmpc in c.values()]
                loss = ddpm(x, _c)
                log_dict['train_loss_per_batch'].append(loss.item())
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                optim.step()


        ddpm.eval()
        with torch.no_grad():

            if not flag_eval==0: 
                for test_config in configs["test"]: 
                    for test_x, test_c in test_dataloaders[test_config]:
                        test_x = test_x.to(device)
                        _test_c = [tmptest_c.to(device) for tmptest_c in test_c.values()]
                        test_loss = ddpm(test_x, _test_c)
                        log_dict['test_loss_per_batch'][test_config].append(test_loss.item())

            if (save_model==0 and ep % 1==0) or (ep >= int(n_epoch-1)) or (ep % 100==99): 
                _configs = list(set(configs["test"] + configs["train"])) 
                for test_config in _configs: 
                    x_real, c_gen = next(iter(test_dataloaders[test_config]))
                    x_real = x_real[:n_sample].to(device)
                    x_gen, x_gen_store = ddpm.sample(n_sample, c_gen, (in_channels, pixel_size, pixel_size), device, guide_w=0.0)
                    np.savez_compressed(save_dir + f"image_"+test_config+"_ep"+str(ep)+".npz", x_gen=x_gen.detach().cpu().numpy()) 
                    print('saved image at ' + save_dir + f"image_"+test_config+"_ep"+str(ep)+".png")

                    if ep == int(n_epoch-1): 
                        np.savez_compressed(save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz", x_gen_store=x_gen_store)
                        print('saved image file at ' + save_dir + f"gen_store_"+test_config+"_ep"+str(ep)+".npz")


        if ep == int(n_epoch-1):
            with open(save_dir + f"training_log_"+str(ep)+".json", "w") as outfile:
                json.dump(log_dict, outfile)

            #if not flag_eval==0: 
            #    torch.save(ddpm.state_dict(), model_save_dir + f"model_{ep}.pth")
            #    print('saved model at ' + model_save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    args = parser.parse_args()

    train_mnist(args)

