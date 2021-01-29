from torch import nn, optim, as_tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.nn.init import *
from torchvision import transforms, utils, datasets, models
from models.inception_resnet_v1 import InceptionResnetV1

import cv2
from PIL import Image
from pdb import set_trace
import time
import copy
import datetime as dt

from pathlib import Path
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage import io, transform
from tqdm import trange, tqdm
import csv
import glob
import dlib

plt.ion()   # interactive mode

import pandas as pd
import numpy as np

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        
    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        return x

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        dtype = tensor.dtype
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor = tensor.mul(self.std[:, None, None]).add(self.mean[:, None, None])
#         for t, m, s in zip(tensor, self.mean, self.std):
#             t.mul_(s).add_(m)
#             # The normalize code -> t.sub_(m).div_(s)
        return tensor

class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if self.inplace:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor
        dtype = tensor.dtype
        self.mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        self.std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        tensor = tensor.sub(self.mean[:, None, None]).div(self.std[:, None, None])
        return tensor

def parse_csv_labels(fn, skip_header=True, cat_separator = ' '):
    df = pd.read_csv(fn, index_col=0, header=0 if skip_header else None, dtype=str)
    fnames = df.index.values
    df.iloc[:,0] = df.iloc[:,0].str.split(cat_separator)
    return fnames, list(df.to_dict().values())[0]

def dict_source(folder, fnames, csv_labels, suffix='', continuous=False):
    all_labels = sorted(list(set(p for o in csv_labels.values() for p in ([] if type(o) == float else o))))
    full_names = [os.path.join(folder,str(fn)+suffix) for fn in fnames]
    label2idx = {v:k for k,v in enumerate(all_labels)}
    label_arr = nhot_labels(label2idx, csv_labels, fnames, len(all_labels))
    is_single = np.all(label_arr.sum(axis=1)==1)
    if is_single: label_arr = np.argmax(label_arr, axis=1)
    return full_names, label_arr, all_labels

def csv_source(folder, csv_file, skip_header=True, suffix='', continuous=False, cat_separator=' '):
    fnames,csv_labels = parse_csv_labels(csv_file, skip_header, cat_separator)
    return dict_source(folder, fnames, csv_labels, suffix, continuous)

def nhot_labels(label2idx, csv_labels, fnames, c):
    all_idx = {k: n_hot([label2idx[o] for o in ([] if type(v) == float else v)], c)
               for k,v in csv_labels.items()}
    return np.stack([all_idx[o] for o in fnames])

def n_hot(ids, c):
    res = np.zeros((c,), dtype=np.float32)
    res[ids] = 1
    return res

def get_cv_idxs(n, cv_idx=0, val_pct=0.2, seed=42):
    np.random.seed(seed)
    n_val = int(val_pct*n)
    idx_start = cv_idx*n_val
    idxs = np.random.permutation(n)
    return idxs[idx_start:idx_start+n_val]

def split_by_idx(idxs, *a):
    mask = np.zeros(len(a[0]),dtype=bool)
    mask[np.array(idxs)] = True
    return [(o[mask],o[~mask]) for o in a]

USE_GPU = torch.cuda.is_available()
def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x

def is_half_tensor(v):
    return isinstance(v, torch.cuda.HalfTensor)

def T(a, half=False, cuda=True):
    """
    Convert numpy array into a pytorch tensor. 
    if Cuda is available and USE_GPU=True, store resulting tensor in GPU.
    """
    if not torch.is_tensor(a):
        a = np.array(np.ascontiguousarray(a))
        if a.dtype in (np.int8, np.int16, np.int32, np.int64):
            a = torch.LongTensor(a.astype(np.int64))
        elif a.dtype in (np.float32, np.float64):
            a = to_half(a) if half else torch.FloatTensor(a)
        else: raise NotImplementedError(a.dtype)
    if cuda: a = to_gpu(a)
    return a

def create_variable(x, volatile, requires_grad=True):
    if type (x) != torch.autograd.Variable:
        x = torch.autograd.Variable(T(x), requires_grad=requires_grad, volatile=volatile)
    return x

def V_(x, requires_grad=True, volatile=False):
    '''equivalent to create_variable, which creates a pytorch tensor'''
    return create_variable(x, volatile=volatile, requires_grad=requires_grad)
def V(x, requires_grad=True, volatile=False):
    '''creates a single or a list of pytorch tensors, depending on input x. '''
    return map_over(x, lambda o: V_(o, requires_grad, volatile))

def to_np(v):
    '''returns an np.array object given an input of np.array, list, tuple, torch variable or tensor.'''
    if isinstance(v, float): return np.array(v)
    if isinstance(v, (np.ndarray, np.generic)): return v
    if isinstance(v, (list,tuple)): return [to_np(o) for o in v]
    if isinstance(v, torch.autograd.Variable): v=v.data
    if torch.cuda.is_available():
        if is_half_tensor(v): v=v.float()
    if isinstance(v, torch.FloatTensor): v=v.float()
    return v.cpu().numpy()

def is_listy(x): return isinstance(x, (list,tuple))
def is_iter(x): return isinstance(x, collections.Iterable)
def map_over(x, f): return [f(o) for o in x] if is_listy(x) else f(x)

def create_noise(b): 
    return V(torch.zeros(b, nz, 1, 1).normal_(0, 1)).to(device)

def gallery(x, nc=3):
    n,h,w,c = x.shape
    nr = n//nc
    assert n == nr*nc
    return (x.reshape(nr, nc, h, w, c)
              .swapaxes(1,2)
              .reshape(h*nr, w*nc, c))

def set_trainable_attr(m,b):
    m.trainable=b
    for p in m.parameters(): p.requires_grad=b

def apply_leaf(m, f):
    c = list(m.children())
    if isinstance(m, nn.Module): f(m)
    if len(c)>0:
        for l in c: apply_leaf(l,f)

def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))

def show_landmarks_batch(sample_batched, mean, sd):
    """Show image for a batch of samples."""
    images_batch = (sample_batched * mean) + sd
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 3

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(i * im_size + (i + 1) * grid_border_size,
                    grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #print(outputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def init_GAN():
    """Function used during presentation to initialize GAN and classifier quickly."""
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 3, 3, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 4, 6, bias=False),
                nn.Tanh()
                # state size. (nc) x 160 x 160
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 160 x 160
                nn.Conv2d(nc, ndf, 4, 4, 6, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 3, 3, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)

    netD = None; netG = None

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    bs,nz = 64,100
    nc = 3; ndf = 160; ngf = 160
    ngpu = 1
    netD = Discriminator(ngpu).to(device)
    netG = Generator(ngpu).to(device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netD.apply(weights_init)
    netG.apply(weights_init)

    # FT

    class_names = ['Adrien_Brody','Alejandro_Toledo','Angelina_Jolie','Arnold_Schwarzenegger','Carlos_Moya','Charles_Moose','James_Blake','Jennifer_Lopez','Michael_Chaykowsky','Roh_Moo-hyun','Venus_Williams']

    model_ft = InceptionResnetV1(pretrained='vggface2', classify=False, num_classes=len(class_names))

    layer_list = list(model_ft.children())[-5:]

    model_ft = nn.Sequential(*list(model_ft.children())[:-5])

    class Flatten(nn.Module):
        def __init__(self):
            super(Flatten, self).__init__()
            
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return x

    class normalize(nn.Module):
        def __init__(self):
            super(normalize, self).__init__()
            
        def forward(self, x):
            x = F.normalize(x, p=2, dim=1)
            return x

    ## IF TRAINING JUST LAST LAYERS

    for param in model_ft.parameters():
        param.requires_grad = False
    
    model_ft.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
    model_ft.last_linear = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=1792, out_features=512, bias=False),
        normalize()
    )
    model_ft.logits = nn.Linear(layer_list[3].in_features, len(class_names))
    model_ft.softmax = nn.Softmax(dim=1)

    model_ft = model_ft.to(device)

    return netD, netG, model_ft