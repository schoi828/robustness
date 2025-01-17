# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import math
import random
import os
import itertools
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
import torch.nn.functional as F
import csv

#from wilds.datasets.fmow_dataset import FMoWDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    #robustness
    "DSPRITES",
    "SHAPES3D",
    "SMALLNORB",
    "CELEBA",
    "DEEPFASHION"
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 2            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def build_imgnet(self,img_size):

        try:
            scale_lb = self.hparams['scale_lower_bound']
        except:
            scale_lb = 0.08
        randomresizecrop = transforms.RandomResizedCrop(img_size, scale=(scale_lb,1))

        return transforms.Compose([
            randomresizecrop,
            #transforms.RandomHorizontalFlip(), #not mentioned in Appendix of the deepmind paper
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
            #transforms.ToTensor(),
        ])
    
    def build_augmix(self,img_size):
        try:
            severity=int(self.hparams['severity'])
            mixture_width=int(self.hparams['mixture_width'])
        except:
            severity=3
            mixture_width=3
        
        print('augmix', 'severity', severity, 'mixture_width', mixture_width)
        return transforms.AugMix(severity=severity,mixture_width=mixture_width) #PIL image recommended. For torch tensor, it should be of torch.uint8

    def build_randaug(self,img_size):
        try:
            num_ops = int(self.hparams['num_ops'])
            magnitude=int(self.hparams['magnitude'])
        except:
            num_ops=3
            magnitude=5

        print('randaug', 'num_ops', num_ops, 'magnitude', magnitude)
        return transforms.RandAugment(num_ops,magnitude) #PIL image recommended. For torch tensor, it should be of torch.uint8
    
    def build_autoaug(self,img_size):
        if img_size <128:
            policy = transforms.autoaugment.AutoAugmentPolicy.CIFAR10
        else:
            policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET

        return transforms.AutoAugment(policy=policy) #PIL image recommended. For torch tensor, it should be of torch.uint8

    #transforms.RandomGrayscale(),

    def get_transforms(self,mode='no_aug', gray=False):
        assert mode in ['no_aug','imgnet','augmix','randaug','autoaug'], 'incorrect preprocessing mode'

        #normalize to [-1,1] (according to the official code in tensorflow/jax/chex)
        
        mean = [0.5] if gray and not self.resize else [0.5]*3
        std = [0.5] if gray and not self.resize else [0.5]*3
        normalize = transforms.Normalize(mean=mean, std=std)
        transforms_list = [lambda x: x.permute(2,0,1)]

        if self.resize:
            if gray:
                transforms_list.append(lambda x: x.repeat(3, 1, 1))
            transforms_list.append(transforms.Resize(224))

        if mode != 'no_aug' and self.split == 'train':
            if mode != 'imgnet':
                transforms_list.append(lambda x : transforms.functional.convert_image_dtype(x,torch.uint8))
            transforms_list.append(getattr(self,f'build_{mode}')(self.input_shape[1]))
            if mode != 'imgnet':
                transforms_list.append(lambda x : transforms.functional.convert_image_dtype(x,torch.float32))
        
        transforms_list.append(normalize)
        preprocess = transforms.Compose(transforms_list)

        return preprocess

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']

class DSPRITES(MultipleDomainDataset):
    def __init__(self, root: str = '/data', 
                 dist_type: str = None, 
                 dataset_size: int = None, 
                 aug='no_aug', 
                 resize=False,  
                 algo: str = 'ERM', 
                 split: str = 'train', 
                 ratio: float = 0.01,
                 attributes = None,
                 hparams = None) -> None:

        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        """
        self.label_names = ['square', 'ellipse', 'heart']
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.input_shape = (3, 64, 64,)
        self.num_classes = 3
        self.ratio = ratio
        self.resize = resize
        self.split = split
        self.algo = algo
        self.hparams = hparams
        
        self.shapes     = ['square', 'ellipse', 'heart']
        self.obj_colors = ['red', 'yellow', 'blue']
        self.bg_colors  = ['orange', 'green', 'purple']
        self.scales     = ['small', 'middle', 'big']
        
        """
        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)
        """
        if split == 'test':
            self._imgs, self._labels = self.IID()
        else:
            if dist_type == 'UNIFORM':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.UNIFORM()
            elif dist_type == 'SC':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self.train_imgs, self.train_labels, self.val_imgs, self.val_labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)
            
            if split == 'train':
                self._imgs = self.train_imgs 
                self._labels = self.train_labels
            else:
                self._imgs = self.val_imgs
                self._labels = self.val_labels
                
        self.postprocess_labels()
        self.transform = self.get_transforms(aug)
        print(split,dist_type, 'imgs:',len(self._imgs),'labels',len(self._labels))

    def postprocess_labels(self):
        self._labels[self._labels == 'square'] = 0
        self._labels[self._labels == 'ellipse'] = 1
        self._labels[self._labels == 'heart'] = 2

    def __getitem__(self, index: int):
        
        if self.algo in ['BPA','PnD', 'OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        return self.transform(torch.Tensor(self._imgs[index])), torch.tensor(int(self._labels[index]))

    def __len__(self) -> int:

        return len(self._imgs)


    def UNIFORM(self):

        split     = self._dataset_size * 1 * 13
        val_split = 1 * 10  

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for shape in ['square', 'ellipse', 'heart']:
            for obj_color in  ['red', 'yellow', 'blue']:
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            train_output, val_output = output[:split], output[split:]
                        else:
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)

        return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

    def SC(self, train: bool = True, ratio: float = 0.01, attributes = 'obj_color'):
        """
        [shape, obj_color]
        (square, red)
        (ellipse, yellow)
        (heart, blue)
        """
        split     = self._dataset_size * 1 * 40
        val_split = 1 * 10

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == 'obj_color':
	
            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]

                        if len(train_output) == 0:
                            train_output, val_output = output[:split], output[split:]
                        else:
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0]))
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


    def LDD(self, train: bool = True, attributes = 'obj_color'):
        """
        [obj_color]
        MANY (square, red),    (ellipse, red),    (heart, red)
        MANY (square, yellow), (eliipse, yellow), (heart, yellow)
        ---------------------------------------------------------
        FEW  (square, purple), (ellipse, purple), (heart, purple)
        #     minor             minor              minor_minor
        """
        major_split = self._dataset_size * 3 * 6
        minor_split = self._dataset_size * 1 * 6
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'obj_color':

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)


    def UDS(self, train: bool = True, attributes = 'obj_color'):
        """
        [obj_color]
        red
        yellow
        THERE IS NO blue 
        """
        split = self._dataset_size * 1 * 20
        val_split = 1 * 5

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'obj_color':

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow']:
                    for bg_color in ['orange', 'green', 'purple']:
                        for scale in ['small', 'middle', 'big']:
                            out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                            shuffle_idx = list(range(out.shape[0])) 
                            random.shuffle(shuffle_idx)
                            out = out[shuffle_idx]
                            output = out[:split + val_split]
                            if len(train_output) == 0:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for bg_color in ['orange', 'green']:
                        for scale in ['small', 'middle', 'big']:
                            out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                            shuffle_idx = list(range(out.shape[0])) 
                            random.shuffle(shuffle_idx)
                            out = out[shuffle_idx]
                            output = out[:split + val_split]
                            if len(train_output) == 0:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for bg_color in ['orange', 'green', 'purple']:
                        for scale in ['small', 'middle']:
                            out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                            shuffle_idx = list(range(out.shape[0])) 
                            random.shuffle(shuffle_idx)
                            out = out[shuffle_idx]
                            output = out[:split + val_split]
                            
                            if len(train_output) == 0:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)


    def SC_LDD(self, train: bool = True, ratio = 0.01, attributes = ['obj_color','bg_color']):
        """
        SC_LDD
        MANY: (square, red, orange), (ellipse, yellow, orange), (heart, blue, orange)
        MANY: (square, red, green),  (ellipse, yellow, green),  (heart, blue, green)
        #   :  major                  major                      major
        -----------------------------------------------------------------------------
        FEW : (square, red, purple), (ellipse, yellow, purple), (heart, blue, purple)
        #   :  minor                  minor_minor                minor_minor
        """
        major_split = self._dataset_size * 3 * 18
        minor_split = self._dataset_size * 1 * 18
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 5
        val_minor_split = 1 * 5
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'orange', scale))

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'green', scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 'purple', scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'red', bg_color, scale))

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'yellow', bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 'blue', bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for obj_color in ['red', 'yellow', 'blue']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['small', 'middle', 'big']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'red', bg_color, scale))

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'yellow', bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 'blue', bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow', 'blue']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'orange', scale))

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow', 'blue']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'green', scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow', 'blue']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 'purple', scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

       
    def SC_UDS(self, train: bool = True, ratio = 0.01, attributes = ['obj_color', 'bg_color']):
        """
        SC_UDS
        (square, red, orange), (ellipse, yellow, orange), (heart, blue, orange)
        (square, red, green),  (ellipse, yellow, green),  (heart, blue, green)
        ----------------------------------------------------------------------------------
        THERE IS NO (square, red, purple), (ellipse, yellow, purple), (heart, blue, purple)
        """
        split = self._dataset_size * 1 * 60
        val_split = 1 * 15

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle', 'big']:
                    for bg_color in ['orange', 'green']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, ['orange', 'green'], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle']:
                    for bg_color in ['orange', 'green', 'purple']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


        elif attributes == ['bg_color', 'obj_color']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle', 'big']:
                    for obj_color in ['red', 'yellow']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, ['red', 'yellow'], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


        elif attributes == ['bg_color', 'scale']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


        elif attributes == ['scale', 'obj_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow']:
                    for bg_color in ['orange', 'green', 'purple']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, ['red', 'yellow'], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


        elif attributes == ['scale', 'bg_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow', 'blue']:
                    for bg_color in ['orange', 'green']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, ['orange', 'green'], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)


    def LDD_UDS(self, train: bool = True, attributes = ['obj_color', 'bg_color']):
        """
        LDD_UDS
        LDD
        +
        THERE IS NO (purple)
        """
        major_split = self._dataset_size * 3 * 9
        minor_split = self._dataset_size * 1 * 9
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == ['obj_color', 'bg_color']:

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color']:

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow']:
                    for scale in ['small', 'middle', 'big']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale']:

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow', 'blue']:
                    for scale in ['small', 'middle']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color']:

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green', 'purple']:
                    for obj_color in ['red', 'yellow']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color']:

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        if len(train_output) == 0:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in ['square', 'ellipse', 'heart']:
                for bg_color in ['orange', 'green']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:major_split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip(['square', 'ellipse', 'heart'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green']:
                    for obj_color in ['red', 'yellow', 'blue']:
                        out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                        shuffle_idx = list(range(out.shape[0])) 
                        random.shuffle(shuffle_idx)
                        out = out[shuffle_idx]
                        output = out[:split + val_split]
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            return train_output, np.reshape(train_label, -1), val_output, np.reshape(val_label, -1)


    def SC_LDD_UDS(self, train: bool = True, ratio = 0.01, attributes = ['obj_color', 'bg_color', 'scale']):
        """
        SC_LDD_UDS
        SC_LDD
        +
        THERE IS NO (big)
        """
        major_split = self._dataset_size * 3 * 27
        minor_split = self._dataset_size * 1 * 27
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 6
        val_minor_split = 1 * 6
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color', 'scale']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'orange', scale))

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'green', scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 'purple', scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale', 'bg_color']:

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, obj_color in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue']):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['red', 'yellow', 'blue'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, ['orange', 'green'], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color', 'scale']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'red', bg_color, scale))

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'yellow', bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['small', 'middle']:
                    out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 'blue', bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale', 'obj_color']:

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, bg_color in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple']):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip(['square', 'ellipse', 'heart'], ['orange', 'green', 'purple'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, ['orange', 'green'], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color', 'bg_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_red_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'red', bg_color, scale))

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_yellow_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 'yellow', bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in ['orange', 'green']:
                    out = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_blue_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 'blue', bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, ['orange', 'green'], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color', 'obj_color']:

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    if len(train_output) == 0:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_orange_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'orange', scale))

            for shape, scale in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big']):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:major_split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_green_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 'green', scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip(['square', 'ellipse', 'heart'], ['small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in ['red', 'yellow']:
                    out = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")
                    shuffle_idx = list(range(out.shape[0])) 
                    random.shuffle(shuffle_idx)
                    out = out[shuffle_idx]
                    output = out[:split + val_split]
                    #output = np.load(f"{self._root}/dsprites/{shape}_{obj_color}_purple_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 'purple', scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, ['red', 'yellow'], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/dsprites/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            return train_output, train_label, val_output, np.reshape(val_label, -1)
	

    def IID(self):
        """
        UNIFORM test data
        """
        output = np.load(f"{self._root}/dsprites/iid_test.npy")
        label = np.load(f"{self._root}/dsprites/label_test.npy")

        return output, label



class SHAPES3D(MultipleDomainDataset):
 
    def __init__(self, root: str = '/data', 
                dist_type: str = None, 
                dataset_size: int = None, 
                aug='no_aug', 
                resize=False,  
                algo: str = 'ERM', 
                split: str = 'train', 
                ratio: float = 0.01,
                attributes = None,
                hparams = None) -> None:

        
        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        
        shapes = 0, 1, 2, 3
        obj_color = 0, 0.1, 0.2, 0.3
        bg_color = 0, 0.1, 0.2, 0.3
        scale = 'tiny', 'small', 'middle', 'big'
        """

        self.label_names = ['cube','cylinder','sphere','capsule']#['0', '1', '2', '3'] #shapes
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.input_shape = (3, 64, 64,)
        self.num_classes = 4
        self.ratio = ratio
        self.resize = resize
        self.split = split
        self.algo = algo
        self.hparams = hparams

        self.shapes = [0, 1, 2, 3]
        self.obj_colors = [0, 0.1, 0.2, 0.3]
        self.bg_colors = [0, 0.1, 0.2, 0.3]
        self.scales = ['tiny', 'small', 'middle', 'big']

        if split == 'train':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = True)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)

        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)

        elif split == 'test':
            self._imgs, self._labels = self.IID()

        self.transform = self.get_transforms(aug)
    
    def __getitem__(self, index: int):
        
        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)

    
    def UNIFORM(self, train: bool = True):

        split     = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
	
        for shape in [0, 1, 2, 3]:
            for obj_color in [0, 0.1, 0.2, 0.3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)

    def SC(self, train: bool = True, ratio: float = 0.01, attributes = 'obj_color'):

        split     = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []
        
        if attributes == 'obj_color':
	
            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD(self, train: bool = True, attributes = 'obj_color'):

        major_split = self._dataset_size * 3
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'obj_color':

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def UDS(self, train: bool = True, attributes = 'obj_color'):

        split = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'obj_color':

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2]:
                    for bg_color in [0, 0.1, 0.2, 0.3]:
                        for scale in ['tiny', 'small', 'middle', 'big']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'bg_color':

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for bg_color in [0, 0.1, 0.2]:
                        for scale in ['tiny', 'small', 'middle', 'big']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'scale':

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for bg_color in [0, 0.1, 0.2, 0.3]:
                        for scale in ['tiny', 'small', 'middle']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([shape] * split)
                            val_label.append([shape] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def SC_LDD(self, train: bool = True, ratio = 0.01, attributes = ['obj_color', 'bg_color']):

        major_split = self._dataset_size * 3
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0, scale))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.1, scale))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.2, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 0.3, scale))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'tiny'))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0, bg_color, scale))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.1, bg_color, scale))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.2, bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 0.3, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'tiny'))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))
            
            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0, bg_color, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.1, bg_color, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.2, bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big'], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 0.3, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.1, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.2, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big'], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 0.3, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            

    def SC_UDS(self, train: bool = True, ratio = 0.01, attributes = ['obj_color', 'bg_color']):

        split = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.shapes, self.obj_colors, [0, 0.1, 0.2], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
                    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    for bg_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['tiny', 'small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['bg_color', 'obj_color']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle', 'big']:
                    for obj_color in [0, 0.1, 0.2]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, [0, 0.1, 0.2], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['bg_color', 'scale']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['tiny', 'small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['scale', 'obj_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2]:
                    for bg_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, [0, 0.1, 0.2], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['scale', 'bg_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for bg_color in [0, 0.1, 0.2]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_{scale}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([shape] * split)
                        val_label.append([shape] * val_split)
                        generated_combinations.append((shape, obj_color, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, [0, 0.1, 0.2], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD_UDS(self, train: bool = True, attributes = ['obj_color', 'bg_color']):

        major_split = self._dataset_size * 3
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
                
        if attributes == ['obj_color', 'bg_color']:

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale']:

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color']:

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2]:
                    for scale in ['tiny', 'small', 'middle', 'big']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale']:

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2, 0.3]:
                    for scale in ['tiny', 'small', 'middle']:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color']:

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2, 0.3]:
                    for obj_color in [0, 0.1, 0.2]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color']:

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            for shape in [0, 1, 2, 3]:
                for bg_color in [0, 0.1, 0.2]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([shape] * major_split)
                        val_label.append([shape] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, split, val_split in zip([0, 1, 2, 3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2]:
                    for obj_color in [0, 0.1, 0.2, 0.3]:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                        val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_LDD_UDS(self, train: bool = True, ratio:float = 0.01, attributes = ['obj_color', 'bg_color', 'scale']):

        major_split = self._dataset_size * 3
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['obj_color', 'bg_color', 'scale']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0, scale))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.1, scale))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.2, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obh_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 0.3, scale))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['tiny', 'small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['obj_color', 'scale', 'bg_color']:

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'tiny'))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, obj_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, obj_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, [0, 0.1, 0.2], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'obj_color', 'scale']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0, bg_color, scale))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.1, bg_color, scale))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.2, bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for scale in ['tiny', 'small', 'middle']:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 0.3, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, self.bg_colors, ['tiny', 'small', 'middle']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['bg_color', 'scale', 'obj_color']:

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_tiny.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'tiny'))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_small.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'small'))

            for shape, bg_color in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3]):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_middle.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, bg_color, 'middle'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, bg_color, split, val_split in zip([0, 1, 2, 3], [0, 0.1, 0.2, 0.3], [minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_{bg_color}_big.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, bg_color, 'big'))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, [0, 0.1, 0.2], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'obj_color', 'bg_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_0_{bg_color}_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0, bg_color, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.1_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.1, bg_color, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.2_{bg_color}_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, 0.2, bg_color, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for bg_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_0.3_{bg_color}_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, 0.3, bg_color, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, self.obj_colors, [0, 0.1, 0.2], self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['scale', 'bg_color', 'obj_color']:

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0_{scale}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.1_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.1, scale))

            for shape, scale in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big']):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.2_{scale}.npy")[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([shape] * major_split)
                    val_label.append([shape] * val_split)
                    generated_combinations.append((shape, obj_color, 0.2, scale))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for shape, scale, split, val_split in zip([0, 1, 2, 3], ['tiny', 'small', 'middle', 'big'], [minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_minor_split]):
                for obj_color in [0, 0.1, 0.2]:
                    output = np.load(f"{self._root}/shapes3d/{shape}_{obj_color}_0.3_{scale}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [shape] * split], axis = 0)
                    val_label = np.concatenate([val_label, [shape] * val_split], axis = 0)
                    generated_combinations.append((shape, obj_color, 0.3, scale))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self.shapes, [0, 0.1, 0.2], self.bg_colors, self.scales))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            print(selected_combination)
            train_label = np.reshape(train_label, -1)
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/shapes3d/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy")[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def IID(self):
        """
        UNIFORM test data
        """
        output = np.load(f"{self._root}/shapes3d/iid_test.npy")
        label = np.load(f"{self._root}/shapes3d/label_test.npy")

        return output, label


class SMALLNORB(MultipleDomainDataset):
    def __init__(self, root: str = '/data', 
                 dist_type: str = None, 
                 dataset_size: int = None, 
                 aug: str = 'no_aug', 
                 resize: bool = False, 
                 algo: str = 'ERM', 
                 split: str = 'train', 
                 ratio: float = 0.01,
                 attributes = None,
                 hparams = None) -> None:
        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        """
        self.label_names = ['animal', 'human', 'airplane', 'truck', 'car']
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.resize: bool = resize
        self.split = split
        self.ratio = ratio
        self.input_shape = (3, 96, 96,) if algo == 'L2D' else (1, 96, 96,) 
        self.num_classes = 5
        self.algo = algo
        self.hparams = hparams

        self.categories = ['animal', 'human', 'airplane', 'truck', 'car']
        self.elevations = [0, 2, 4, 6, 8]
        self.azimuths   = [0, 8, 16, 24, 32]
        self.lightings  = [0, 1, 2, 3, 4]

        if split == 'train':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = True)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)

        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)

        elif split == 'test':
            self._imgs, self._labels = self.IID()
        
        imgs = []
        for i in self._imgs:
            imgs.append(i)
        self._imgs = np.array(imgs)
        self._labels[self._labels == 'animal'] = 0
        self._labels[self._labels == 'human'] = 1
        self._labels[self._labels == 'airplane'] = 2
        self._labels[self._labels == 'truck'] = 3
        self._labels[self._labels == 'car'] = 4
        self.transform = self.get_transforms(aug, gray=True)


    def __getitem__(self, index: int):
        img = torch.Tensor(self._imgs[index])
        if len(img.shape) < 3:
            img = img.unsqueeze(-1)
        if self.algo in ['BPA','PnD', 'OccamNets'] and self.split == 'train':
            return self.transform(img), int(self._labels[index]), index
        return self.transform(img), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)

    def UNIFORM(self, train: bool = True):

        split     = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for category in ['animal', 'human', 'airplane', 'truck', 'car']:
            for azimuth in [0, 8, 16, 24, 32]:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)


    def SC(self, train: bool = True, ratio: float = 0.01, attributes = 'azimuth'):
        """
        category_elevation_azimuth_lighting
        """

        split     = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == 'azimuth':

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'elevation':

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'lighting':

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
        

    def LDD(self, train: bool = True, attributes = 'azimith'):

        major_split = self._dataset_size * 2
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 2
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'azimuth':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == 'elevation':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == 'lighting':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def UDS(self, train: bool = True, attributes = 'azimuth'):

        split = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'azimuth':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24]:
                    for lighting in [0, 1, 2, 3, 4]:
                        for elevation in [0, 2, 4, 6, 8]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([category] * split)
                            val_label.append([category] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == 'elevation':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for lighting in [0, 1, 2, 3, 4]:
                        for elevation in [0, 2, 4, 6]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([category] * split)
                            val_label.append([category] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == 'lighting':

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for lighting in [0, 1, 2, 3]:
                        for elevation in [0, 2, 4, 6, 8]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([category] * split)
                            val_label.append([category] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def SC_LDD(self, train: bool = True, ratio: float = 0.01, attributes = ['azimuth', 'elevation']):

        major_split = self._dataset_size * 2
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 2
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['azimuth', 'elevation']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 0, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 2, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 4, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 6, azimuth, lighting))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, azimuth, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, 8, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['azimuth', 'lighting']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 0))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 1))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 2))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 3))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, azimuth, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, azimuth, 4))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'azimuth']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 0, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 8, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 16, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 24, lighting))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, elevation, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, 32, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'lighting']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24, 32]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 0))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 2))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 2))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 3))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, elevation, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, azimuth, 4))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'azimuth']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 0, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 8, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 16, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 24, lighting))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, lighting, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for elevation in [0, 2, 4, 6, 8]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, 32, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'elevation']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 0, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 2, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 4, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 6, azimuth, lighting))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, lighting, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24, 32]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, 8, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['azimuth', 'elevation']):

        split = self._dataset_size * 1
        val_split = 1

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['azimuth', 'elevation']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, [0, 2, 4, 6], self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['azimuth', 'lighting']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, [0, 1, 2, 3]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'azimuth']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, [0, 8, 16, 24], self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'lighting']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, [0, 1, 2, 3]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'azimuth']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, [0, 8, 16, 24], self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'elevation']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([category] * split)
                        val_label.append([category] * val_split)
                        generated_combinations.append((category, elevation, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, [0, 2, 4, 6], self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
                

    def LDD_UDS(self, train: bool = True, attributes = ['azimuth', 'elevation']):

        major_split = self._dataset_size * 2
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 2
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == ['azimuth', 'elevation']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['azimuth', 'lighting']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'azimuth']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3, 4]:
                    for azimuth in [0, 8, 16, 24]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'lighting']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for azimuth in [0, 8, 16, 24, 32]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'azimuth']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24]:
                    for elevation in [0, 2, 4, 6, 8]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'elevation']:

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            for category in ['animal', 'human', 'airplane', 'truck', 'car']:
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([category] * major_split)
                        val_label.append([category] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24, 32]:
                    for elevation in [0, 2, 4, 6]:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [category] * split], axis = 0)
                        val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
            
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        
    def SC_LDD_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['azimuth', 'elevation', 'lighting']):

        major_split = self._dataset_size * 2
        minor_split = self._dataset_size * 1
        minor_minor_split = self._dataset_size * 0
        val_split = 2
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['azimuth', 'elevation', 'lighting']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 0, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 2, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 4, azimuth, lighting))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 6, azimuth, lighting))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, azimuth, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, 8, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, [0, 1, 2, 3]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['azimuth', 'lighting', 'elevation']:

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 0))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 1))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 2))

            for category, azimuth in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 3))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, azimuth, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 8, 16, 24, 32], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, azimuth, 4))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, [0, 2, 4, 6], self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'azimuth', 'lighting']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 0, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 8, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 16, lighting))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 24, lighting))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, elevation, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for lighting in [0, 1, 2, 3]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, 32, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, self.azimuths, [0, 1, 2, 3]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['elevation', 'lighting', 'azimuth']:

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_0.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 0))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_1.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 1))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_2.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 2))

            for category, elevation in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_3.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, azimuth, 3))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, elevation, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 2, 4, 6, 8], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_{azimuth}_4.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, azimuth, 4))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, [0, 8, 16, 24], self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'azimuth', 'elevation']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_0_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 0, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_8_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 8, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_16_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 16, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_24_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, elevation, 24, lighting))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, lighting, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for elevation in [0, 2, 4, 6]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_{elevation}_32_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, elevation, 32, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, [0, 2, 4, 6], self.azimuths, self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['lighting', 'elevation', 'azimuth']:

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_0_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 0, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_2_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 2, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_4_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 4, azimuth, lighting))

            for category, lighting in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_6_{azimuth}_{lighting}.npy", allow_pickle = True)[:major_split + val_split]
                    train_output = np.append(train_output, output[:major_split], axis = 0)
                    val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([category] * major_split)
                    val_label.append([category] * val_split)
                    generated_combinations.append((category, 6, azimuth, lighting))

            train_label = np.reshape(train_label, -1)
            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for category, lighting, split, val_split in zip(['animal', 'human', 'airplane', 'truck', 'car'], [0, 1, 2, 3, 4], [minor_split, minor_split, minor_split, minor_split, minor_minor_split], [val_minor_split, val_minor_split, val_minor_split, val_minor_split, val_minor_minor_split]):
                for azimuth in [0, 8, 16, 24]:
                    output = np.load(f"{self._root}/smallnorb/smallnorb_split/{category}_8_{azimuth}_{lighting}.npy", allow_pickle = True)[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [category] * split], axis = 0)
                    val_label = np.concatenate([val_label, [category] * val_split], axis = 0)
                    generated_combinations.append((category, 8, azimuth, lighting))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.categories, self.elevations, [0, 8, 16, 24], self.lightings))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.load(f"{self._root}/smallnorb/smallnorb_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0].reshape(1,96,96), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def IID(self):
        """
        UNIFORM test data
        """
        output = np.load(f"{self._root}/smallnorb/smallnorb_split/iid_test.npy", allow_pickle = True)
        label = np.load(f"{self._root}/smallnorb/smallnorb_split/label_test.npy", allow_pickle = True)

        return output, label
    
# class CELEBA_C(MultipleDomainDataset):
    
#     def __init__(self, root: str = '/data', 
#                  dist_type: str = None, 
#                  dataset_size: int = None, 
#                  aug: str = 'no_aug', 
#                  resize: bool = False, 
#                  algo: str = 'ERM', 
#                  split: str = 'train', 
#                  ratio: float = 0.01,
#                  attributes = None,
#                  return_attr: bool = False,
#                  hparams = None) -> None:
#         """
#         dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
#         dataset_size: 1 for MAIN EXPERIMENTS
#         split: train, val, test
#         """
#         self.label_names = ['Female', 'Male']
#         self._root: str  = '/data'#root
#         self._dataset_size: int = dataset_size
#         self.resize = resize
#         self.split = split
#         self.input_shape = (3, 256, 256,) 
#         self.num_classes = 2
#         self.ratio = ratio
#         self.algo = algo
#         self.hparams = hparams
#         self.return_attr = return_attr

#         self.male = ['-1', '1']
#         self.impulse = ['-1', '1']
#         self.snow = ['-1', '1']
#         self.elastic = ['-1', '1']

#         if split == 'train':
#             self._imgs, self._labels = self.CORRUPT(train = True, attribute=attributes)

#         if split == 'val':
#                 self._imgs, self._labels = self.CORRUPT(train = False, attribute=attributes)

#         elif split == 'test':
#             self._imgs, self._labels = self.IID()
            
#         if not self.return_attr:
#             self._labels[self._labels == -1] = 0
#             self._labels[self._labels == '-1'] = 0

#         #self._labels[self._labels == '1'] = 1
#         self.transform = self.get_transforms(aug)

#     def CORRUPT(self, train:bool = True, attribute=None):
#         split     = self._dataset_size * 113
#         val_split = 16
#         train_output = np.array([])
#         val_output   = np.array([])
#         train_label = []
#         val_label   = []
#         for male in ['-1', '1']:
#             if len(train_output) == 0:
#                 output = np.load(f'{self._root}/celeba_c_split/{male}_{attribute}')[:split + val_split]
#                 train_output, val_output = output[:split], output[split:]
#             else:
#                 output = np.load(f'{self._root}/celeba_c_split/{male}_{attribute}')[:split + val_split]
#                 train_output = np.append(train_output, output[:split], axis = 0)
#                 val_output = np.append(val_output, output[split:], axis = 0)
            
#             train_label.append([male] * split)
#             val_label.append([male] * val_split)
        
#         if train:
#             return train_output, np.reshape(train_label, -1)#train_label
#         else:
#             return val_output, np.reshape(val_label, -1)#val_label
        
#     def IID(self):
#         """
#         UNIFORM test data
#         """
#         output = np.load(f"{self._root}/celeba_c_split/iid_test.npy", allow_pickle = True)
#         label = np.load(f"{self._root}/celeba_c_split/label_test.npy", allow_pickle = True)

#         return output, label

#     def __len__(self) -> int:

#         return len(self._imgs)
    
#     def __getitem__(self, index: int):

#         if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
#             return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        
#         if self.return_attr:
#             return self.transform(torch.Tensor(self._imgs[index])), self._labels[index]
#         else:
#             return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])



class CELEBA_CLUSTER(MultipleDomainDataset):
    def __init__(self, root: str = '/data', 
                 dist_type: str = None, 
                 dataset_size: int = 1, 
                 aug: str = 'no_aug', 
                 resize: bool = True, 
                 algo: str = 'ERM', 
                 split: str = 'train', 
                 ratio: float = 0.01,
                 attributes = None,
                 return_attr: bool = False,
                 hparams = None) -> None:
        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        """
        self.label_names = ['Female', 'Male']
        self._root: str  = '/data'
        self._dataset_size: int = dataset_size
        self.resize = True
        self.split = split
        self.input_shape = (3, 224, 224,) 
        self.num_classes = 2
        self.ratio = ratio
        self.algo = algo
        self.hparams = hparams
        
        if split == 'train':
            self._imgs, self._labels = self.CLUSTER(True, attributes)    
        if split == 'val':
            self._imgs, self._labels = self.CLUSTER(False, attributes)
        elif split == 'test':
            self._imgs, self._labels = self.CLUSTER_TEST()
            
        #self._labels[self._labels == '1'] = 1
        self.transform = self.get_transforms(aug)

    def CLUSTER(self, train: bool = True, attribute=0):

        split     = self._dataset_size * 113
        val_split = 16

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for male in ['0', '1']:
            output = np.load(f"{self._root}/celeba_cluster/{male}_{attribute}.npy")
            if len(train_output) == 0:
                train_output, val_output = output[:split], output[split:split+val_split]
            else:
                train_output = np.append(train_output, output[:split], axis = 0)
                val_output = np.append(val_output, output[split:split+val_split], axis = 0)
            train_label.append([male] * split)
            val_label.append([male] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)


    def CLUSTER_TEST(self):

        split     = self._dataset_size * 400

        train_output = np.array([])
        train_label = []

        for male in ['0', '1']:
            output = np.load(f"{self._root}/celeba_cluster/{male}_test.npy")
            if len(train_output) == 0:
                train_output = output[:split]
            else:
                train_output = np.append(train_output, output[:split], axis = 0)
            train_label.append([male] * split)

        return train_output, np.reshape(train_label, -1)
    

    def __getitem__(self, index: int):

        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        
        return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)


class CELEBA_C(MultipleDomainDataset):
    def __init__(self, root: str = '/data', 
                 dist_type: str = None, 
                 dataset_size: int = None, 
                 aug: str = 'no_aug', 
                 resize: bool = False, 
                 algo: str = 'ERM', 
                 split: str = 'train', 
                 ratio: float = 0.01,
                 attributes = None,
                 return_attr: bool = False,
                 hparams = None) -> None:

        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        """
        self.label_names = ['Female', 'Male']
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.resize = resize
        self.split = split
        self.input_shape = (3, 256, 256,) 
        self.num_classes = 2
        self.ratio = ratio
        self.algo = algo
        self.hparams = hparams
        self.return_attr = return_attr

        self.male = ['-1', '1']
        self.black_hair = ['-1', '1']
        self.smiling = ['-1', '1']
        self.straight_hair = ['-1', '1']
        self.impulse = ['-1', '1']
        self.snow = ['-1', '1']
        self.elastic = ['-1', '1']

        if split == 'train':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = True)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)

        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)

        elif split == 'test':
            self._imgs, self._labels = self.IID()
            
        if not self.return_attr:
            self._labels[self._labels == -1] = 0
            self._labels[self._labels == '-1'] = 0

        #self._labels[self._labels == '1'] = 1
        self.transform = self.get_transforms(aug)

    def __getitem__(self, index: int):

        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        
        if self.return_attr:
            return self.transform(torch.Tensor(self._imgs[index])), self._labels[index]
        else:
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)

    def UNIFORM(self, train: bool = True):

        split     = self._dataset_size * 1 * 7
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for male in ['-1', '1']:
            for black_hair in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)


    def SC(self, train: bool = True, ratio: float = 0.01, attributes: str = 'black_hair'):
        """
        1. Male
        2. Black_Hair
        3. Smiling
        4. Straight_Hair
        """
        split     = self._dataset_size * 1 * 14 
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == 'black_hair':

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'straight_hair':

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD(self, train: bool = True, attributes: str = 'black_hair'):
        major_split = self._dataset_size * 3 * 4 
        minor_split = self._dataset_size * 1 * 4 
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'black_hair':

            for male in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'straight_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def UDS(self, train: bool = True, attributes: str = 'black_hair'):
        split = self._dataset_size * 1 * 14 
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'black_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        for straight_hair in ['-1', '1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        for straight_hair in ['-1', '1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == 'straight_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        for straight_hair in ['-1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_LDD(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling']):
        major_split = self._dataset_size * 3 * 8 
        minor_split = self._dataset_size * 1 * 8 
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['black_hair', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['smiling', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            
        if attributes == ['smiling', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['straight_hair', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['straight_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling']):
        split = self._dataset_size * 1 * 28 
        val_split = 1 * 8

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            #selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['black_hair', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD_UDS(self, train: bool = True, attributes = ['black_hair', 'smiling']):
        """
        LDD_UDS
        LDD
        +
        THERE IS NO (purple)
        """
        major_split = self._dataset_size * 3 * 8 
        minor_split = self._dataset_size * 1 * 8 
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == ['black_hair', 'smiling']:

            for male in ['-1', '1']:
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['black_hair', 'straight_hair']:

            for male in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling']:

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def SC_LDD_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling', 'straight_hair']):
        major_split = self._dataset_size * 3 * 16 
        minor_split = self._dataset_size * 1 * 16 
        minor_minor_split = self._dataset_size * 0
        val_split = 3  * 4
        val_minor_split = 1 * 4
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['black_hair', 'straight_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba_c_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba_c_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

    def IID(self):
        """
        UNIFORM test data
        """
        output = np.load(f"{self._root}/celeba_c_split/iid_test.npy")
        label = np.load(f"{self._root}/celeba_c_split/label_test.npy")

        return output, label

class CELEBA(MultipleDomainDataset):
    def __init__(self, root: str = '/data', 
                 dist_type: str = None, 
                 dataset_size: int = None, 
                 aug: str = 'no_aug', 
                 resize: bool = False, 
                 algo: str = 'ERM', 
                 split: str = 'train', 
                 ratio: float = 0.01,
                 attributes = None,
                 return_attr: bool = False,
                 hparams = None) -> None:

        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        """
        self.label_names = ['Female', 'Male']
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.resize = resize
        self.split = split
        self.input_shape = (3, 256, 256,) 
        self.num_classes = 2
        self.ratio = ratio
        self.algo = algo
        self.hparams = hparams
        self.return_attr = return_attr

        self.male = ['-1', '1']
        self.black_hair = ['-1', '1']
        self.smiling = ['-1', '1']
        self.straight_hair = ['-1', '1']
        self.impulse = ['-1', '1']
        self.snow = ['-1', '1']
        self.elastic = ['-1', '1']

        if split == 'train':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = True)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)

        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)

        elif split == 'test':
            self._imgs, self._labels = self.IID()
            
        if not self.return_attr:
            self._labels[self._labels == -1] = 0
            self._labels[self._labels == '-1'] = 0

        #self._labels[self._labels == '1'] = 1
        self.transform = self.get_transforms(aug)

    def __getitem__(self, index: int):

        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        
        if self.return_attr:
            return self.transform(torch.Tensor(self._imgs[index])), self._labels[index]
        else:
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)

    def UNIFORM(self, train: bool = True):

        split     = self._dataset_size * 1 * 7
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for male in ['-1', '1']:
            for black_hair in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)


    def SC(self, train: bool = True, ratio: float = 0.01, attributes: str = 'black_hair'):
        """
        1. Male
        2. Black_Hair
        3. Smiling
        4. Straight_Hair
        """
        split     = self._dataset_size * 1 * 14 
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == 'black_hair':

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'straight_hair':

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD(self, train: bool = True, attributes: str = 'black_hair'):
        major_split = self._dataset_size * 3 * 4 
        minor_split = self._dataset_size * 1 * 4 
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'black_hair':

            for male in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'straight_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def UDS(self, train: bool = True, attributes: str = 'black_hair'):
        split = self._dataset_size * 1 * 14 
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'black_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        for straight_hair in ['-1', '1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'smiling':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        for straight_hair in ['-1', '1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == 'straight_hair':

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        for straight_hair in ['-1']:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([male] * split)
                            val_label.append([male] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_LDD(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling']):
        major_split = self._dataset_size * 3 * 8 
        minor_split = self._dataset_size * 1 * 8 
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['black_hair', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['smiling', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            
        if attributes == ['smiling', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['straight_hair', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        if attributes == ['straight_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling']):
        split = self._dataset_size * 1 * 28 
        val_split = 1 * 8

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            #selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['black_hair', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)
                        train_label.append([male] * split)
                        val_label.append([male] * val_split)
                        generated_combinations.append((male, black_hair, smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD_UDS(self, train: bool = True, attributes = ['black_hair', 'smiling']):
        """
        LDD_UDS
        LDD
        +
        THERE IS NO (purple)
        """
        major_split = self._dataset_size * 3 * 8 
        minor_split = self._dataset_size * 1 * 8 
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == ['black_hair', 'smiling']:

            for male in ['-1', '1']:
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['black_hair', 'straight_hair']:

            for male in ['-1', '1']:
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1', '1']:
                    for straight_hair in ['-1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    for straight_hair in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for straight_hair in ['-1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair']:

            for male in ['-1', '1']:
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    for smiling in ['-1', '1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling']:

            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([male] * major_split)
                        val_label.append([male] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, split, val_split in zip(['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1', '1']:
                    for smiling in ['-1']:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [male] * split], axis = 0)
                        val_label = np.concatenate([val_label, [male] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def SC_LDD_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['black_hair', 'smiling', 'straight_hair']):
        major_split = self._dataset_size * 3 * 16 
        minor_split = self._dataset_size * 1 * 16 
        minor_minor_split = self._dataset_size * 0
        val_split = 3  * 4
        val_minor_split = 1 * 4
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['black_hair', 'smiling', 'straight_hair']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


        elif attributes == ['black_hair', 'straight_hair', 'smiling']:

            for male, black_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, black_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'black_hair', 'straight_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for straight_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for straight_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, self.smiling, ['-1']))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['smiling', 'straight_hair', 'black_hair']:

            for male, smiling in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_-1.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, smiling, '-1'))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, smiling, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_{smiling}_1.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, smiling, '1'))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'black_hair', 'smiling']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for smiling in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_-1_{smiling}_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, '-1', smiling, straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for smiling in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_1_{smiling}_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, '1', smiling, straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, self.black_hair, ['-1'], self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['straight_hair', 'smiling', 'black_hair']:

            for male, straight_hair in zip(['-1', '1'], ['-1', '1']):
                for black_hair in ['-1']:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_-1_{straight_hair}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([male] * major_split)
                    val_label.append([male] * val_split)
                    generated_combinations.append((male, black_hair, '-1', straight_hair))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for male, straight_hair, split, val_split in zip(['-1', '1'], ['-1', '1'], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for black_hair in ['-1']:
                    output = np.load(f"{self._root}/celeba/celeba_split/{male}_{black_hair}_1_{straight_hair}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [male] * split], axis = 0)
                    val_label = np.concatenate([val_label, [male] * val_split], axis = 0)
                    generated_combinations.append((male, black_hair, '1', straight_hair))
            
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
            
            all_combinations = list(itertools.product(self.male, ['-1'], self.smiling, self.straight_hair))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/celeba/celeba_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

    def IID(self):
        """
        UNIFORM test data
        """
        test_output = np.array([])
        attr_label  = [] 
	    
        if self.return_attr:
            for male in ['-1', '1']:
                for black_hair in ['-1', '1']:
                    for smiling in ['-1', '1']:
                        for straight_hair in ['-1', '1']:
                            if len(test_output) == 0:
                                test_output = np.load(f"{self._root}/celeba/celeba_split_/test_{male}_{black_hair}_{smiling}_{straight_hair}.npy")[:50]
                            else:
                                output = np.load(f"{self._root}/celeba/celeba_split_/test_{male}_{black_hair}_{smiling}_{straight_hair}.npy")
                                test_output = np.append(test_output, output[:50], axis = 0)
                            attr_label.extend([[int(male), int(black_hair), int(smiling), int(straight_hair)]] * 50)
                            
            return test_output, attr_label  

        else:
            output = np.load(f"{self._root}/celeba/celeba_split/iid_test.npy")
            label = np.load(f"{self._root}/celeba/celeba_split/label_test.npy")

            return output, label  
                                                                                                  

class DEEPFASHION(MultipleDomainDataset):
    def __init__(self, root: str = '/data', dist_type: str = None, dataset_size: int = None, aug: str = 'no_aug', resize: bool = False, algo: str = 'ERM', split: str = 'train', ratio: float = 0.01, attributes = None, hparams = None) -> None:

        """
        dist_type: SC, LDD, UDS, SC_LDD, SC_UDS, LDD_UDS, SC_LDD_UDS
        dataset_size: 1 for MAIN EXPERIMENTS
        split: train, val, test
        attributes: str or List of str
        """
        self.label_names = ['dress', 'no_dress']
        self._root: str  = root
        self._dataset_size: int = dataset_size
        self.resize = resize
        self.split = split
        self.input_shape = (3, 128, 128,)
        self.num_classes = 2
        self.ratio = ratio
        self.algo = algo

        self.hparams = hparams
        self._dress = [11, 12]
        self._texture = [0, 5]
        self._sleeve = [7, 8]
        self._fabric = [18, 19]
        if split == 'train':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = True)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = True, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = True, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = True, ratio = self.ratio, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = True, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = True, ratio = self.ratio, attributes = attributes)

        if split == 'val':
            if dist_type == 'UNIFORM':
                self._imgs, self._labels = self.UNIFORM(train = False)
            elif dist_type == 'SC':
                self._imgs, self._labels = self.SC(train = False, attributes = attributes)
            elif dist_type == 'LDD':
                self._imgs, self._labels = self.LDD(train = False, attributes = attributes)
            elif dist_type == 'UDS':
                self._imgs, self._labels = self.UDS(train = False, attributes = attributes)
            elif dist_type == 'SC_LDD':
                self._imgs, self._labels = self.SC_LDD(train = False, attributes = attributes)
            elif dist_type == 'SC_UDS':
                self._imgs, self._labels = self.SC_UDS(train = False, attributes = attributes)
            elif dist_type == 'LDD_UDS':
                self._imgs, self._labels = self.LDD_UDS(train = False, attributes = attributes)
            else:
                self._imgs, self._labels = self.SC_LDD_UDS(train = False, attributes = attributes)

        elif split == 'test':
            self._imgs, self._labels = self.IID()

        self._labels[self._labels == 11] = 0
        self._labels[self._labels == 12] = 1

        self.transform = self.get_transforms(aug)

    def __getitem__(self, index: int):

        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index]), index
        return self.transform(torch.Tensor(self._imgs[index])), int(self._labels[index])

    def __len__(self) -> int:

        return len(self._imgs)

    def UNIFORM(self, train: bool = True):

        split     = self._dataset_size * 1 * 6
        val_split = 1 * 3

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        for dress in [11, 12]:
            for texture in [0, 5]:
                for sleeve in [7, 8]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)                    
                        train_label.append([dress] * split)
                        val_label.append([dress] * val_split)

        if train:
            return train_output, np.reshape(train_label, -1)
        else:
            return val_output, np.reshape(val_label, -1)       

    def SC(self, train: bool = True, ratio: float = 0.01, attributes: str = 'texture'):
        """
        1. [0,   5]
        2. [7,   8]
        3. [18, 19]
        4. [11, 12] target for main experiments
        attributes: 'texture', 'sleeve', 'fabric'
        """
        split     = self._dataset_size * 1 * 12 
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == 'texture':

            for dress, texture in zip([11, 12], [0, 5]):
                for sleeve in [7, 8]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)                    
                        train_label.append([dress] * split)
                        val_label.append([dress] * val_split)
                        generated_combinations.append((dress, texture, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
        
        elif attributes == 'sleeve':

            for dress, sleeve in zip([11, 12], [7, 8]):
                for texture in [0, 5]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)                    
                        train_label.append([dress] * split)
                        val_label.append([dress] * val_split)
                        generated_combinations.append((dress, texture, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'fabric':

            for dress, fabric in zip([11, 12], [18, 19]):
                for texture in [0, 5]:
                    for sleeve in [7, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output, val_output = output[:split], output[split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                            train_output = np.append(train_output, output[:split], axis = 0)
                            val_output = np.append(val_output, output[split:], axis = 0)                    
                        train_label.append([dress] * split)
                        val_label.append([dress] * val_split)
                        generated_combinations.append((dress, texture, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            

    def LDD(self, train: bool = True, attributes: str = 'texture'):
        major_split = self._dataset_size * 3 * 4
        minor_split = self._dataset_size * 1 * 4
        minor_minor_split = self._dataset_size * 0
        val_split = 3
        val_minor_split = 1
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'texture':

            for dress in [11, 12]:
                for sleeve in [7, 8]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7, 8]:
                    for fabric in [18, 19]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'sleeve':

            for dress in [11, 12]:
                for texture in [0, 5]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0, 5]:
                    for fabric in [18, 19]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)     

        elif attributes == 'fabric':

            for dress in [11, 12]:
                for texture in [0, 5]:
                    for sleeve in [7, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0, 5]:
                    for fabric in [18, 19]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)        


    def UDS(self, train: bool = True, attributes = 'texture'):
        split = self._dataset_size * 1 * 12
        val_split = 1 * 4

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == 'texture':

            for dress in [11, 12]:
                for texture in [0]:
                    for sleeve in [7, 8]:
                        for fabric in [18, 19]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([dress] * split)
                            val_label.append([dress] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'sleeve':

            for dress in [11, 12]:
                for texture in [0, 5]:
                    for sleeve in [7]:
                        for fabric in [18, 19]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([dress] * split)
                            val_label.append([dress] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == 'fabric':

            for dress in [11, 12]:
                for texture in [0, 5]:
                    for sleeve in [7, 8]:
                        for fabric in [18]:
                            if len(train_output) == 0:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output, val_output = output[:split], output[split:]
                            else:
                                output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_{fabric}.npy")[:split + val_split]
                                train_output = np.append(train_output, output[:split], axis = 0)
                                val_output = np.append(val_output, output[split:], axis = 0)
                            train_label.append([dress] * split)
                            val_label.append([dress] * val_split)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

    def SC_LDD(self, train: bool = True, ratio: float = 0.01, attributes = ['texture', 'sleeve']):
        major_split = self._dataset_size * 3 * 7
        minor_split = self._dataset_size * 1 * 7
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['texture', 'sleeve']:

            for dress, texture in zip([11, 12], [0, 5]):
                for fabric in [18, 19]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, texture, split, val_split in zip([11, 12], [0, 5], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18, 19]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, 8, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            
        elif attributes == ['texture', 'fabric']:

            for dress, texture in zip([11, 12], [0, 5]):
                for sleeve in [7, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, texture, split, val_split in zip([11, 12], [0,5], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7, 8]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_19.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, sleeve, 19))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['sleeve', 'texture']:

            for dress, sleeve in zip([11, 12], [7, 8]):
                for fabric in [18, 19]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, sleeve, split, val_split in zip([11, 12], [7, 8], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18, 19]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, 5, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['sleeve', 'fabric']:

            for dress, sleeve in zip([11, 12], [7, 8]):
                for texture in [0, 5]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, sleeve, split, val_split in zip([11, 12], [7, 8], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0, 5]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_19.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, sleeve, 19))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['fabric', 'texture']:

            for dress, fabric in zip([11, 12], [18, 19]):
                for sleeve in [7, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, fabric, split, val_split in zip([11, 12], [18, 19], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7, 8]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, 5, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['fabric', 'sleeve']:

            for dress, fabric in zip([11, 12], [18, 19]):
                for texture in [0, 5]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, fabric, split, val_split in zip([11, 12], [18, 19], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0, 5]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, 8, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)



    def SC_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['texture', 'sleeve']):
        split = self._dataset_size * 1 * 25
        val_split = 1 * 5

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []
        
        if attributes == ['texture', 'sleeve']:
            for dress, texture in zip([11, 12], [0, 5]):
                for fabric in [18, 19]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, self._texture, [7], self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
        
        elif attributes == ['texture', 'fabric']:
            for dress, texture in zip([11, 12], [0, 5]):
                for sleeve in [7, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, [18]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
 
        elif attributes == ['sleeve', 'texture']:
            for dress, sleeve in zip([11, 12], [7, 8]):
                for fabric in [18, 19]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, [0], self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
 
        elif attributes == ['sleeve', 'fabric']:
            for dress, sleeve in zip([11, 12], [7, 8]):
                for texture in [0, 5]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, [18]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
 
        elif attributes == ['fabric', 'texture']:
            for dress, fabric in zip([11, 12], [18, 19]):
                for sleeve in [7, 8]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, [0], self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
 
        elif attributes == ['fabric', 'sleeve']:
            for dress, fabric in zip([11, 12], [18, 19]):
                for texture in [0, 5]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:split + val_split]
                        train_output, val_output = output[:split], output[split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                    train_label.append([dress] * split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))
    
            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)
    
            all_combinations = list(itertools.product(self._dress, [0], self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]
    
            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)
    
            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)
    
            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def LDD_UDS(self, train: bool = True, attributes = ['texture', 'sleeve']):
        major_split = self._dataset_size * 3 * 7
        minor_split = self._dataset_size * 1 * 7
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []

        if attributes == ['texture', 'sleeve']:

            for dress in [11, 12]:
                for sleeve in [7]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7]:
                    for fabric in [18, 19]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)
                
        elif attributes == ['texture', 'fabric']:

            for dress in [11, 12]:
                for fabric in [18]:
                    for sleeve in [7, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18]:
                    for sleeve in [7, 8]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)
            
        elif attributes == ['sleeve', 'texture']:

            for dress in [11, 12]:
                for texture in [0]:
                    for fabric in [18, 19]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0]:
                    for fabric in [18, 19]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)
            
        elif attributes == ['sleeve', 'fabric']:

            for dress in [11, 12]:
                for fabric in [18]:
                    for texture in [0, 5]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18]:
                    for texture in [0, 5]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)
            
        elif attributes == ['fabric', 'texture']:

            for dress in [11, 12]:
                for texture in [0]:
                    for sleeve in [7, 8]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0]:
                    for sleeve in [7, 8]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_19.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['fabric', 'sleeve']:

            for dress in [11, 12]:
                for fabric in [18]:
                    for texture in [0, 5]:
                        if len(train_output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output, val_output = output[:major_split], output[major_split:]
                        else:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                            train_output = np.append(train_output, output[:major_split], axis = 0)
                            val_output = np.append(val_output, output[major_split:], axis = 0)
                        train_label.append([dress] * major_split)
                        val_label.append([dress] * val_split)

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, split, val_split in zip([11, 12], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18]:
                    for texture in [0, 5]:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                        train_output = np.append(train_output, output[:split], axis = 0)
                        val_output = np.append(val_output, output[split:], axis = 0)
                        train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                        val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)

            if train:
                return train_output, np.reshape(train_label, -1)
            else:
                return val_output, np.reshape(val_label, -1)


    def SC_LDD_UDS(self, train: bool = True, ratio: float = 0.01, attributes = ['texture', 'sleeve', 'fabric']):
        major_split = self._dataset_size * 3 * 13
        minor_split = self._dataset_size * 1 * 13
        minor_minor_split = self._dataset_size * 0
        val_split = 3 * 2
        val_minor_split = 1 * 2
        val_minor_minor_split = 0

        train_output = np.array([])
        val_output   = np.array([])
        train_label = []
        val_label   = []
        generated_combinations = []

        if attributes == ['texture', 'sleeve', 'fabric']:

            for dress, texture in zip([11, 12], [0, 5]):
                for fabric in [18]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, texture, split, val_split in zip([11, 12], [0, 5], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, 8, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, [18]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['texture', 'fabric', 'sleeve']:

            for dress, texture in zip([11, 12], [0, 5]):
                for sleeve in [7]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, texture, split, val_split in zip([11, 12], [0, 5], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_19.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, sleeve, 19))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, [7], self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['sleeve', 'texture', 'fabric']:

            for dress, sleeve in zip([11, 12], [7, 8]):
                for fabric in [18]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, sleeve, split, val_split in zip([11, 12], [7, 8], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for fabric in [18]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, 5, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, self._sleeve, [18]))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)
            
        elif attributes == ['sleeve', 'fabric', 'texture']:

            for dress, sleeve in zip([11, 12], [7, 8]):
                for texture in [0]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_18.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, sleeve, 18))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, sleeve, split, val_split in zip([11, 12], [7, 8], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_{sleeve}_19.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, sleeve, 19))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, [0], self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['fabric', 'texture', 'sleeve']:

            for dress, fabric in zip([11, 12], [18, 19]):
                for sleeve in [7]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_0_{sleeve}_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, 0, sleeve, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, fabric, split, val_split in zip([11, 12], [18, 19], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for sleeve in [7]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_5_{sleeve}_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, 5, sleeve, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, self._texture, [7], self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)

        elif attributes == ['fabric', 'sleeve', 'texture']:

            for dress, fabric in zip([11, 12], [18, 19]):
                for texture in [0]:
                    if len(train_output) == 0:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output, val_output = output[:major_split], output[major_split:]
                    else:
                        output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_7_{fabric}.npy")[:major_split + val_split]
                        train_output = np.append(train_output, output[:major_split], axis = 0)
                        val_output = np.append(val_output, output[major_split:], axis = 0)
                    train_label.append([dress] * major_split)
                    val_label.append([dress] * val_split)
                    generated_combinations.append((dress, texture, 7, fabric))

            train_label = np.reshape(train_label, -1)
            val_label = np.reshape(val_label, -1)

            for dress, fabric, split, val_split in zip([11, 12], [18, 19], [minor_split, minor_minor_split], [val_minor_split, val_minor_minor_split]):
                for texture in [0]:
                    output = np.load(f"{self._root}/deepfashion/deepfashion_split/{dress}_{texture}_8_{fabric}.npy")[:split + val_split]
                    train_output = np.append(train_output, output[:split], axis = 0)
                    val_output = np.append(val_output, output[split:], axis = 0)
                    train_label = np.concatenate([train_label, [dress] * split], axis = 0)
                    val_label = np.concatenate([val_label, [dress] * val_split], axis = 0)
                    generated_combinations.append((dress, texture, 8, fabric))

            # Counterexample
            num_count = math.ceil(len(train_output) * ratio)

            all_combinations = list(itertools.product(self._dress, [0], self._sleeve, self._fabric))
            unique_combinations = [combo for combo in all_combinations if combo not in generated_combinations]

            selected_combination = [random.choice(unique_combinations) for i in range(num_count)]
            train_label = np.reshape(train_label, -1)
            train_output = np.stack(train_output, axis=0)

            for comb in selected_combination:
                train_output = np.append(train_output, np.array([np.load(f"{self._root}/deepfashion/deepfashion_split/{comb[0]}_{comb[1]}_{comb[2]}_{comb[3]}.npy", allow_pickle= True)[0]]), axis = 0)
                train_label = np.append(train_label, [comb[0]], axis = 0)

            if train:
                return train_output, train_label
            else:
                return val_output, np.reshape(val_label, -1)


    def IID(self):
        """
        UNIFORM test data
        target: dress, texture, sleeve, fabric
        """
        output = np.array([])
        label  = []
        for dress in [11, 12]:
            for texture in [0, 5]:
                for sleeve in [7, 8]:
                    for fabric in [18, 19]:
                        if len(output) == 0:
                            output = np.load(f"{self._root}/deepfashion/deepfashion_split/test_{dress}_{texture}_{sleeve}_{fabric}.npy")
                        else:
                            output = np.append(output, np.load(f"{self._root}/deepfashion/deepfashion_split/test_{dress}_{texture}_{sleeve}_{fabric}.npy"), axis = 0)
                        label.append([dress] * 5)
        
        label = np.array(label)
        
        return np.array(output), np.reshape(label, -1)
