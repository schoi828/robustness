import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter
from domainbed.datasets import MultipleDomainDataset
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
from wilds import get_dataset
import torchvision.transforms as transforms
from domainbed.fmow import FMoW

def crop_wilds(image):
    return crop(image, 10, 0, 400, 448)

def get_counts(labels):
    values, counts = np.unique(labels, return_counts=True)
    sorted_tuples = zip(*sorted(zip(values, counts))) # this just ensures we are getting the counts in the sorted order of the keys
    values, counts = [ list(tuple) for tuple in  sorted_tuples]
    fracs   = 1 / torch.Tensor(counts)
    return fracs / torch.max(fracs)

LOCATION_MAP = {1: 0, 78: 1}
LOCATION_MAP_INV = {0: 1, 1: 78}

class iwildcam(FMoW):
    def __init__(self, root='../../../data', split='train', aug='no_aug',algo='ERM',  hparams=None):
        dataset = get_dataset(dataset="iwildcam",root_dir=root, download=True)
        self.dataset = dataset.get_subset(split=split)
        self.input_shape = (3, 224, 224,)
        self.num_classes = 182
        self.resize = True ############################
        self.split = split
        self.algo = algo
        self.random_aug = True if '+R' in aug else False
        aug = aug.split('+R')[0]
        self.hparams=hparams
        self.transform = None if self.algo == 'zeroshot'else self.get_transforms(aug) 
        csv = pd.read_csv(f'{root}/categories.csv')
        label_names = []
        for row in csv.iterrows():
            idx, content = row
            if idx > 181:
                break
            name = content['other_name'].split('|')[0]
            label_names.append(name)

        self.label_names = label_names
        
        if algo in ['ADA', 'ME_ADA'] and split=='train':
            self.samples = [(data, label) for data, label, _ in self.dataset]
            i = random.randint(0,10000)
            ada_root = os.path.join(root,f'ADA_{i}')
            while os.path.exists(ada_root):
                i=random.randint(0,10000)
                ada_root = os.path.join(root,f'ADA_{i}')

            self.ada_root = ada_root
            os.makedirs(self.ada_root)
            self.ada_samples = []

        self.ori_samples = None

class iwildcam7(MultipleDomainDataset):
    """
    Specific subset of WILDS containing 6 classes and 2 test locations.
    """
    def __init__(self, root='../../../data', split='train', aug='no_aug',algo='ERM', hparams=None):
        root_path = os.path.join(root,'iwildcam_v2.0')
        self.root = os.path.join(root_path,'train')
        self.df = pd.read_csv(os.path.join(root_path,f'{split}_subset.csv'))
        self.input_shape = (3, 224, 224,)
        self.num_classes = 7
        self.resize = True ############################
        self.split = split
        self.algo = algo
        aug = aug.split('+R')[0]
        self.hparams=hparams
        self.transform = None if self.algo == 'zeroshot'else self.get_transforms(aug) 
        self.label_names = sorted(self.df.y.unique())
        self.label_map = {j:i for i, j in enumerate(self.label_names)}
        self.labels = [self.label_map[i] for i in self.df.y]
        self.samples = [(os.path.join(self.root, i), l) for (i, l) in zip(self.df.filename, self.labels)]
        self.targets = [l for _, l in self.samples]
        self.classes = list(sorted(self.df.y.unique()))
        self.locations = self.df.location_remapped.unique()
        self.location_map = {j:i for i, j in enumerate(self.locations)}
        self.location_labels = [self.location_map[i] for i in self.df.location_remapped]
        self.groups = self.location_labels
        self.class_weights = get_counts(self.labels)
        self.group_names = self.locations
        
        self.label_names = ['background', 'cattle', 'elephants', 'impalas', 'zebras', 'giraffes', 'dik-diks']
        if algo in ['ADA', 'ME_ADA'] and split=='train':
            i = random.randint(0,10000)
            ada_root = os.path.join(root,f'ADA_{i}')
            while os.path.exists(ada_root):
                i=random.randint(0,10000)
                ada_root = os.path.join(root,f'ADA_{i}')

            self.ada_root = ada_root
            os.makedirs(self.ada_root)
            self.ada_samples = []

        print(f"Num samples per class {Counter(self.labels)}")
        self.ori_samples = None

    def change_mode(self, mode='mixed'):
        assert mode in ['mixed', 'OOD', 'IID'], 'mode should be mixed, OOD, or IID'
        if self.ori_samples is None:
            self.ori_samples = self.samples

        if mode == 'mixed':
            self.samples = self.ori_samples
        elif mode == 'OOD':
            self.samples = self.ori_samples[:492]
        elif mode == 'IID':
            self.samples = self.ori_samples[492:]

    def preload(self):
        imgs = np.array([])
        labels = []
        for sample in self.samples:
            img_path, label = sample
            img = Image.open(img_path)

            labels.append(label)
            imgs = np.append(imgs,np.array(img)/255)
        return imgs, labels
    
    def build_imgnet(self,img_size):

        if hasattr(self, 'hparams'):
            scale_lb = self.hparams['scale_lower_bound']
        else:
            scale_lb = 0.08
        randomresizecrop = transforms.RandomResizedCrop(img_size, scale=(scale_lb,1))

        return transforms.Compose([
            randomresizecrop,
            #transforms.RandomHorizontalFlip(), #not mentioned in Appendix of the deepmind paper
            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
            #transforms.ToTensor(),
        ])
    
    def build_augmix(self,img_size):
        if hasattr(self, 'hparams'):
            severity=self.hparams['severity']
            mixture_width=self.hparams['mixture_width']
        else:
            severity=3
            mixture_width=3
        return transforms.AugMix(severity=severity,mixture_width=mixture_width) #PIL image recommended. For torch tensor, it should be of torch.uint8

    def build_randaug(self,img_size):
        if hasattr(self, 'hparams'):
            num_ops = self.hparams['num_ops']
            magnitude=self.hparams['magnitude']
        else:
            num_ops=3
            magnitude=5
        return transforms.RandAugment(num_ops,magnitude) #PIL image recommended. For torch tensor, it should be of torch.uint8
    
    def build_autoaug(self,img_size):

        policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET

        return transforms.AutoAugment(policy=policy) #PIL image recommended. For torch tensor, it should be of torch.uint8

    def get_transforms(self,mode='no_aug', gray=False):
        assert mode in ['no_aug','imgnet','augmix','randaug','autoaug'], 'incorrect preprocessing mode'

        #normalize to [-1,1] (according to the official code in tensorflow/jax/chex)
        mean = [0.5] if gray else [0.5]*3
        std = [0.5] if gray else [0.5]*3
        normalize = transforms.Normalize(mean=mean, std=std)
        
        transforms_list = []

        if self.resize:
            resize = [transforms.ToTensor(),
                                  transforms.Resize((448, 448)),
                                  transforms.Lambda(crop_wilds),
                                  transforms.Resize((224, 224)),
                                  transforms.ToPILImage()]
            transforms_list+=resize

        if mode != 'no_aug' and self.split == 'train':
            if mode != 'imgnet':
                transforms_list.append(transforms.ToTensor())
                transforms_list.append(lambda x : transforms.functional.convert_image_dtype(x,torch.uint8))
            transforms_list.append(getattr(self,f'build_{mode}')(self.input_shape[1]))
            if mode != 'imgnet':
                transforms_list.append(lambda x : transforms.functional.convert_image_dtype(x,torch.float32))
        
        if mode not in['augmix','autoaug','randaug'] or self.split!='train':
            transforms_list.append(transforms.ToTensor())
        
        transforms_list.append(normalize)
        preprocess = transforms.Compose(transforms_list)

        return preprocess
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        if self.algo in ['BPA','PnD','OccamNets'] and self.split == 'train':
            return img, label,idx
        return img, label

    def inspect_location(self, location):
        assert location in self.locations
        location_df = self.df[self.df.location_remapped == location]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(location_df))))
        location_df['y'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Location {location} (n={len(location_df)}) class counts')
        axs[1].imshow(Image.open(os.path.join(self.root, location_df.iloc[idx].filename)))
        axs[1].set_title(f'Location {location} (n={len(location_df)}) class {location_df.iloc[idx].y} (idx={idx})')
        axs[1].axis('off')
        plt.show()

    def inspect_class(self, class_idx):
        assert class_idx in self.classes
        class_df = self.df[self.df.y == class_idx]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        idx = np.random.choice(list(range(len(class_df))))
        class_df['location_remapped'].value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title(f'Class {class_idx} (n={len(class_df)}) location counts')
        axs[1].imshow(Image.open(os.path.join(self.root, class_df.iloc[idx].filename)))
        axs[1].set_title(f'Class {class_idx} (n={len(class_df)}) location {class_df.iloc[idx].location_remapped} (idx={idx}) ({class_df.iloc[idx].filename})')
        axs[1].axis('off')
        plt.show()