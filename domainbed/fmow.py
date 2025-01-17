import torch
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from domainbed.datasets import MultipleDomainDataset
from torchvision import transforms
import random
from torchvision.transforms.functional import crop
from wilds.datasets.fmow_dataset import FMoWDataset

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


class FMoW(MultipleDomainDataset):
    """
    Specific subset of WILDS containing 6 classes and 2 test locations.
    """
    def __init__(self, root='/data/ubuntu/robustness/data', split='train', aug='no_aug',algo='ERM',hparams=None):
        dataset = FMoWDataset(root_dir=root, download=True)
        self.dataset = dataset.get_subset(split)
        self.input_shape = (3, 224, 224,)
        self.num_classes = 62
        self.resize = True
        self.split = split
        self.algo = algo
        self.random_aug = True if '+R' in aug else False
        aug = aug.split('+R')[0]
        self.root = root
        self.hparams=hparams
        self.transform = None if self.algo == 'zeroshot'else self.get_transforms(aug) 
        
        
        self.label_names = ["airport", "airport_hangar", "airport_terminal", "amusement_park", "aquaculture", 
                            "archaeological_site", "barn", "border_checkpoint", "burial_site", "car_dealership", 
                            "construction_site", "crop_field", "dam", "debris_or_rubble", "educational_institution", 
                            "electric_substation", "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
                              "gas_station", "golf_course", "ground_transportation_station", "helipad", "hospital",
                                "impoverished_settlement", "interchange", "lake_or_pond", "lighthouse", "military_facility", 
                                "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park", 
                                "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track", "railway_bridge", 
                                "recreational_facility", "road_bridge", "runway", "shipyard", "shopping_mall", "single-unit_residential", 
                                "smokestack", "solar_farm", "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", 
                                "toll_booth", "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]

        if algo in ['ADA', 'ME_ADA'] and split=='train':
            self.init_ADA()
        self.ori_samples = None
    
    def init_ADA(self):
        self.samples = [(data, label) for data, label, _ in self.dataset]
        i = random.randint(0,10000)
        ada_root = os.path.join(self.root,f'ADA_{i}')
        while os.path.exists(ada_root):
            i=random.randint(0,10000)
            ada_root = os.path.join(self.root,f'ADA_{i}')

        self.ada_root = ada_root
        os.makedirs(self.ada_root)
        self.ada_samples = []

    def preload(self):
        imgs = np.array([])
        labels = []
        for sample in self.samples:
            img_path, label = sample
            img = Image.open(img_path)

            labels.append(label)
            imgs = np.append(imgs,np.array(img)/255)
        return imgs, labels
    
    def get_transforms(self,mode='no_aug', gray=False):
        assert mode in ['no_aug','imgnet','augmix','randaug','autoaug'], 'incorrect preprocessing mode'

        #normalize to [-1,1] (according to the official code in tensorflow/jax/chex)
        mean = [0.5] if gray else [0.5]*3
        std = [0.5] if gray else [0.5]*3
        normalize = transforms.Normalize(mean=mean, std=std)
        
        transforms_list = []

        if self.resize:
            resize = [transforms.ToTensor(),
                                #   transforms.Grayscale(num_output_channels=3),
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
        return len(self.dataset)

    def __getitem__(self, idx):
            
        if 'ADA' in self.algo and self.split == 'train':
            img, label = self.samples[idx]
        else:
            img, label, _ = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        if self.algo in ['BPA','PnD', 'OccamNets'] and self.split == 'train':
            return img, label,idx
        return img, label
