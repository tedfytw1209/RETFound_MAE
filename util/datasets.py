# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

AD_LIST = ['control','mci','ad']

class CSV_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,is_train,transfroms=[],k=0):
        #common args
        self.transfroms = transfroms
        self.root_dir = img_dir
        self.loader = datasets.folder.default_loader
        data = pd.read_csv(csv_file)
        self.annotations = data[data['split']==is_train]
        print('Split: ', is_train,' Data len: ', self.annotations.shape[0])
        self.classes = [str(c) for c in self.annotations['label'].unique()]
        self.num_class = len(self.classes)
        #assert index order control, mci, ad
        self.class_to_idx = {}
        i = 0
        for c in AD_LIST:
            if c in self.classes:
                self.class_to_idx[c] = i
                i+=1
        print('Class to idx: ', self.class_to_idx)
        self.channel = 3
        image_names, labels = self.annotations['OCT'], self.annotations['label']
        #case for 2.5D
        image_sample = image_names[0]
        if image_sample.endswith(']') and image_sample.startswith('['):
            if is_train == 'train':
                samples = []
                for image_name,label in zip(image_names, labels):
                    image_name = image_name[1:-1].split(',')
                    for each_name in image_name:
                        samples.append((each_name+'.jpg', self.class_to_idx[str(label)]))
                self.half3D = False
            else:
                samples = [([each_name+'.jpg' for each_name in image_name[1:-1].split(',')], self.class_to_idx[str(label)]) for image_name,label in zip(image_names, labels)]
                self.half3D = True
        else:
            samples = [(image_name+'.jpg', self.class_to_idx[str(label)]) for image_name,label in zip(image_names, labels)]
            self.half3D = False
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.k = k #!! not used now

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.half3D: #output multiple images
            img_name = [os.path.join(self.root_dir, each_name) for each_name in sample[0]]
            image = [self.loader(each_name) for each_name in img_name]
            image = [self.transfroms(each_image) for each_image in image]
        else:
            img_name = os.path.join(self.root_dir, sample[0])
            image = self.loader(img_name)
            image = self.transfroms(image)
        
        label = int(sample[1])

        return image, label

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    img_dir = '/orange/bianjiang/tienyu/OCT_AD/all_images/'
    if args.data_path.endswith('.csv'):
        dataset = CSV_Dataset(args.data_path, img_dir, is_train, transform)
    else:
        root = os.path.join(args.data_path, is_train)
        dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            #normalize=False,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
