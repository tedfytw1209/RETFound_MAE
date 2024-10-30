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

class CSV_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,is_train,transfroms=[]):
        #common args
        self.transfroms = transfroms
        self.root_dir = img_dir
        self.loader = datasets.folder.default_loader
        data = pd.read_csv(csv_file)
        self.annotations = data[data['split']==is_train]
        self.classes = [str(c) for c in self.annotations['label'].unique()]
        self.num_class = len(self.classes)
        self.channel = 3

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations['OCT'][idx])
        image = self.loader(img_name)
        label = int(self.annotations['label'][idx].values)

        for transfrom in self.transfroms:
            image = transfrom(image)

        return image, label

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    img_dir = ''
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
