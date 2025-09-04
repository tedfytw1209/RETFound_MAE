# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os

import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import pydicom
import matplotlib.pyplot as plt
import math

AD_LIST = ['control','mci','ad']
def process_dcm(dicom_root,sample_name,k=0):
    sample_path = os.path.join(dicom_root,sample_name)
    #print(sample_path)
    ds = pydicom.dcmread(sample_path)
    new_image = ds.pixel_array.astype(float)
    #print(new_image.shape)
    final_images = []
    depth = new_image.shape[0]
    med_point = (new_image.shape[0]+1)//2
    for i in range(med_point-k//2,med_point+k//2+1):
        med_image = new_image[i]
        #scaled_image = (np.maximum(med_image, 0) / med_image.max()) * 255.0
        #scaled_image = np.uint8(scaled_image)
        scaled_image = med_image
        final_image = Image.fromarray(scaled_image)
        final_images.append(final_image)
    #final_image.show()
    return final_images, depth

def select_n_slices(k,depth):
    if k<1:
        return int(k*depth/2)
    else:
        return k//2

def get_path(row,col):
    return os.path.join(row['folder'],row[col])

Thickness_List = ['ILM (ILM)_RNFL-GCL (RNFL-GCL)', 'RNFL-GCL (RNFL-GCL)_GCL-IPL (GCL-IPL)', 'GCL-IPL (GCL-IPL)_IPL-INL (IPL-INL)', 
                  'IPL-INL (IPL-INL)_INL-OPL (INL-OPL)', 'INL-OPL (INL-OPL)_OPL-Henles fiber layer (OPL-HFL)', 
                  'OPL-Henles fiber layer (OPL-HFL)_Boundary of myoid and ellipsoid of inner segments (BMEIS)', 
                  'Boundary of myoid and ellipsoid of inner segments (BMEIS)_IS/OS junction (IS/OSJ)', 'IS/OS junction (IS/OSJ)_Inner boundary of OPR (IB_OPR)', 'Inner boundary of OPR (IB_OPR)_Inner boundary of RPE (IB_RPE)', 'Inner boundary of RPE (IB_RPE)_Outer boundary of RPE (OB_RPE)'
                  ]

class CSV_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,is_train,transfroms=[],k=0,class_to_idx={}, modality='OCT', patient_ids=None, pid_key = 'patient_id', select_layers=None,th_resize=True,th_heatmap=False):
        #common args
        self.transfroms = transfroms
        self.root_dir = img_dir
        self.loader = datasets.folder.default_loader
        data = pd.read_csv(csv_file)
        if not isinstance(is_train, list):
            is_train_l = [is_train]
        else:
            is_train_l = is_train
        is_train = is_train_l[0]
        if patient_ids is not None:
            self.annotations = data[data[pid_key].isin(patient_ids)]
            self.annotations['split'] = is_train
        elif 'split' in data.columns:
            self.annotations = data[data['split'].isin(is_train_l)]
        else:
            self.annotations = data
        print('Split: ', is_train_l,' Data len: ', self.annotations.shape[0])
        self.classes = [str(c) for c in self.annotations['label'].unique()]
        self.num_class = len(self.classes)
        #assert index order control, mci, ad
        if not class_to_idx:
            self.class_to_idx = {}
            i = 0
            for c in AD_LIST:
                if c in self.classes:
                    self.class_to_idx[c] = i
                    i+=1
            if i==0: #not in AD_LIST
                for c in sorted(self.classes):
                    self.class_to_idx[c] = int(c)
        else:
            self.class_to_idx = class_to_idx
        print('Class to idx: ', self.class_to_idx)
        self.channel = 3
        if th_resize:
            th_col = 'Thickness Resize Name'
        else:
            th_col = 'Thickness Name'
        if modality == 'OCT' and 'OCT' in self.annotations.columns:
            image_names = self.annotations['OCT']
            print('OCT images: ', len(image_names))
            print('OCT image example: ', image_names.head())
        elif modality == 'CFP' or modality == 'Fundus' and 'folder' in self.annotations.columns and 'fundus_imgname' in self.annotations.columns:
            image_names = self.annotations['folder'] + '/' + self.annotations['fundus_imgname']
            print('CFP images: ', len(image_names))
            print('CFP image example: ', image_names.head())
        elif modality == 'Thickness':
            image_names = self.annotations.apply(lambda row: get_path(row, th_col), axis=1)
            print('Thickness images: ', len(image_names))
            print('Thickness image example: ', image_names.head())
        else:
            print('Incompatible modality or missing columns: ', self.annotations.columns)
            image_names = self.annotations['image']
            print('Images: ', len(image_names))
        labels = self.annotations['label']
        #case for 2.5D
        if k>0:
            dcm_depths = self.annotations['depth']
            if is_train == 'train':
                samples = []
                for image_name,label,depth in zip(image_names, labels, dcm_depths):
                    med_point = (depth+1)//2
                    n_slice = select_n_slices(k,depth)
                    idx_start, idx_end = med_point-n_slice,med_point+n_slice+1
                    for i in range(idx_start,idx_end):
                        image_name = image_name.replace('_%d'%med_point,'_%d'%i) #!!! bug
                        if not image_name.endswith('.jpg'):
                            image_name = image_name + '.jpg'
                        samples.append((image_name, self.class_to_idx[str(label)]))
                self.half3D = False
            else:
                samples = []
                for image_name,label,depth in zip(image_names, labels, dcm_depths):
                    med_point = (depth+1)//2
                    n_slice = select_n_slices(k,depth)
                    idx_start, idx_end = med_point-n_slice,med_point+n_slice+1
                    image_name = image_name.replace('_%d'%med_point,'_%d'%i)
                    if not image_name.endswith('.jpg'):
                        image_name = image_name + '.jpg'
                    samples.append(([image_name for i in range(idx_start,idx_end)], self.class_to_idx[str(label)]))
                self.half3D = True
        else:
            samples = []
            for image_name, label in zip(image_names, labels):
                # Handle different file extensions
                if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
                    final_image_name = image_name
                elif image_name.endswith('.png'):
                    final_image_name = image_name
                elif image_name.endswith('.npy'):
                    final_image_name = image_name
                else:
                    # Default to .jpg if no extension specified
                    final_image_name = image_name + '.jpg'
                
                samples.append((final_image_name, self.class_to_idx[str(label)]))
            self.half3D = False
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.k = k
        if k>0 and k<1:
            self.max_slice = int(((k*25) // 2)*2 + 1)
        elif k>1:
            self.max_slice = int((k//2) * 2 + 1)
        else:
            self.max_slice = 1
        self.modality = modality
        if select_layers is None:
            self.select_idx = None
        else:
            self.select_idx = [Thickness_List.index(n) for n in select_layers if n in Thickness_List]
            print('Selected layers: ', select_layers)
            print('Selected idx: ', self.select_idx)
        self.th_heatmap = th_heatmap

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.half3D: #output multiple images
            img_name = [os.path.join(self.root_dir, each_name) for each_name in sample[0]]
            image = [self.loader(each_name) for each_name in img_name]
            image = [self.transfroms(each_image) for each_image in image]
            image_len = len(image)
            p3d = (0, 0, 0, 0 ,0 ,0 , 0, self.max_slice - image_len)
            image = torch.stack(image)
            image = torch.nn.functional.pad(image, p3d, mode='constant', value=0)
        else:
            img_name = os.path.join(self.root_dir, sample[0])
            if self.modality != 'Thickness':
                image = self.loader(img_name)
                image = self.transfroms(image)
            else:
                npy_data = np.load(img_name)
                image = np.sum(npy_data[self.select_idx], axis=0, keepdims=True) #C,H,W
                if self.th_heatmap:
                    # Normalize
                    normed = (image[0] - image.min()) / (image.max() - image.min())
                    # Apply colormap
                    cmap = plt.get_cmap("jet")
                    heatmap_rgba = cmap(normed)  # (H, W, 4)
                    image = (heatmap_rgba[..., :self.channel] * 255).astype(np.uint8).transpose(2, 0, 1)  # (H,W,C)->(C, H, W)
                else:
                    # Expand to N channels (repeat)
                    image = np.repeat(image, self.channel, axis=0)  # (C,H,W)
            # To tensor (C,H,W)
            print(image.shape)
            image = self.transfroms(image)
            image_len = 1

        label = int(sample[1])
        #debug visualization
        #print(image)
        return image, label, image_len

def build_dataset(is_train, args, k=0, img_dir = '/orange/bianjiang/tienyu/OCT_AD/all_images/',transform=None, modality='OCT', patient_ids=None, pid_key='patient_id', 
                  select_layers=None,th_resize=True,th_heatmap=False):
    if transform is None:
        transform = build_transform(is_train, args)
    
    if args.data_path.endswith('.csv'):
        dataset = CSV_Dataset(args.data_path, img_dir, is_train, transform, k, modality=modality, patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap)
    else:
        root = os.path.join(args.data_path, is_train)
        dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    #!!TODO: debug in 'train' not the first
    if not isinstance(is_train, list):
        is_train = [is_train]
    # train transform
    if 'train' in is_train:
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

def build_transform_public(is_train, args):
    tfms = transforms.Compose([
        transforms.Resize((380,380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tfms

class DistributedSamplerWrapper(torch.utils.data.distributed.DistributedSampler):
    def __init__(
        self,
        base_sampler,
        num_replicas = None,
        rank = None,
        seed = 0,
        shuffle = True,
        drop_last = False,
    ):
        self.base_sampler = base_sampler
        super().__init__(base_sampler, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

    def __iter__(self):
        base_indices = list(self.base_sampler.__iter__())
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        indices = [base_indices[i] for i in indices]
            
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

class TransformWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        inputs = self.transform(images=x, return_tensors="pt")
        return inputs["pixel_values"][0]    