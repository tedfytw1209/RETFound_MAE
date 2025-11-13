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
from pathlib import Path
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

def masking_image(image, mask_slice):
    binary_mask = np.zeros_like(image, dtype=np.uint8)
    for i in range(mask_slice.shape[0]-1):
        upper = mask_slice[i].astype(int)
        lower = mask_slice[i+1].astype(int)
        for x in range(image.shape[1]):
            binary_mask[upper[x]:lower[x], x] = 1

    # 套用 mask (把 mask=0 的地方設為 0)
    masked_img = image.copy()
    masked_img[binary_mask == 0] = 0
    return masked_img


def _fill_nan_linear_1d(row: np.ndarray) -> np.ndarray:
    row = row.astype(np.float32, copy=True)
    nans = np.isnan(row)
    if not nans.any():
        return row
    valid = ~nans
    if not valid.any():
        row[:] = 0.0
        return row
    x = np.arange(row.shape[0], dtype=np.float32)
    row[nans] = np.interp(x[nans], x[valid], row[valid])
    return row

def _resample_width(mask_slice: np.ndarray, W: int) -> np.ndarray:
    L, Wm = mask_slice.shape
    if Wm == W:
        out = np.empty_like(mask_slice, dtype=np.float32)
        for i in range(L):
            out[i] = _fill_nan_linear_1d(mask_slice[i])
        return out

    x_src = np.linspace(0.0, 1.0, Wm, dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, W,  dtype=np.float32)
    out = np.empty((L, W), dtype=np.float32)
    for i in range(L):
        row = _fill_nan_linear_1d(mask_slice[i])
        out[i] = np.interp(x_dst, x_src, row.astype(np.float32))
    return out

def _build_binary_mask(mask_slice: np.ndarray, image_size, add_bound=0):
    W, H = image_size
    if mask_slice is None:
        return None
    mask_slice = np.asarray(mask_slice)
    if mask_slice.ndim != 2:
        return None
    L, Wm = mask_slice.shape
    if L == 0 or H <= 0 or W <= 0:
        return None

    mask_slice = _resample_width(mask_slice, W)  # (L, W)
    y = np.rint(mask_slice).astype(np.int32, copy=False)
    y = np.clip(y, 0, H - 1)

    binary_mask = np.zeros((H, W), dtype=np.uint8)
    if L == 1:
        rr = y[0]
        binary_mask[rr, np.arange(W)] = 255
        return Image.fromarray(binary_mask, mode="L")

    rows = np.arange(H, dtype=np.int32)[:, None]   # (H,1)
    for i in range(L - 1):
        if i==0:
            upper = np.clip(y[i] - add_bound, 0, H - 1)
        else:
            upper = y[i]
        if i==L-2:
            lower = np.clip(y[i + 1] + add_bound, 0, H - 1)
        else:
            lower = y[i + 1]
        ul = np.minimum(upper, lower)[None, :]
        ll = np.maximum(upper, lower)[None, :]
        band = (rows >= ul) & (rows < ll)
        eq = (rows == ul) & (ul == ll)
        binary_mask[band | eq] = 255

    return Image.fromarray(binary_mask, mode="L")

def masking_image_pil(image, mask_slice, fill_color=(0, 0, 0), transform_binary_mask=True):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image_rgb = image.convert("RGB")
    max_size = max(image_rgb.size)
    add_bound = int(max_size * 0.025)
    if transform_binary_mask:
        mask_img = _build_binary_mask(mask_slice, image_rgb.size, add_bound=add_bound)
    else:
        mask_img = Image.fromarray(mask_slice, mode="L")
    if mask_img is None:
        return image_rgb, None
    bg = Image.new("RGB", image_rgb.size, fill_color)
    masked_img = Image.composite(image_rgb, bg, mask_img)
    return masked_img, mask_img

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
Thickness_DIR = "/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
Thickness_CSV = "/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"

class CSV_Dataset(Dataset):
    def __init__(self,csv_file,img_dir,is_train,transfroms=[],k=0,class_to_idx={}, modality='OCT', patient_ids=[], pid_key = 'patient_id', select_layers=None,th_resize=True,th_heatmap=False, use_ducan_preprocessing=False, thickness_dir=Thickness_DIR, add_mask=False, output_mask=False, mask_transforms=None, use_img_per_patient=False, CV=False):
        #common args
        self.transfroms = transfroms
        self.root_dir = img_dir
        self.loader = datasets.folder.default_loader
        self.use_ducan_preprocessing = use_ducan_preprocessing
        self.modality = modality
        self.pid_key = pid_key
        self.mask_transforms = mask_transforms
        # Initialize DuCAN preprocessor if needed
        if use_ducan_preprocessing:
            try:
                from preprocessing import DuCANPreprocessor
                self.ducan_preprocessor = DuCANPreprocessor(target_size=(224, 224))
            except ImportError:
                print("Warning: preprocessing module not found. Falling back to standard transforms.")
                self.use_ducan_preprocessing = False
        data = pd.read_csv(csv_file)
        if not isinstance(is_train, list):
            is_train_l = [is_train]
        else:
            is_train_l = is_train
        is_train = is_train_l[0]
        if CV and patient_ids: #for cross-validation with patient ids
            self.annotations = data[data[pid_key].isin(patient_ids)].reset_index(drop=True)
            self.annotations['split'] = is_train
        elif 'split' in data.columns:
            self.annotations = data[data['split'].isin(is_train_l)].reset_index(drop=True)
        else:
            self.annotations = data
        #for subgroup analysis
        if patient_ids:
            self.annotations = self.annotations[self.annotations[pid_key].isin(patient_ids)].reset_index(drop=True)
            print('After filtering with patient ids, data len: ', self.annotations.shape[0])
        print('Split: ', is_train_l,' Data len: ', self.annotations.shape[0])
        if use_img_per_patient and pid_key in self.annotations.columns:
            self.annotations = self.annotations.drop_duplicates(subset=[pid_key,'eye']).reset_index(drop=True)
            print('After random selection of two image per patient, data len: ', self.annotations.shape[0])
        self.classes = [str(c) for c in self.annotations['label'].unique()]
        self.num_class = len(self.classes)
        #add mask, filter out samples without mask
        self.add_mask = add_mask
        self.output_mask = output_mask
        self.dataset_type = 'UF' if "IRB2024_v5" in csv_file else 'Public'
        if self.add_mask or self.output_mask:
            #UF Dataset
            if "IRB2024_v5" in csv_file:
                masked_df = pd.read_csv(Thickness_CSV)
                masked_df = masked_df.rename(columns={'OCT':'folder'}).dropna(subset=['Surface Name'])
                self.annotations = self.annotations.merge(masked_df,on='folder',how='inner').reset_index(drop=True)
                print('After adding mask, data len: ', self.annotations.shape[0])
        self.thickness_dir = thickness_dir
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
        
        mask_names = None
        if modality == 'OCT' and 'OCT' in self.annotations.columns:
            image_names = self.annotations['OCT']
            print('OCT images: ', len(image_names))
            print('OCT image example: ', image_names.head())
            if self.add_mask or self.output_mask:
                mask_names = self.annotations['folder'] + '/' + self.annotations['Surface Name']
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
        if mask_names is None:
            mask_names = ['']*len(image_names)
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
            for image_name, label, mask_name in zip(image_names, labels, mask_names):
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
                
                samples.append((final_image_name, self.class_to_idx[str(label)], mask_name))
            self.half3D = False
        print(samples[0])
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
            select_idx = []
            for n in select_layers:
                if isinstance(n, int) and n>=0 and n<len(Thickness_List):
                    select_idx.append(n)
                elif isinstance(n, str) and n in Thickness_List:
                    select_idx.append(Thickness_List.index(n))
            if len(select_idx) == 0:
                select_idx = None
            self.select_idx = select_idx
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
            else:
                npy_data = np.load(img_name)
                image = np.sum(npy_data[self.select_idx], axis=0, keepdims=True) #C,H,W
                if self.th_heatmap:
                    # Normalize
                    normed = (image[0] - image.min()) / (image.max() - image.min())
                    # Apply colormap
                    cmap = plt.get_cmap("jet")
                    heatmap_rgba = cmap(normed)  # (H, W, 4)
                    heatmap_array = (heatmap_rgba[..., :self.channel] * 255).astype(np.uint8) # (H,W,C)
                    image = Image.fromarray(heatmap_array, mode='RGB')  # RGB
                else:
                    # Expand to N channels (repeat)
                    image = np.repeat(image, self.channel, axis=0).transpose(1,2,0)  # (C,H,W)->(H,W,C)
                    image = Image.fromarray(image, mode='RGB')  # RGB
            
            # Apply DuCAN preprocessing if enabled
            if self.use_ducan_preprocessing and hasattr(self, 'ducan_preprocessor'):
                # Convert PIL Image to numpy array for preprocessing
                if hasattr(image, 'mode'):  # PIL Image
                    image_np = np.array(image)
                else:
                    image_np = image
                
                # Apply modality-specific preprocessing
                if self.modality in ['CFP', 'Fundus']:
                    # Fundus preprocessing: ROI extraction, size standardization, CLAHE
                    preprocessed_np = self.ducan_preprocessor.preprocess_fundus(image_np)
                elif self.modality == 'OCT':
                    # OCT preprocessing: denoising, intensity normalization, size standardization
                    preprocessed_np = self.ducan_preprocessor.preprocess_oct(image_np)
                else:
                    # Default preprocessing
                    preprocessed_np = image_np
                # Convert back to PIL Image for transforms
                image = Image.fromarray(preprocessed_np)
            
            # mask processing
            output_mask = None
            if self.add_mask or self.output_mask:
                if self.dataset_type == 'UF':
                    mask_path = os.path.join(self.thickness_dir, sample[2])
                    mask = np.load(mask_path) # (Layer Interface, slice_num, W) 
                    slice_index = int(os.path.basename(img_name).split("_")[-1].split(".")[0])
                    mask_slice = mask[:, slice_index, :]
                    image_masked, output_mask = masking_image_pil(image.copy(), mask_slice)
                else:
                    image_name = Path(img_name)
                    mask_path = os.path.join(self.thickness_dir, image_name.stem + '.npy')
                    print('Loading mask from: ', mask_path)
                    mask = np.load(mask_path) # (H, W)
                    image_masked, output_mask = masking_image_pil(image.copy(), mask, transform_binary_mask=False)
                image = image_masked
            
            # (H,W,C)
            image = self.transfroms(image)
            image_len = 1

        label = int(sample[1])
        #debug visualization
        #print(image)
        return image, label, image_len

class CSV_Dataset_eval(CSV_Dataset):
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
            else:
                npy_data = np.load(img_name)
                image = np.sum(npy_data[self.select_idx], axis=0, keepdims=True) #C,H,W
                if self.th_heatmap:
                    # Normalize
                    normed = (image[0] - image.min()) / (image.max() - image.min())
                    # Apply colormap
                    cmap = plt.get_cmap("jet")
                    heatmap_rgba = cmap(normed)  # (H, W, 4)
                    heatmap_array = (heatmap_rgba[..., :self.channel] * 255).astype(np.uint8) # (H,W,C)
                    image = Image.fromarray(heatmap_array, mode='RGB')  # RGB
                else:
                    # Expand to N channels (repeat)
                    image = np.repeat(image, self.channel, axis=0).transpose(1,2,0)  # (C,H,W)->(H,W,C)
                    image = Image.fromarray(image, mode='RGB')  # RGB
            
            # Apply DuCAN preprocessing if enabled
            if self.use_ducan_preprocessing and hasattr(self, 'ducan_preprocessor'):
                # Convert PIL Image to numpy array for preprocessing
                if hasattr(image, 'mode'):  # PIL Image
                    image_np = np.array(image)
                else:
                    image_np = image
                
                # Apply modality-specific preprocessing
                if self.modality in ['CFP', 'Fundus']:
                    # Fundus preprocessing: ROI extraction, size standardization, CLAHE
                    preprocessed_np = self.ducan_preprocessor.preprocess_fundus(image_np)
                elif self.modality == 'OCT':
                    # OCT preprocessing: denoising, intensity normalization, size standardization
                    preprocessed_np = self.ducan_preprocessor.preprocess_oct(image_np)
                else:
                    # Default preprocessing
                    preprocessed_np = image_np
                # Convert back to PIL Image for transforms
                image = Image.fromarray(preprocessed_np)
            
            # mask processing
            output_mask = None
            if self.add_mask or self.output_mask:
                if self.dataset_type == 'UF':
                    mask_path = os.path.join(self.thickness_dir, sample[2])
                    mask = np.load(mask_path) # (Layer Interface, slice_num, W) 
                    slice_index = int(os.path.basename(img_name).split("_")[-1].split(".")[0])
                    mask_slice = mask[:, slice_index, :]
                    image_masked, output_mask = masking_image_pil(image.copy(), mask_slice)
                else:
                    image_name = Path(img_name)
                    mask_path = os.path.join(self.thickness_dir, image_name.name + '.npy')
                    print('Loading mask from: ', mask_path)
                    mask = np.load(mask_path) # (H, W)
                    image_masked, output_mask = masking_image_pil(image.copy(), mask, transform_binary_mask=False)
                if self.add_mask:
                    image = image_masked
            # (H,W,C)
            image = self.transfroms(image)
            if output_mask is not None:
                output_mask_tensor = self.mask_transforms(output_mask)
            else:
                output_mask_tensor = None
            image_len = 1
        #print(output_mask_tensor.shape,output_mask_tensor.min(), output_mask_tensor.max())
        label = int(sample[1])
        #output image name for evaluation
        return image, label, image_len, sample[0], output_mask_tensor

class DualCSV_Dataset(Dataset):
    def __init__(self,data_oct,data_cfp):
        self.data_oct = data_oct
        self.data_cfp = data_cfp
        #TODO:Merge to csv to match to modality
        self.annotations = self.data_oct.annotations
        self.classes = self.data_oct.classes
        self.num_class = self.data_oct.num_class
        self.class_to_idx = self.data_oct.class_to_idx
        self.channel = 3
        self.targets = self.data_oct.targets #!!! Assume the targets are the same
        assert len(data_oct) == len(data_cfp), "The number of OCT and CFP data must be the same"
    def __len__(self):
        return len(self.data_oct)
    def __getitem__(self, idx):
        sample_oct, label_oct, image_len_oct = self.data_oct[idx]
        sample_cfp, label_cfp, image_len_cfp = self.data_cfp[idx]
        assert label_oct == label_cfp, "The label of OCT and CFP must be the same"
        return sample_oct, sample_cfp, label_oct

def build_dataset(is_train, args, k=0, img_dir = '/orange/bianjiang/tienyu/OCT_AD/all_images/',transform=None, modality='OCT', patient_ids=None, pid_key='patient_id', select_layers=None,th_resize=True,th_heatmap=False, CV=False, eval_mode=False):
    if transform is None:
        transform = build_transform(is_train, args)
    print('Image transform: ', transform)
    mask_transforms = build_transform_mask(args)
    print('Mask transform: ', mask_transforms)
    #csv dataset
    if eval_mode:
        csv_func = CSV_Dataset_eval
    else:
        csv_func = CSV_Dataset
    
    output_mask = False if not hasattr(args, 'output_mask') else args.output_mask
    if 'dual_input_cnn'  in args.model: #Dual model special dataset
        img_dir_oct = "/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
        img_dir_cfp = "/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
        dataset_oct = csv_func(args.data_path, img_dir_oct, is_train, transform, k, modality="Thickness", patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap, output_mask=output_mask, use_img_per_patient=args.use_img_per_patient, CV=CV)
        dataset_cfp = csv_func(args.data_path, img_dir_cfp, is_train, transform, k, modality="CFP", patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap, use_img_per_patient=args.use_img_per_patient, CV=CV)
        dataset = DualCSV_Dataset(dataset_oct, dataset_cfp)
    elif 'ducan' in args.model: #DuCAN dual-modal dataset
        # DuCAN requires both fundus and OCT images with specialized preprocessing
        dataset_oct = csv_func(args.data_path, img_dir, is_train, transform, k, modality="OCT", patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap, use_ducan_preprocessing=True,add_mask=args.add_mask, output_mask=output_mask, mask_transforms=mask_transforms, use_img_per_patient=args.use_img_per_patient, CV=CV)
        dataset_fundus = csv_func(args.data_path, img_dir, is_train, transform, k, modality="CFP", patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap, use_ducan_preprocessing=True, use_img_per_patient=args.use_img_per_patient, CV=CV)
        dataset = DualCSV_Dataset(dataset_fundus, dataset_oct)  # Note: fundus first, OCT second for DuCAN
    elif args.data_path.endswith('.csv'):
        dataset = csv_func(args.data_path, img_dir, is_train, transform, k, modality=modality, patient_ids=patient_ids, pid_key=pid_key, select_layers=select_layers, th_resize=th_resize, th_heatmap=th_heatmap, thickness_dir=args.thickness_dir, add_mask=args.add_mask, output_mask=output_mask, mask_transforms=mask_transforms, use_img_per_patient=args.use_img_per_patient, CV=CV)
    else:
        root = os.path.join(args.data_path, is_train)
        dataset = datasets.ImageFolder(root, transform=transform)

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if not isinstance(is_train, list):
        is_train = [is_train]
    # train transform
    if 'train' in is_train and not args.train_no_aug:
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

def build_transform_mask(args):
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