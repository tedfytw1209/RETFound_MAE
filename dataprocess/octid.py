import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OCTID_path = '/orange/ruogu.fang/tienyuchang/OCTID'

csr = os.listdir(f'{OCTID_path}/CSR')
dr = os.listdir(f'{OCTID_path}/DR')
mh = os.listdir(f'{OCTID_path}/MH')
normal = os.listdir(f'{OCTID_path}/NORMAL')
dr = [i for i in dr if 'jpeg' in i]
csr.sort()
dr.sort()
mh.sort()
normal.sort()

train_csr, test_csr = train_test_split(csr, test_size=0.2, random_state=42)
train_dr, test_dr = train_test_split(dr, test_size=0.2, random_state=42)
train_mh, test_mh = train_test_split(mh, test_size=0.2, random_state=42)
train_normal, test_normal = train_test_split(normal, test_size=0.2, random_state=42)

train_csr_path = [f'{OCTID_path}/CSR/'+i for i in train_csr] + [f'{OCTID_path}/NORMAL/'+i for i in train_normal]
train_csr_label = [1]*len(train_csr) + [0]*len(train_normal)
test_csr_path = [f'{OCTID_path}/CSR/'+i for i in test_csr] + [f'{OCTID_path}/NORMAL/'+i for i in test_normal]
test_csr_label = [1]*len(test_csr) + [0]*len(test_normal)

train_dr_path = [f'{OCTID_path}/DR/'+i for i in train_dr] + [f'{OCTID_path}/NORMAL/'+i for i in train_normal]
train_dr_label = [1]*len(train_dr) + [0]*len(train_normal)
test_dr_path = [f'{OCTID_path}/DR/'+i for i in test_dr] + [f'{OCTID_path}/NORMAL/'+i for i in test_normal]
test_dr_label = [1]*len(test_dr) + [0]*len(test_normal)

train_mh_path = [f'{OCTID_path}/MH/'+i for i in train_mh] + [f'{OCTID_path}/NORMAL/'+i for i in train_normal]
train_mh_label = [1]*len(train_mh) + [0]*len(train_normal)
test_mh_path = [f'{OCTID_path}/MH/'+i for i in test_mh] + [f'{OCTID_path}/NORMAL/'+i for i in test_normal]
test_mh_label = [1]*len(test_mh) + [0]*len(test_normal)

def create_csv(image, label, name):
    data = {
        'image': image,
        'label': label
    }
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)
    
create_csv(train_csr_path, train_csr_label, f'{OCTID_path}/csr_train.csv')
create_csv(test_csr_path, test_csr_label, f'{OCTID_path}/csr_test.csv')
create_csv(train_dr_path, train_dr_label, f'{OCTID_path}/dr_train.csv')
create_csv(test_dr_path, test_dr_label, f'{OCTID_path}/dr_test.csv')
create_csv(train_mh_path, train_mh_label, f'{OCTID_path}/mh_train.csv')
create_csv(test_mh_path, test_mh_label, f'{OCTID_path}/mh_test.csv')