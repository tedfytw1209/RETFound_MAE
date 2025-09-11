import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DIR_path = '/orange/ruogu.fang/tienyuchang/OCTDL'

amd = os.listdir(DIR_path + '/AMD')
dme = os.listdir(DIR_path + '/DME')
no = os.listdir(DIR_path + '/NO')
amd.sort()
dme.sort()
no.sort()

train_amd, test_amd = train_test_split(amd, test_size=0.2, random_state=42)
train_dme, test_dme = train_test_split(dme, test_size=0.2, random_state=42)
train_no, test_no = train_test_split(no, test_size=0.2, random_state=42)

train_amd_path = [DIR_path + '/AMD/'+i for i in train_amd] + [DIR_path + '/NO/'+i for i in train_no]
train_amd_label = [1]*len(train_amd) + [0]*len(train_no)
test_amd_path = [DIR_path + '/AMD/'+i for i in test_amd] + [DIR_path + '/NO/'+i for i in test_no]
test_amd_label = [1]*len(test_amd) + [0]*len(test_no)

train_dme_path = [DIR_path + '/DME/'+i for i in train_dme] + [DIR_path + '/NO/'+i for i in train_no]
train_dme_label = [1]*len(train_dme) + [0]*len(train_no)
test_dme_path = [DIR_path + '/DME/'+i for i in test_dme] + [DIR_path + '/NO/'+i for i in test_no]
test_dme_label = [1]*len(test_dme) + [0]*len(test_no)

def create_csv(image, label, name):
    data = {
        'image': image,
        'label': label
    }
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)

create_csv(train_amd_path, train_amd_label, DIR_path + '/AMD_train.csv')
create_csv(test_amd_path, test_amd_label, DIR_path + '/AMD_test.csv')
create_csv(train_dme_path, train_dme_label, DIR_path + '/DME_train.csv')
create_csv(test_dme_path, test_dme_label, DIR_path + '/DME_test.csv')
