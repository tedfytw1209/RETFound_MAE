import os
import numpy as np
import pandas as pd

CellData_path = '/orange/ruogu.fang/tienyuchang/CellData/OCT'

CNV_train = os.listdir(os.path.join(CellData_path, 'train/CNV'))
CNV_train.sort()
DME_train = os.listdir(os.path.join(CellData_path, 'train/DME'))
DME_train.sort()
DRUSEN_train = os.listdir(os.path.join(CellData_path, 'train/DRUSEN'))
DRUSEN_train.sort()
NORMAL_train = os.listdir(os.path.join(CellData_path, 'train/NORMAL'))
NORMAL_train.sort()

CNV_test = os.listdir(os.path.join(CellData_path, 'test/CNV'))
CNV_test.sort()
DME_test = os.listdir(os.path.join(CellData_path, 'test/DME'))
DME_test.sort()
DRUSEN_test = os.listdir(os.path.join(CellData_path, 'test/DRUSEN'))
DRUSEN_test.sort()
NORMAL_test = os.listdir(os.path.join(CellData_path, 'test/NORMAL'))
NORMAL_test.sort()

train_CNV_path = [os.path.join(CellData_path, 'train/CNV', i) for i in CNV_train] + [os.path.join(CellData_path, 'train/NORMAL', i) for i in NORMAL_train]
train_CNV_label = [1]*len(CNV_train) + [0]*len(NORMAL_train)
test_CNV_path = [os.path.join(CellData_path, 'test/CNV', i) for i in CNV_test] + [os.path.join(CellData_path, 'test/NORMAL', i) for i in NORMAL_test]
test_CNV_label = [1]*len(CNV_test) + [0]*len(NORMAL_test)

train_DME_path = [os.path.join(CellData_path, 'train/DME', i) for i in DME_train] + [os.path.join(CellData_path, 'train/NORMAL', i) for i in NORMAL_train]
train_DME_label = [1]*len(DME_train) + [0]*len(NORMAL_train)
test_DME_path = [os.path.join(CellData_path, 'test/DME', i) for i in DME_test] + [os.path.join(CellData_path, 'test/NORMAL', i) for i in NORMAL_test]
test_DME_label = [1]*len(DME_test) + [0]*len(NORMAL_test)

train_DRUSEN_path = [os.path.join(CellData_path, 'train/DRUSEN', i) for i in DRUSEN_train] + [os.path.join(CellData_path, 'train/NORMAL', i) for i in NORMAL_train]
train_DRUSEN_label = [1]*len(DRUSEN_train) + [0]*len(NORMAL_train)
test_DRUSEN_path = [os.path.join(CellData_path, 'test/DRUSEN', i) for i in DRUSEN_test] + [os.path.join(CellData_path, 'test/NORMAL', i) for i in NORMAL_test]
test_DRUSEN_label = [1]*len(DRUSEN_test) + [0]*len(NORMAL_test)

def create_csv(image, label, name):
    data = {
        'image': image,
        'label': label
    }
    df = pd.DataFrame(data)
    df.to_csv(name, index=False)

create_csv(train_CNV_path, train_CNV_label, os.path.join(CellData_path, 'CNV_train.csv'))
create_csv(test_CNV_path, test_CNV_label, os.path.join(CellData_path, 'CNV_test.csv'))
create_csv(train_DME_path, train_DME_label, os.path.join(CellData_path, 'DME_train.csv'))
create_csv(test_DME_path, test_DME_label, os.path.join(CellData_path, 'DME_test.csv'))
create_csv(train_DRUSEN_path, train_DRUSEN_label, os.path.join(CellData_path, 'DRUSEN_train.csv'))
create_csv(test_DRUSEN_path, test_DRUSEN_label, os.path.join(CellData_path, 'DRUSEN_test.csv'))