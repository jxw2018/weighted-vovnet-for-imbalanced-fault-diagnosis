# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:01:44 2020

@author: DL001
"""
import  os
import  numpy as np
from    few_shot_dataset import few_shot_dataset
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  argparse
from meta import meta
from pre_train import pre_train
import scipy.io as sio
from load_imbalanced_data import imbalanced_data, creat_sample_weight, create_class_weight
from sklearn.model_selection import train_test_split


def dataprocess(args):
    data = sio.loadmat(r'F:\科研资料\实验验证\DL\Compound_dataset\Imb_dataset\dds_compound_1800_unnorm.mat')
    final_data = data['final_data']
    final_label = data['final_label']
    
    x_train_b, x_test_b, y_train_b, y_test_b = train_test_split(final_data, final_label, test_size=0.1)
    
    train_x = np.zeros((1, 4, 1800))
    train_y = np.zeros((1,1))
    test_x = np.zeros((1, 4, 1800))
    test_y = np.zeros((1,1))
    for cls in range(11):
        train_data_size = int(final_label.shape[0]/args.pre_class)
        selected_data = final_data[train_data_size * cls:train_data_size * (cls + 1), :, :]
        selected_label = final_label[train_data_size * cls:train_data_size * (cls + 1), :]
        selected_data_idx = np.random.choice(len(selected_data), len(selected_data), False)
        np.random.shuffle(selected_data_idx)
        indexDtrain = np.array(selected_data_idx[:int(len(selected_data)*0.8)])
        indexDtest = np.array(selected_data_idx[int(len(selected_data)*0.8):])
        train_x = np.concatenate((np.array(selected_data)[indexDtrain], train_x), axis=0)
        train_y = np.concatenate((np.array(selected_label)[indexDtrain], train_y), axis=0)
        test_x = np.concatenate((np.array(selected_data)[indexDtest], test_x), axis=0)
        test_y = np.concatenate((np.array(selected_label)[indexDtest], test_y), axis=0)
    train_x = np.delete(train_x, -1, axis=0)
    train_y = np.delete(train_y, -1, axis=0)
    test_x = np.delete(test_x, -1, axis=0)
    test_y = np.delete(test_y, -1, axis=0)
    
    if args.sensor_num != 4:
        train_x = train_x[:,args.sensor_num:args.sensor_num+1,:]
        test_x = test_x[:,args.sensor_num:args.sensor_num+1,:]
    
    imbalanced_dict = {0: 50, 1: 15, 2: 15, 3: 15, 4: 15, 5: 5, 6: 5, 7: 5, 8: 5, 9: 2, 10: 2}
#    imbalanced_dict = {0: 50, 1: 15, 2: 15, 3: 15, 4: 15, 5: 3, 6: 3, 7: 3, 8: 3, 9: 1, 10: 1}
    x_train_im, y_train_im, x_test_im, y_test_im, imbalanced_dict_1 = imbalanced_data(train_x, train_y, imbalanced_dict, args.radio,
                                                                                      refresh=False, seed=1)
    multi_class_weight = create_class_weight(imbalanced_dict_1)
    multi_sample_weight, ir_overall = creat_sample_weight(imbalanced_dict_1, multi_class_weight, args.radio)
    im_radio = np.zeros(args.pre_class)
    for k, v in multi_class_weight.items():
        im_radio[k] = v
#    sio.savemat('E:\DL\JiangXinwei\python\weighted vovnet with self attention\\train_test_data.mat', {'x_train_im': x_train_im, 'y_train_im': y_train_im, 'im_radio': im_radio, 'x_test_b': x_test_b, 'y_test_b': y_test_b, 'multi_sample_weight': multi_sample_weight})    
    return x_train_im, y_train_im, multi_class_weight, x_test_b, y_test_b, multi_sample_weight