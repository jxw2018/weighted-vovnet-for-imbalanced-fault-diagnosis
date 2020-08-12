# -*- coding: utf-8 -*-
"""
Created on Sun May  3 16:55:57 2020

@author: Administrator
"""

import  torch, os
import  numpy as np
from    few_shot_dataset import few_shot_dataset
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import meta
from pre_train_CNN import pre_train, train_test
import scipy.io as sio
from load_imbalanced_data import imbalanced_data, creat_sample_weight, create_class_weight
import deepdish.io as ddio
from sklearn.model_selection import train_test_split
import xlwt
from dataprocess import dataprocess

def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)
    
    config_pre = [
        ('conv1d', [16, args.input_channel, 25, 1, 12, 0]),
        ('bn', [16]),
        ('relu', [True]),
        ('max_pool1d', [2, 0]),
        ('conv1d', [16, 16, 25, 1, 12, 0]),
        ('bn', [16]),
        ('relu', [True]),
        ('max_pool1d', [2, 0]),
        ('conv1d', [16, 16, 25, 1, 12, 0]),
        ('bn', [16]),
        ('relu', [True]),
        ('max_pool1d', [2, 0]),
        ('dropout', [True]),
        ('conv1d', [16, 16, 25, 1, 12, 0]),
        ('bn', [16]),
        ('relu', [True]),
        ('max_pool1d', [2, 0]),
        ('dropout', [True]),
        ('flatten', []),
        ('linear', [100, 1792]),
        ('relu', [True]),
        ('linear', [args.pre_class, 100]),
        ('softmax', [True])
    ]
    
    x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight = dataprocess(args)
    
#    pre_model = pre_train(args, config_pre)
#    tmp = filter(lambda x: x.requires_grad, pre_model.parameters())
#    num = sum(map(lambda x: np.prod(x.shape), tmp))
#    print(pre_model)
#    print('Total trainable tensors:', num)
#    
#    pre_train_acc, pre_test_acc, pre_time_train = pre_model(x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight)
    
    train_acc = np.zeros(12)
    test_acc = np.zeros(12)
    time_train = np.zeros(12)    
    for i in range(1):
        pre_model = pre_train(args, config_pre)
        tmp = filter(lambda x: x.requires_grad, pre_model.parameters())
        num = sum(map(lambda x: np.prod(x.shape), tmp))
        print(pre_model)
        print('Total trainable tensors:', num)
        
#        train_acc[i], test_acc[i], time_train[i] = pre_model(x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight)
        train_acc[i], test_acc[i], time_train[i], test_confusion, f_score, g_mean = pre_model(x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight)
    train_acc[10] = np.mean(train_acc[:10])
    train_acc[11] = np.std(train_acc[:10])
    test_acc[10] = np.mean(test_acc[:10])
    test_acc[11] = np.std(test_acc[:10])
    time_train[10] = np.mean(time_train[:10])
    time_train[11] = np.std(time_train[:10])
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('CNN_matrix',cell_overwrite_ok=True)   
    for i in range(11):
        for j in range(11):
            sheet.write(i, j, str(test_confusion[i][j]))
    sheet.write(12, 1, f_score)
    sheet.write(13, 1, g_mean)
    book.save('CNN_matrix.xls')
                

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--pre_class', type=int, help='n class', default=11)
    argparser.add_argument('--pre_epoch', type=int, help='pre train epoch number', default=50)
    argparser.add_argument('--pre_batch_size', type=int, help='pre train batch size', default=32)
    argparser.add_argument('--pre_lr', type=float, help='pre train learning rate', default=1e-3)
    argparser.add_argument('--radio', type=float, help='train test split', default=0.8)
    argparser.add_argument('--se_radio', type=float, help='se radio', default=2)
    argparser.add_argument('--training', type=bool, help='dropout train', default= True)
    argparser.add_argument('--sensor_num', type=int, help='sensor num', default= 4)
    argparser.add_argument('--input_channel', type=int, help='input channel', default= 4)

    args = argparser.parse_args()

    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    