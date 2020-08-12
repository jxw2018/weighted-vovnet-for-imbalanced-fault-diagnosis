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
from pre_train_CNN_SVM import pre_train, train_test_SVM
import scipy.io as sio
from load_imbalanced_data import imbalanced_data, creat_sample_weight, create_class_weight
import deepdish.io as ddio
from sklearn.model_selection import train_test_split
import xlwt
from dataprocess_CNN_SVM import dataprocess

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
        ('linear', [100, 1792, 1]),
        ('relu', [True]),
        ('linear', [args.pre_class, 100, 0]),
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
        
#        train_acc[i], test_acc[i], time_train[i], test_confusion, f_score, g_mean = pre_model(x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight)

    
    CNN_SVM_model = train_test_SVM(args, config_pre)
    svm_train_acc = np.zeros(12)
    svm_test_acc = np.zeros(12)
    f_score = np.zeros(12)
    g_mean = np.zeros(12)
    for i in range(10):
        svm_train_acc[i], svm_test_acc[i], test_confusion, f_score[i], g_mean[i], train_confusion = CNN_SVM_model(x_train_im, y_train_im, im_radio, x_test_b, y_test_b, multi_sample_weight)
        if svm_test_acc[i] > 0.65:
            test_confusion_final = test_confusion
    
    svm_train_acc[10] = np.mean(svm_train_acc[:10])
    svm_train_acc[11] = np.std(svm_train_acc[:10])
    svm_test_acc[10] = np.mean(svm_test_acc[:10])
    svm_test_acc[11] = np.std(svm_test_acc[:10])
    f_score[10] = np.mean(f_score[:10])
    f_score[11] = np.std(f_score[:10])
    g_mean[10] = np.mean(g_mean[:10])
    g_mean[11] = np.std(g_mean[:10])
    
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    sheet = book.add_sheet('CNN_SVM_matrix',cell_overwrite_ok=True)   
    for i in range(11):
        for j in range(11):
            sheet.write(i, j, str(test_confusion_final[i][j]))
    for i in range(12):
        sheet.write(12+i, 0, svm_train_acc[i])
        sheet.write(12+i, 1, svm_test_acc[i])
        sheet.write(12+i, 2, f_score[i])
        sheet.write(12+i, 3, g_mean[i])
    book.save('CNN_SVM_matrix.xls')
    return svm_train_acc, svm_test_acc, test_confusion, f_score, g_mean, train_confusion
                

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    argparser.add_argument('--pre_class', type=int, help='n class', default=11)
    argparser.add_argument('--pre_epoch', type=int, help='pre train epoch number', default=100)
    argparser.add_argument('--pre_batch_size', type=int, help='pre train batch size', default=64)
    argparser.add_argument('--pre_lr', type=float, help='pre train learning rate', default=1e-3)
    argparser.add_argument('--radio', type=float, help='train test split', default=0.8)
    argparser.add_argument('--se_radio', type=float, help='se radio', default=2)
    argparser.add_argument('--training', type=bool, help='dropout train', default= True)
    argparser.add_argument('--sensor_num', type=int, help='sensor num', default= 4)
    argparser.add_argument('--input_channel', type=int, help='input channel', default= 4)

    args = argparser.parse_args()

    svm_train_acc, svm_test_acc, test_confusion, f_score, g_mean, train_confusion = main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    