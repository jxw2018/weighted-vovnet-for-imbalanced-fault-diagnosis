import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import time
import re, os

from    learn_model import learn_model
from    copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, outputs, labels, weight):  # 定义前向的函数运算即可
        return torch.mean(-torch.sum(weight*labels*torch.log(outputs+1e-10), 1))


class pre_train(nn.Module):
    def __init__(self, args, config):
        super(pre_train, self).__init__()

        self.pre_epoch = args.pre_epoch
        self.pre_batch_size = args.pre_batch_size
        self.pre_lr = args.pre_lr
        self.pre_class = args.pre_class
#        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.device = torch.device('cuda')

        self.net = learn_model(config).to(self.device)
        self.pre_optim = optim.Adam(self.net.parameters(), lr=self.pre_lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_train_im, y_train_im, imbalanced_dict_1, final_data_test, final_label_test, multi_sample_weight):
        x_train = torch.Tensor(x_train_im)
        y_train = torch.Tensor(y_train_im)
        x_test = torch.Tensor(final_data_test)
        y_test = torch.Tensor(final_label_test)
        multi_sample_weight = torch.Tensor(multi_sample_weight)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train, multi_sample_weight)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.pre_batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(y_test.shape[0]/1), shuffle=False)
        
#        torch.cuda.synchronize()
        start = time.time()
        for i in range(self.pre_epoch):
            print('\nEpoch: %d'%(i+1))
            self.net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0

            for i, data in enumerate(trainloader, 0):
                length = len(trainloader)
                inputs, labels, sample_weight = data
                inputs, labels, sample_weight = inputs.to(self.device), labels.to(self.device), sample_weight.to(self.device)
                self.pre_optim.zero_grad()

                outputs = self.net(inputs, dropout_training=True)
                labels = labels.long()
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.pre_optim.step()

                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.squeeze()).cpu().sum()
                
            train_acc = correct.item() / total
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                      % (i + 1, (i + 1 + i * length), sum_loss / (i + 1), 100. * correct.item() / total))
        
            print('testing!')
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                for data in testloader:
                    self.net.eval()
                    x_test, y_test = data
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                    output_test = self.net(x_test, dropout_training=False)
                    _, predicted = torch.max(output_test.data, 1)
                    total += y_test.size(0)
                    # correct += (predicted == y_test.squeeze()).sum()
                    correct += predicted.eq(y_test.long().squeeze()).cpu().sum()
                    
                    #confusion_matrix
                    y_test_confu = y_test.cpu().numpy().astype(np.int).squeeze()
                    predicted_confu = predicted.cpu().numpy()
                    f_score = f1_score(y_test_confu, predicted_confu, average='macro')
                    test_confusion = confusion_matrix(y_test_confu, predicted_confu)
                    accr_confusion = self.accuracy(test_confusion, y_test_confu, num_classes=self.pre_class)
                    g_mean = np.power(self.accr_confusion_multiply(accr_confusion, num_classes=self.pre_class), 1/self.pre_class)
                
                test_acc = correct.item() / total
                print('测试分类准确率为：%.3f%%' % (100 * correct.item() / total))
            if (100 * correct.item() / total) > 60:
                break
        
#        torch.cuda.synchronize()
        end = time.time()
        time_train = end-start
        print(time_train)
        torch.save(self.net.state_dict(), 'E:\DL\JiangXinwei\python\weighted vovnet with self attention\pre_model_CNN.pth')
        return train_acc, test_acc, time_train, test_confusion, f_score, g_mean
    
    def accuracy(self, confusion_matrix, true_labels, num_classes):
        list_data = self.count_nums(true_labels, num_classes)
     
        initial_value = 0
        list_length = num_classes
        true_pred = [ initial_value for i in range(list_length)]
        for i in range(0,num_classes-1):
            true_pred[i] = confusion_matrix[i][i]
    
        acc = []
        for i in range(0,num_classes-1):
            acc.append(0)
     
        for i in range(0,num_classes-1):
            acc[i] = true_pred[i] / list_data[i]
     
        return acc
    
    def count_nums(self, true_labels, num_classes):
            initial_value = 0
            list_length = num_classes
            list_data = [ initial_value for i in range(list_length)]
            list_data = np.bincount(true_labels)
            return list_data   
    
    def accr_confusion_multiply(self, accr_confusion,num_classes):
        accr_confusion_multiply = 1
        for i in range(0,num_classes-1):
            accr_confusion_multiply=accr_confusion_multiply*accr_confusion[i]
        return accr_confusion_multiply
        
        
class train_test(nn.Module):
    def __init__(self, args, config):
        super(train_test, self).__init__()

        self.pre_batch_size = args.pre_batch_size
        self.pre_lr = args.pre_lr
#        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.device = torch.device('cuda')

        self.net = learn_model(config).to(self.device)
        self.trained_dict = torch.load("E:\DL\JiangXinwei\python\weighted vovnet with self attention\pre_model.pth")
        self.net.load_state_dict(self.trained_dict)
        
    def forward(self, x_train_im, y_train_im, imbalanced_dict_1, final_data_test, final_label_test, multi_sample_weight):
        # imbalanced_dict_1 = torch.Tensor(imbalanced_dict_1).to(self.device)
        # self.pre_optim = optim.Adam(self.net.parameters(), lr=self.pre_lr)
        # self.criterion = My_loss()
        x_train = torch.Tensor(x_train_im)
        y_train = torch.Tensor(y_train_im)
        x_test = torch.Tensor(final_data_test)
        y_test = torch.Tensor(final_label_test)
        multi_sample_weight = torch.Tensor(multi_sample_weight)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train, multi_sample_weight)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.pre_batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=int(y_test.shape[0]/1), shuffle=False)
        
        print('training!')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in trainloader:
                self.net.eval()
                x_train, y_train, sample_weight = data
                x_train, y_train = x_train.to(self.device), y_train.to(self.device)
                output_train = self.net(x_train, dropout_training=False)
                _, predicted = torch.max(output_train.data, 1)
                total += y_train.size(0)
                # correct += (predicted == y_test.squeeze()).sum()
                correct += predicted.eq(y_train.long().squeeze()).cpu().sum()
            
            train_acc = correct.item() / total
            print('训练分类准确率为：%.3f%%' % (100 * correct.item() / total))
        
        print('testing!')
        with torch.no_grad():
            correct = 0.0
            total = 0.0
            for data in testloader:
                self.net.eval()
                x_test, y_test = data
                x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                output_test = self.net(x_test, dropout_training=False)
                _, predicted = torch.max(output_test.data, 1)
                total += y_test.size(0)
                # correct += (predicted == y_test.squeeze()).sum()
                correct += predicted.eq(y_test.long().squeeze()).cpu().sum()
            
            test_acc = correct.item() / total
            print('测试分类准确率为：%.3f%%' % (100 * correct.item() / total))
        return train_acc, test_acc