from IPython import display

import torch
import sys
import pdb
import time
import numpy as np
import pandas as pd
import copy
import random
import os
import csv
import itertools
import functools
import torch
import argparse
import errno
import foolbox as fb

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.nn import init
from torch.nn import Parameter as P
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import ImageFolder
#import torchvision.utils as vutils
from torchvision.utils import save_image
#from torch.optim.lr_scheduler import StepLR, MultiStepLR
from matplotlib import pyplot
from pandas import DataFrame
from numpy import genfromtxt

from sklearn.datasets import make_circles, make_moons

from utils_sy_0527_2 import *
#from utils_else_sy_0527 import *
#from constants import *


class Strategy:
    def __init__(self,strategy_type,data_loader,idxs_lb,net,args):
        self.strategy_type = strategy_type
        self.n_epoch = args.active_num_epoch
        self.num_workers  = data_loader.num_workers
        self.batch_size = data_loader.batch_size
        self.X = data_loader.dataset.tensors[0]
        self.Y = data_loader.dataset.tensors[1]
        self.idxs_lb = idxs_lb
        self.net = net
        self.n_pool = len(self.Y)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.gan = args.gan
        self.lr = args.lr
        self.wd = args.wd
        self.args = args


    def query(self,n):
        if self.strategy_type == 'RandomSampling':
            temp = np.random.choice(np.where(self.idxs_lb==0)[0], 2*n)
            temp = np.unique(temp)
            return temp[0:n]

        elif self.strategy_type == 'GANgeneration':
            temp = np.random.choice(np.where(self.idxs_lb==0)[0], 2*n)
            temp = np.unique(temp)
            return temp[0:n]
        
        elif self.strategy_type == 'GANdistance':
            idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
            unlabelX = self.X[idxs_unlabeled]
            unlabelY = self.Y[idxs_unlabeled]
            data_ = self.X
            label_= self.Y
            query_metric = np.empty((1,len(self.X)))
            newY = np.empty((1,len(self.X)))
            for i in range(len(unlabelY)):
                _,_,dis1 = get_proj_distance_square(unlabelX[i], 0, data_,label_,self.args)
                _,_,dis2 = get_proj_distance_square(unlabelX[i], 1, data_,label_,self.args)

                print('distance calculating, with unlabeled set {} / {}'.format(i+1,len(unlabelY)))
                query_metric[0,i] = np.absolute(dis1.detach().cpu()-dis2.detach().cpu())
                newY[0,i] = 0 if dis1<dis2 else 1

            query_index = np.unique(np.ceil((np.argsort(query_metric)-1)/2))
            query_index = query_index.astype(int)
            #unlabelY[query_index[0:n]]=newY[query_index[0:n]]
            self.Y[idxs_unlabeled[query_index[0:n]]] = torch.LongTensor(newY[0,query_index[0:n]])

            return idxs_unlabeled[query_index[0:n]]

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, loader_tr, optimizer):
        self.clf.train()
        criterion = self.args.criterion
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x = x.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            out = self.clf.forward(x.cuda().float())  
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    def train(self):
        self.clf = self.net.to(self.device)
        optimizer = optim.Adam(self.clf.parameters(), lr=self.lr, weight_decay=self.wd)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = torch.utils.data.DataLoader(TensorDataset(self.X[idxs_train], self.Y[idxs_train],torch.Tensor(idxs_train)), batch_size=self.batch_size,
                            shuffle=True, num_workers=self.num_workers)

        for epoch in range(1, self.n_epoch+1):
            self._train(loader_tr, optimizer)

        self.net = self.clf
        #return self.clf

    def predict(self,test_loader):
        self.clf=self.net
        self.clf.eval()
        # = torch.zeros(len(test_loader), dtype=Y.dtype)
        acc = 0
        with torch.no_grad():
            for batch_idx, (x,y) in enumerate(test_loader):
                x, y = x.to(self.device), y.to(self.device)
                output = self.clf.forward(x.cuda().float())
                _, pred = torch.max(output,1)
                #P[batch_idx.type(torch.LongTensor)] = pred.cpu()
                acc += torch.sum(pred==y.long()).sum().item()/len(y)
            acc = acc / len(test_loader)
        return acc
        
class simple_Ndim_Net(torch.nn.Module):
    def __init__(self, args):
        super(simple_Ndim_Net, self).__init__()
        
        self.num_layer = args.num_layer
        self.data_dim = args.data_dim
        self.num_class = args.num_class
        
        self.fc1 = torch.nn.Linear(self.data_dim, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        if self.num_layer >= 4:
            self.fc3 = torch.nn.Linear(128, 128)
            self.fc4 = torch.nn.Linear(128, 128)
        if self.num_layer == 6:
            self.fc5 = torch.nn.Linear(128, 128)
            self.fc6 = torch.nn.Linear(128, 128)        
        self.fc7 = torch.nn.Linear(128, self.num_class)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        if self.num_layer == 6:
            x = F.relu(self.fc5(x))
            x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x


###########################
##### active_learning #####
###########################

def train_fullysup(train_loader,test_loader,model,args):
    device= args.device
    #model = simple_Ndim_Net(args)    
    #print(model)
    criterion = args.criterion.to(device) #nn.MSELoss() # nn.CrossEntropyLoss() #
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) 
    total_train_loss,total_train_acc,total_test_loss,total_test_acc=[],[],[],[]
    model = model.to(device)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_acc = 0
        #print(list(model.parameters()))
                
        for batch_idx, (data,labels) in enumerate(train_loader):
            data,labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            
            loss = criterion(output, labels) #torch.eye(2)[labels].cuda())     
            #print(output,labels,loss)   
            loss.backward()
            optimizer.step()
            #print(model.parameters())
            total_loss += loss.item()
            _, val_pred = torch.max(model.forward(data.float()),1)
            total_acc += torch.sum(val_pred==labels.long()).sum().item()/len(labels)
        
        total_train_loss.append(total_loss/len(train_loader))
        total_train_acc.append(total_acc/len(train_loader))

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            for batch_idx, (data,labels) in enumerate(test_loader):
                data,labels = data.to(device), labels.to(device)
                output = model.forward(data.float())
                loss = criterion(output, labels) # torch.eye(2)[labels].cuda()) # labels)        
                total_loss += loss.item()
                _, val_pred = torch.max(model.forward(data.float()),1)
                total_acc += torch.sum(val_pred==labels.long()).sum().item()/len(labels)
            
            total_test_loss.append(total_loss/len(test_loader))
            total_test_acc.append(total_acc/len(test_loader))

        #print(total_train_loss[-1],total_train_acc[-1],total_test_loss[-1],total_test_acc[-1])

    return total_train_acc[-1],total_test_acc[-1]#total_train_loss,total_train_acc,total_test_loss,total_test_acc


def train_active(train_loader,test_loader,net,strategy_type,args):
    NUM_ROUND = args.NUM_ROUND
    NUM_INIT_LABEL = args.NUM_INIT_LABEL
    NUM_QUERY = args.NUM_QUERY
    #strategy_type = args.strategy_type
    acc = torch.zeros((1,NUM_ROUND+1))

    n_pool = len(train_loader.dataset.tensors[1])
    idxs_lb = np.zeros(n_pool,dtype=bool)
    idxs_temp = np.arange(n_pool)
    np.random.shuffle(idxs_temp)
    idxs_lb[idxs_temp[:NUM_INIT_LABEL]] = True

    #X_te = active_test_loader.dataset.tensors[0]
    #Y_te = active_test_loader.dataset.tensors[1]

    strategy = Strategy(strategy_type,train_loader,idxs_lb,net,args)

    # strategy_type,data_loader,idxs_lb,net,args
    strategy.train()
    acc[0,0] = strategy.predict(test_loader)
    #print('ROUND 0')
    print('ROUND 0 Labeded size {} testing accuracy {}'.format(sum(idxs_lb==1), acc[0,0]))

    for rd in range(1,NUM_ROUND+1):
        #print('ROUND {}'.format(rd))
        
        # query
        q_idx = strategy.query(NUM_QUERY)
        idxs_lb[q_idx] = True
        #print(idxs_lb)

        # update
        strategy.update(idxs_lb)
        strategy.train()

        # round accuracy
        acc[0,rd] = strategy.predict(test_loader)
        print('ROUND {} Labeded size {} testing accuracy {}'.format(rd, sum(idxs_lb==1), acc[0,rd]))

    return acc


def main(args):
    ############## CONSTANTS #######################
    data_type = args.data_type #'circle'
    distance_type = args.distance_type #'L2'
    train_add_noise = args.train_add_noise #False
    lr = args.lr #0.005
    wd = args.wd # 1e-4
    network_name = args.network_name ### need to change
    batch_size = args.batch_size # 4, active learning batch size
    epochs = args.epochs # 40
    label_GAN = args.label_GAN #2

    if data_type in ['circle', 'moon']:    n_samples_train = 100
    else:    n_samples_train = 1000 #200 #1000 #200 # number of samples for train
    n_samples_test = 1000 # number of samples for test

    use_cuda = torch.cuda.is_available()
    if torch.cuda.device_count() > 2:
        device = 'cuda:3' if use_cuda else 'cpu' 
    else:
        device = 'cuda' if use_cuda else 'cpu'

    if use_cuda:    n_workers = 4
    else:    n_workers = 1
    
    if data_type in ['mnist','mnist_ext']:
        label_first, label_last = [7,9]
        image_size_ref = 28
        data_class = 'Real'
        num_class,num_class_total,numCH,z_dim, data_dim=[2,10,1,100,2]
        if network_name == 'LeNet' or 'MLP': image_size_ref = 32

    elif data_type in ['cifar10']:
        label_first, label_last = [7,9]
        image_size_ref = 32
        data_class = 'Real'
        num_class,num_class_total,numCH,z_dim=[2,10,3,100]

    else:
        label_first, label_last = [0,1]
        image_size_ref = 32
        data_class = 'Synthetic'
        num_class,num_class_total=[2,2]
        num_layer = 6
        if data_type == 'bracket':    num_layer = 4 
    
        # why we don't have numCH here?
        if data_type in ['high_circle']: z_dim, data_dim= [2,3]
        else:                            z_dim, data_dim= [1,2]

    if data_type == 'circle':
        NUM_INIT_LABEL = 4
        NUM_QUERY = 4
        
    elif data_type == 'mnist':
        args.batch_size = 200
        NUM_INIT_LABEL = 200
        NUM_QUERY = 200

    NUM_ROUND = int((n_samples_train-NUM_INIT_LABEL)/NUM_QUERY) #20 #
    NUM_ITER = 30#5#00

    args.n_workers = n_workers
    args.device = device
    args.label_first = label_first
    args.label_last = label_last
    args.image_size_ref = image_size_ref
    
    args.num_class = num_class
    args.num_class_total = num_class_total
    args.data_class = data_class
    args.num_layer = num_layer
    args.z_dim = z_dim
    args.data_dim = data_dim

    args.num_epochs_z = 10
    args.num_random_z = 3

    args.NUM_INIT_LABEL = NUM_INIT_LABEL
    args.NUM_QUERY = NUM_QUERY
    args.NUM_ROUND = NUM_ROUND
    
    args.active_num_epoch = 3 #1

    args.criterion = nn.CrossEntropyLoss()

    ######## LOAD DATASET
    ###### LOAD Train data ########

    if data_class in ['Real']: #data_type in ['mnist','mnist_ext']:
        train_dataset, test_dataset = load_real_data(data_type, image_size_ref, label_first, label_second, num_mnist_train_data, entire_class, False, no_normalize=False)
        n_samples_hard_train = len(train_dataset)

    elif data_class in ['Synthetic']:
        data_, label_, test_data_, test_label_, X_test = load_synthetic_data(data_type, data_dim, n_samples_train, n_samples_test, False, device, train_add_noise) 
        
    saved_generator = load_GAN(data_type, data_dim, z_dim, num_class, device, train_add_noise, label_GAN) 
    saved_generator.eval()
    saved_generator = saved_generator.to(device)   
    #print(saved_generator)

    args.gan = saved_generator

    tr_dataset = TensorDataset(data_, label_)
    te_dataset = TensorDataset(test_data_, test_label_)
    active_train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size = batch_size, shuffle=True, num_workers = n_workers)    
    active_test_loader = torch.utils.data.DataLoader(te_dataset, batch_size = batch_size, shuffle=True, num_workers = n_workers)    
    
    ############# start active training
    trial_num = int(np.random.randint(10))
    seed((int)(42+trial_num))

    acc = torch.zeros((2)) #NUM_ITER))#,NUM_ROUND+1))
    fin_acc_rand = torch.zeros((NUM_ROUND+1)) #NUM_ITER,
    fin_acc_GAN = torch.zeros((NUM_ROUND+1))#NUM_ITER,

    #train_loader,test_loader,net,strategy_type,args

    for num_iter in range(NUM_ITER):
        #total_train_loss,total_train_acc,total_test_loss,total_test_acc
        network = simple_Ndim_Net(args)
        #print(network)
        acc_sup_train,acc_sup_test = train_fullysup(active_train_loader,active_test_loader,network,args)
        acc[0]=((num_iter*acc[0])+acc_sup_train)/(num_iter+1)
        acc[1]=((num_iter*acc[1])+acc_sup_test)/(num_iter+1)
        print(acc)
        
        network = simple_Ndim_Net(args)
        strategy_type = 'RandomSampling'
        acc_rand = train_active(active_train_loader,active_test_loader,network,strategy_type,args)
        #train_with_randomquery(tr_dataset,active_test_loader,network,criterion,active_batch_size,n_workers,NUM_ROUND,NUM_INIT_LABEL,NUM_QUERY)
        #acc2 = train_with_GANmixupmove(tr_dataset,network,optimizer,criterion,active_batch_size,n_workers)
        
        network = simple_Ndim_Net(args)
        strategy_type = 'GANdistance'
        acc_GAN = train_active(active_train_loader,active_test_loader,network,strategy_type,args)

        fin_acc_rand = ((num_iter*fin_acc_rand)+torch.FloatTensor(acc_rand))/(num_iter+1)
        fin_acc_GAN = ((num_iter*fin_acc_rand)+torch.FloatTensor(acc_GAN))/(num_iter+1)

        print('single iteration finished')
        print(num_iter,'\n',acc,'\n',fin_acc_rand,'\n',fin_acc_GAN)



if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='PyTorch Training: GAN-active learning')

    parser.add_argument('--data_type', default='circle', type=str, choices=['moon', 'circle', 'high_circle', 'v_shaped', 'bracket', 'mnist', 'mnist_ext', 'cifar10'], help='Data Type')
    parser.add_argument('--distance_type', default='L2', type=str, choices=['L2', 'L1', 'Linf', 'Linf+L2'], help='Distance Type')
    parser.add_argument('--train_add_noise', default=False, type=bool, help='Add noise to the train data')
    parser.add_argument('--network_name', default=None, type=str, choices=['LeNet', 'ConvNet', 'MLP'], help='name of the network used for training real data')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate (lr)')
    parser.add_argument('--wd', default=1e-5, type=float, help='Weight decay (wd)')
    parser.add_argument('--epochs', default=20, type=int, help='Training epochs')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--label_GAN', default=2, type=int, choices = [2,10], help='Dimension of label y in conditional GAN (for mnist)')
    args = parser.parse_args()
        
    #sys.stdout = open("outputfile.txt","w")
    main(args)
    #sys.stdout.close()

    
    # if data_class == 'Synthetic':
    #     model = simple_Ndim_Net(num_layer, data_dim, num_class).to(device)
    #     best_model = simple_Ndim_Net(num_layer, data_dim, num_class).to(device)
    # else:
    #     if network_name == 'ConvNet': model, best_model = ConvNet().to(device), ConvNet().to(device)
    #     elif network_name == 'LeNet': model, best_model = LeNet(num_class_total).to(device), LeNet(num_class_total).to(device)
    #     elif network_name == 'MLP': model, best_model = MLP().to(device), MLP().to(device)

    