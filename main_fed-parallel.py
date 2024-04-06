#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_non_iid, mnist_dvs_iid, mnist_dvs_non_iid, nmnist_iid, nmnist_non_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedLearn
from models.Fed import model_deviation
from models.test import test_img
import models.vgg as ann_models
import models.resnet as resnet_models
import models.vgg_spiking_bntt as snn_models_bntt


import os
import tables
import yaml
import glob
import json
import time
import pickle

import random

from torch import multiprocessing

from PIL import Image

class CheckPoint():
    def __init__(self, args) -> None:
        self.args = args

    def checkpoint(self, w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list):
        # Write metric store into a CSV
        metrics_df = pd.DataFrame(
            {
                'Train acc': ms_acc_train_list,
                'Test acc': ms_acc_test_list,
                'Train loss': ms_loss_train_list,
                'Test loss': ms_loss_test_list
            })

        metrics_df.to_csv(f"./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv", sep='\t')

        torch.save(w_glob, f"./{self.args.result_dir}/saved_model{self.args.db_postfix}")

    def load_checkpoint_if_exists(self):
        print(f"Checkpoint model: ./{self.args.result_dir}/saved_model{self.args.db_postfix}")
        print(f"Checkpoint stats: ./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv")
        
        if not os.path.exists(f"./{self.args.result_dir}/saved_model{self.args.db_postfix}"):
            return None, None, None, None, None
        
        metrics_df = pd.read_csv(f"./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv", sep='\t')

        ms_acc_train_list = metrics_df['Train acc'].tolist()
        ms_acc_test_list = metrics_df['Test acc'].tolist()
        ms_loss_train_list = metrics_df['Train loss'].tolist()
        ms_loss_test_list = metrics_df['Test loss'].tolist()

        w_glob = torch.load(f"./{self.args.result_dir}/saved_model{self.args.db_postfix}")

        return w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list


class Client:
    def __init__(self, args, net_class, model_args, node_id) -> None:
        self.args = args
        self.node_id = node_id
        self.net_class = net_class
        self.model_args = model_args
        if 'device' in self.model_args:
            self.model_args['device'] = self.args.device
        
        self.result_cache_path = f"miscellaneous/database{self.args.db_postfix}/local_result_{self.node_id}.pkl"

    
    def fit(self):
        while True:
            try:
                with open(f"miscellaneous/database{self.args.db_postfix}/data.pkl", 'rb') as file:
                    trainsets, _ = pickle.load(file)
                ldr_train = DataLoader(trainsets[self.node_id], batch_size=self.args.local_bs, shuffle=True, drop_last=True)
                
                w_glob = torch.load(f"miscellaneous/database{self.args.db_postfix}/w_glob_noised.pkl")

                net = self.net_class(**self.model_args).to(self.args.device)
                net.load_state_dict(w_glob)
                # Load Data
                net.train()
                # train and update
                loss_func = nn.CrossEntropyLoss()
                if self.args.optimizer == "SGD":
                    optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay = self.args.weight_decay)
                elif self.args.optimizer == "Adam":
                    optimizer = torch.optim.Adam(net.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay, amsgrad = True)
                else:
                    print("Invalid optimizer")
                epoch_loss = []
                for iter in range(self.args.local_ep):
                    batch_loss = []
                    for batch_idx, (images, labels) in enumerate(ldr_train):
                        images, labels = images.to(self.args.device), labels.to(self.args.device)
                        net.zero_grad()
                        log_probs = net(images)
                        loss = loss_func(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        if self.args.verbose and batch_idx % 10 == 0:
                            print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                iter, batch_idx * len(images), len(ldr_train.dataset),
                                    100. * batch_idx / len(ldr_train), loss.item()))
                        batch_loss.append(loss.item())
                    epoch_loss.append(sum(batch_loss)/len(batch_loss))
                

                with open(self.result_cache_path, 'wb') as file:
                    pickle.dump((net.cpu().state_dict(), sum(epoch_loss) / len(epoch_loss)), file)

                del net, loss_func, epoch_loss, w_glob, ldr_train

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return 
            except Exception as e:
                print(f"Training failed {e}")
                return
                print(f"Sleeping for random seconds before retrying")
                time.sleep(random.randrange(5, 14))


    def clear_result(self):
        del self.local_w, self.local_loss

    def clear_result_cache(self):
        if os.path.exists(self.result_cache_path):
            os.remove(self.result_cache_path)

    def spawn_new_local_training(self):
        
        
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=())

        self.clear_result_cache()

        self.local_traing_process.start()
    
    def load_local_training_result_if_done(self):
        if not os.path.exists(self.result_cache_path) or self.local_traing_process.exitcode != 0:
            print("Training failed")
            return False
        
        with open(self.result_cache_path, 'rb') as file:
            self.local_w, self.local_loss = pickle.load(file)

        self.clear_result_cache()
        del self.local_traing_process
        return True
    

def test(args, model_args):
    if 'device' in model_args:
        model_args['device'] = args.device

    with open(f"miscellaneous/database{args.db_postfix}/data.pkl", 'rb') as file:
        _, dataset_test = pickle.load(file)
     
    net_glob = args.net_class(**model_args).to(args.device)
    w_glob = torch.load(f"miscellaneous/database{args.db_postfix}/w_glob.pkl")
    net_glob.load_state_dict(w_glob)
    net_glob.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset_test, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_glob(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    
    with open(f"miscellaneous/database{args.db_postfix}/test_result.pkl", 'wb') as file:
        pickle.dump((accuracy.item(), test_loss), file)

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')

    start_time = time.time()
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.cpu_device = torch.device('cpu')
    args.device = torch.device('cuda:{}'.format(int(args.gpu.split(',')[0])) if torch.cuda.is_available() and args.gpu != '-1' else 'cpu')
    print(args.device)
    if args.device != 'cpu':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset_keys = None
    h5fs = None
    # load dataset and split users
    if args.dataset == 'CIFAR10':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    elif args.dataset == 'CIFAR100':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_non_iid(dataset_train, args.num_classes, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    args.db_postfix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}_cmprsrate{}'.format(
                                        args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                        args.grad_abs_noise_stdev,
                                        args.grad_rltv_noise_stdev,
                                        args.params_compress_rate,)
    
    cache_paths = [
        f"miscellaneous/database{args.db_postfix}/test_result.pkl",
        f"miscellaneous/database{args.db_postfix}/data.pkl",
        f"miscellaneous/database{args.db_postfix}/w_glob_noised.pkl",
        f"miscellaneous/database{args.db_postfix}/w_glob.pkl"
    ]
    for path in cache_paths:
        if os.path.exists(path):
            os.remove(path)


    if not os.path.exists(f"miscellaneous/database{args.db_postfix}"):
        os.mkdir(f"miscellaneous/database{args.db_postfix}")

    trainsets = [Subset(dataset_train, list(dict_users[idx]))
                        for idx in range(len(dict_users))]
    
    with open(f"miscellaneous/database{args.db_postfix}/data.pkl", 'wb') as file:
        pickle.dump((trainsets, dataset_test), file)

    # build model
    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg':
        if args.snn:
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps, 'device': args.cpu_device}
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).to(args.cpu_device)
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).to(args.cpu_device)
    elif args.model[0:6].lower() == 'resnet':
        if args.snn:
            pass
        else:
            model_args = {'num_cls': args.num_classes}
            net_glob = resnet_models.Network(**model_args).to(args.cpu_device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    cp = CheckPoint(args)

    a1, a2, a3, a4, a5 = cp.load_checkpoint_if_exists()


    if a1 is not None:
        print(f"Checkpoint exists! Epoch: {len(a2)}")
        w_glob = a1
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = a2, a3, a4, a5
    else:
        w_glob = net_glob.state_dict()
        # metrics to store
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = [], [], [], []

    args.net_class = type(net_glob)
    clients = [Client(args, type(net_glob), model_args, node_id) for node_id in range(args.num_users)]

    del net_glob

    # Define Fed Learn object
    fl = FedLearn(args)

    iter = len(ms_acc_train_list)
    while iter < args.epochs:
        w_locals_selected, loss_locals_selected = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)
        w_glob = fl.AddNoiseAbs(w_glob)
        w_glob = fl.AddNoiseRltv(w_glob)
        w_glob = fl.CompressParams(w_glob)
        torch.save(w_glob, f"miscellaneous/database{args.db_postfix}/w_glob_noised.pkl")

        for idx in idxs_users:
            if args.verbose:
                print(f"Training client {idx}")
            clients[idx].spawn_new_local_training()
        
        for idx in idxs_users:
            clients[idx].local_traing_process.join()

        for idx in idxs_users:
            clients[idx].load_local_training_result_if_done()

        for idx in idxs_users:
            w = clients[idx].local_w
            w = fl.AddNoiseAbs(w)
            w = fl.AddNoiseRltv(w)
            w = fl.CompressParams(w)
            w_locals_selected.append(copy.deepcopy(w))
            loss_locals_selected.append(clients[idx].local_loss)

            del w
            clients[idx].clear_result()

        # update global weights
        del w_glob
        w_glob = fl.FedAvg(w_locals_selected)

        

        torch.save(w_glob, f"miscellaneous/database{args.db_postfix}/w_glob.pkl")
 
        # print loss
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
 

        acc_train, loss_train = 0.0, 0.0
        
        testing_process = multiprocessing.Process(target=test, args=(args, model_args,))
        testing_process.start()
        testing_process.join()
        
        with open(f"miscellaneous/database{args.db_postfix}/test_result.pkl", 'rb') as file:
            acc_test, loss_test = pickle.load(file)

        print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))

        # Add metrics to store
        ms_acc_train_list.append(acc_train)
        ms_acc_test_list.append(acc_test)
        ms_loss_train_list.append(loss_train)
        ms_loss_test_list.append(loss_test)

        if iter % args.checkpoint_every == 0:
            cp.checkpoint(w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list)

        iter += 1

        for w in w_locals_selected:
            del w

        del w_locals_selected, loss_locals_selected

    print(f"Total training time: {(time.time() - start_time)/3600}")