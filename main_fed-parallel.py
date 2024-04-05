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
        self.fn_suffix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}_cmprsrate{}'.format(
                                            args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                            args.grad_abs_noise_stdev,
                                            args.grad_rltv_noise_stdev,
                                            args.params_compress_rate,)

    def checkpoint(self, w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list):
        # Write metric store into a CSV
        metrics_df = pd.DataFrame(
            {
                'Train acc': ms_acc_train_list,
                'Test acc': ms_acc_test_list,
                'Train loss': ms_loss_train_list,
                'Test loss': ms_loss_test_list
            })

        metrics_df.to_csv(f"./{args.result_dir}/fed_stats{self.fn_suffix}.csv", sep='\t')

        torch.save(w_glob, f"./{args.result_dir}/saved_model{self.fn_suffix}")

    def load_checkpoint_if_exists(self):
        
        if not os.path.exists(f"./{args.result_dir}/saved_model{self.fn_suffix}"):
            return None, None, None, None, None
        
        metrics_df = pd.read_csv(f"./{args.result_dir}/fed_stats{self.fn_suffix}.csv", sep='\t')

        ms_acc_train_list = metrics_df['Train acc'].tolist()
        ms_acc_test_list = metrics_df['Test acc'].tolist()
        ms_loss_train_list = metrics_df['Train loss'].tolist()
        ms_loss_test_list = metrics_df['Test loss'].tolist()

        w_glob = torch.load(f"./{args.result_dir}/saved_model{self.fn_suffix}")

        return w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list


class Client:
    def __init__(self, args, net_class, model_args, node_id, dataset_train, train_idxs) -> None:
        self.db_postfix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}_cmprsrate{}'.format(
                                            args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                            args.grad_abs_noise_stdev,
                                            args.grad_rltv_noise_stdev,
                                            args.params_compress_rate,)
        self.args = args
        self.node_id = node_id
        self.net_class = net_class
        self.model_args = model_args
        self.dataset_train = dataset_train
        self.train_idxs = train_idxs

        
        self.result_cache_path = f"miscellaneous/database{self.db_postfix}/local_result_{self.node_id}.pkl"

    
    def fit(self):
        while True:
            try:
                with open(f"miscellaneous/database{self.db_postfix}/data.pkl", 'rb') as file:
                    trainsets = pickle.load(file)
                ldr_train = DataLoader(trainsets[self.node_id], batch_size=self.args.local_bs, shuffle=True, drop_last=True)
                
                w_glob = torch.load(f"miscellaneous/database{self.db_postfix}/w_glob.pkl")

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
                    pickle.dump((net.state_dict(), sum(epoch_loss) / len(epoch_loss)), file)


                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                return 
            except Exception as e:
                print(f"Training failed {e}")
                print(f"Sleeping for random seconds before retrying")
                time.sleep(random.randrange(5, 14))


    def clear_result_cache(self):
        if os.path.exists(self.result_cache_path):
            os.remove(self.result_cache_path)

    def spawn_new_local_training(self):
        
        
        self.local_traing_process = multiprocessing.Process(target=self.fit, args=())

        self.clear_result_cache()

        self.local_traing_process.start()
        self.local_training_in_progress = True
    
    def load_local_training_result_if_done(self):
        if not os.path.exists(self.result_cache_path) or self.local_traing_process.exitcode != 0:
            print("Training failed")
            return False
        
        with open(self.result_cache_path, 'rb') as file:
            self.local_w, self.local_loss = pickle.load(file)

        self.clear_result_cache()
        del self.local_traing_process
        return True
    

if __name__ == '__main__':
    start_time = time.time()
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    db_postfix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}_cmprsrate{}'.format(
                                        args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                        args.grad_abs_noise_stdev,
                                        args.grad_rltv_noise_stdev,
                                        args.params_compress_rate,)
    
    if not os.path.exists(f"miscellaneous/database{db_postfix}"):
            os.mkdir(f"miscellaneous/database{db_postfix}")

    trainsets = [Subset(dataset_train, list(dict_users[idx]))
                        for idx in range(len(dict_users))]
    
    with open(f"miscellaneous/database{db_postfix}/data.pkl", 'wb') as file:
        pickle.dump(trainsets, file)

    # build model
    model_args = {'args': args}
    if args.model[0:3].lower() == 'vgg':
        if args.snn:
            model_args = {'num_cls': args.num_classes, 'timesteps': args.timesteps, 'device': args.device}
            net_glob = snn_models_bntt.SNN_VGG9_BNTT(**model_args).to(args.device)
        else:
            model_args = {'vgg_name': args.model, 'labels': args.num_classes, 'dataset': args.dataset, 'kernel_size': 3, 'dropout': args.dropout}
            net_glob = ann_models.VGG(**model_args).to(args.device)
    elif args.model[0:6].lower() == 'resnet':
        if args.snn:
            pass
        else:
            model_args = {'num_cls': args.num_classes}
            net_glob = resnet_models.Network(**model_args).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    cp = CheckPoint(args)

    a1, a2, a3, a4, a5 = cp.load_checkpoint_if_exists()


    if a1 is not None:
        print(f"Checkpoint exists! Epoch: {len(a2)}")
        net_glob.load_state_dict(a1)
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = a2, a3, a4, a5
    else:
        # metrics to store
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = [], [], [], []

    clients = [Client(args, type(net_glob), model_args, node_id, dataset_train, dict_users[node_id]) for node_id in range(args.num_users)]

    multiprocessing.set_start_method('forkserver')

    # Define Fed Learn object
    fl = FedLearn(args)

    iter = len(ms_acc_train_list)
    while iter < args.epochs:
        w_locals_selected, loss_locals_selected = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)
        w_glob = net_glob.state_dict()
        w_glob = fl.AddNoiseAbs(w_glob)
        w_glob = fl.AddNoiseRltv(w_glob)
        w_glob = fl.CompressParams(w_glob)
        torch.save(w_glob, f"miscellaneous/database{db_postfix}/w_glob.pkl")

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
            w_locals_selected.append(w)
            loss_locals_selected.append(clients[idx].local_loss)

        # update global weights
        w_glob = fl.FedAvg(w_locals_selected)
        
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)
 
        # print loss
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
 
        
        net_glob.eval()
        # acc_train, loss_train = test_img(net_glob, dataset_train, args)
        # print("Round {:d}, Training accuracy: {:.2f}".format(iter, acc_train))
        acc_train, loss_train = 0.0, 0.0
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Round {:d}, Testing accuracy: {:.2f}".format(iter, acc_test))

        # Add metrics to store
        ms_acc_train_list.append(acc_train)
        ms_acc_test_list.append(acc_test)
        ms_loss_train_list.append(loss_train)
        ms_loss_test_list.append(loss_test)

        if iter % args.checkpoint_every == 0:
            cp.checkpoint(w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list)

        iter += 1

    print(f"Total training time: {(time.time() - start_time)/3600}")