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

from PIL import Image

def checkpoint(args, w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list):
    # Write metric store into a CSV
    metrics_df = pd.DataFrame(
        {
            'Train acc': ms_acc_train_list,
            'Test acc': ms_acc_test_list,
            'Train loss': ms_loss_train_list,
            'Test loss': ms_loss_test_list
        })

    fn_suffix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}'.format(
                                            args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                            args.grad_abs_noise_stdev,
                                            args.grad_rltv_noise_stdev,)

    metrics_df.to_csv(f"./{args.result_dir}/fed_stats{fn_suffix}.csv", sep='\t')

    torch.save(w_glob, f"./{args.result_dir}/saved_model{fn_suffix}")

def load_checkpoint_if_exists(args):
    fn_suffix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}'.format(
                                            args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                            args.grad_abs_noise_stdev,
                                            args.grad_rltv_noise_stdev,)
    
    if not os.path.exists(f"./{args.result_dir}/saved_model{fn_suffix}"):
        return None, None, None, None, None
    
    metrics_df = pd.read_csv(f"./{args.result_dir}/fed_stats{fn_suffix}.csv", sep='\t')

    ms_acc_train_list = metrics_df['Train acc'].tolist()
    ms_acc_test_list = metrics_df['Test acc'].tolist()
    ms_loss_train_list = metrics_df['Train loss'].tolist()
    ms_loss_test_list = metrics_df['Test loss'].tolist()

    w_glob = torch.load(f"./{args.result_dir}/saved_model{fn_suffix}")

    return w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list


if __name__ == '__main__':
    start_time = time.time()
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
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

    a1, a2, a3, a4, a5 = load_checkpoint_if_exists(args)


    if a1 is not None:
        print(f"Checkpoint exists! Epoch: {len(a2)}")
        net_glob.load_state_dict(a1)
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = a2, a3, a4, a5
    else:
        # metrics to store
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list = [], [], [], []

    # Define LR Schedule
    values = args.lr_interval.split()
    lr_interval = []
    for value in values:
        lr_interval.append(int(float(value)*args.epochs))

    # Define Fed Learn object
    fl = FedLearn(args)

    iter = len(ms_acc_train_list)
    for i in range(iter):
        if i in lr_interval:
            args.lr = args.lr/args.lr_reduce
    while iter < args.epochs:
        net_glob.train()
        w_locals_selected, loss_locals_selected = [], []
        w_locals_all, loss_locals_all = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print("Selected clients:", idxs_users)
        w_glob = copy.deepcopy(net_glob.state_dict())
        w_glob = fl.AddNoiseAbs(w_glob)
        w_glob = fl.AddNoiseRltv(w_glob)

        for idx in idxs_users:
            if args.verbose:
                print(f"Training client {idx}")
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # idxs needs the list of indices assigned to this particular client
            model_copy = type(net_glob)(**model_args) # get a new instance
            model_copy.load_state_dict(w_glob) # copy weights and stuff
            w, loss = local.train(net=model_copy.to(args.device))
            w = fl.AddNoiseAbs(w)
            w = fl.AddNoiseRltv(w)
            w_locals_all.append(copy.deepcopy(w))
            loss_locals_all.append(copy.deepcopy(loss))
            if idx in idxs_users:
                w_locals_selected.append(copy.deepcopy(w))
                loss_locals_selected.append(copy.deepcopy(loss))
        
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
            checkpoint(args, w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list)

        if iter in lr_interval:
            args.lr = args.lr/args.lr_reduce

        iter += 1

    print(f"Total training time: {(time.time() - start_time)/3600}")