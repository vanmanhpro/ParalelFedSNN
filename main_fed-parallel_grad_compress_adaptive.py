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


def load_obj(file_name):
    with open(file_name, 'rb') as file:
        obj = pickle.load(file)

    return obj

def save_obj(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)

def linear_cmprate(args, epoch):
    return np.linspace(args.initial_cmp_rate, args.final_cmp_rate, args.epochs)[epoch]

def linear_cycle_cmprate(args, cycle_count):
    y = np.linspace(args.initial_cmp_rate, args.final_cmp_rate, args.cycle_cmp_rate)
    if cycle_count >= args.cycle_cmp_rate:
        return y[-1]
    else: 
        return y[cycle_count]

def exp_cmprate(args, epoch):
    return np.exp(np.linspace(np.log(args.initial_cmp_rate), np.log(args.final_cmp_rate), args.epochs))[epoch]

def exp_cycle_cmprate(args, cycle_count):
    y = np.exp(np.linspace(np.log(args.initial_cmp_rate), np.log(args.final_cmp_rate), args.cycle_cmp_rate))
    if cycle_count >= args.cycle_cmp_rate:
        return y[-1]
    else: 
        return y[cycle_count]

class CheckPoint():
    def __init__(self, args) -> None:
        self.args = args

    def checkpoint(self, w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list, ms_cmp_rate_list, iter, cycle_count):
        # Write metric store into a CSV
        metrics_df = pd.DataFrame(
            {
                'Train acc': ms_acc_train_list,
                'Test acc': ms_acc_test_list,
                'Train loss': ms_loss_train_list,
                'Test loss': ms_loss_test_list, 
                'Cmp rate': ms_cmp_rate_list
            })

        metrics_df.to_csv(f"./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv", sep='\t')

        save_obj((w_glob, iter, cycle_count), f"./{self.args.result_dir}/saved_model{self.args.db_postfix}")

    def load_checkpoint_if_exists(self):
        print(f"rm ./{self.args.result_dir}/saved_model{self.args.db_postfix}")
        print(f"rm ./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv")
        
        if not os.path.exists(f"./{self.args.result_dir}/saved_model{self.args.db_postfix}"):
            return None, None, None, None, None, None, 0, 0 
        
        metrics_df = pd.read_csv(f"./{self.args.result_dir}/fed_stats{self.args.db_postfix}.csv", sep='\t')

        ms_acc_train_list = metrics_df['Train acc'].tolist()
        ms_acc_test_list = metrics_df['Test acc'].tolist()
        ms_loss_train_list = metrics_df['Train loss'].tolist()
        ms_loss_test_list = metrics_df['Test loss'].tolist()
        ms_cmp_rate_list = metrics_df['Cmp rate'].tolist()

        w_glob, iter, cycle_count = load_obj(f"./{self.args.result_dir}/saved_model{self.args.db_postfix}")

        return w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list, ms_cmp_rate_list, iter, cycle_count


class Client:
    def __init__(self, args, net_class, model_args, node_id) -> None:
        self.args = args
        self.node_id = node_id
        self.net_class = net_class
        self.model_args = model_args
        if 'device' in self.model_args:
            self.model_args['device'] = self.args.device
        
        self.result_cache_path = f"miscellaneous/database{self.args.db_postfix}/local_result_{self.node_id}.pkl"

    def test(self, w):
        _, dataset_test = load_obj(f"miscellaneous/database{self.args.db_postfix}/data.pkl")
        
        net = self.args.net_class(**self.model_args).to(self.args.device)
        net.load_state_dict(w)
        net.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(dataset_test, batch_size=self.args.bs)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            if self.args.gpu != -1:
                data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = net(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        if self.args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))

    def fit(self):
        
        while True:
            try:
                fl = FedLearn(self.args)
                trainsets, _ = load_obj(f"miscellaneous/database{self.args.db_postfix}/data.pkl")
                ldr_train = DataLoader(trainsets[self.node_id], batch_size=self.args.local_bs, shuffle=True, drop_last=True)
                
                w_glob = load_obj(f"miscellaneous/database{self.args.db_postfix}/w_glob.pkl")

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

                w_trained = net.cpu().state_dict()
                g_trained = fl.GetGrad(w_trained, w_glob)
                g_sel = fl.MapComprGrad(g_trained)

                save_obj(((g_trained, g_sel), sum(epoch_loss) / len(epoch_loss)), self.result_cache_path)

                del net, loss_func, epoch_loss, w_glob, ldr_train, w_trained, g_trained, g_sel

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                        
                return 
            except Exception as e:
                print(f"Training failed {e}")
                print(f"Sleeping for random seconds before retrying")
                time.sleep(random.randrange(50, 60))


    def clear_result(self):
        del self.local_g, self.local_loss

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
        
        self.local_g, self.local_loss = load_obj(self.result_cache_path)

        self.clear_result_cache()
        del self.local_traing_process
        return True
    

def test(args, model_args):
    if 'device' in model_args:
        model_args['device'] = args.device

    _, dataset_test = load_obj(f"miscellaneous/database{args.db_postfix}/data.pkl")
     
    net_glob = args.net_class(**model_args).to(args.device)
    w_glob = load_obj(f"miscellaneous/database{args.db_postfix}/w_glob.pkl")
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
    
    save_obj((accuracy.item(), test_loss), f"miscellaneous/database{args.db_postfix}/test_result.pkl")

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

    args.db_postfix = '_{}_{}_snn{}_epoch{}-{}_C{}-{}_iid{}_absnoise{}_rltvnoise{}_cmprsrate-{}-{}-{}-{}-{}'.format(
                                        args.dataset, args.model, args.snn, args.epochs, args.local_ep, args.num_users, args.frac, args.iid,
                                        args.grad_abs_noise_stdev,
                                        args.grad_rltv_noise_stdev,
                                        args.cmp_rate_decrease,
                                        args.initial_cmp_rate,
                                        args.final_cmp_rate,
                                        args.cycle_cmp_rate,
                                        args.loss_diff_tolerance)
    
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
    
    save_obj((trainsets, dataset_test), f"miscellaneous/database{args.db_postfix}/data.pkl")

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

    a1, a2, a3, a4, a5, a6, a7, a8 = cp.load_checkpoint_if_exists()


    if a1 is not None:
        print(f"Checkpoint exists! Epoch: {len(a2)}")
        w_glob = a1
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list, ms_cmp_rate_list = a2, a3, a4, a5, a6
        iter, cycle_count = a7, a8
    else:
        w_glob = net_glob.state_dict()
        # metrics to store
        ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list, ms_cmp_rate_list = [], [], [], [], []
        iter, cycle_count = 0, 0 


    args.net_class = type(net_glob)
    clients = [Client(args, type(net_glob), model_args, node_id) for node_id in range(args.num_users)]

    del net_glob

    args.cmp_rate = args.initial_cmp_rate
    
    # Define Fed Learn object
    fl = FedLearn(args)
    

    save_obj(w_glob, f"miscellaneous/database{args.db_postfix}/w_glob.pkl")

    while iter < args.epochs:
        epoch_start_time = time.time()
        
        if args.cycle_cmp_rate is None:
            if args.cmp_rate_decrease == 'linear':
                args.cmp_rate = linear_cmprate(args, iter)
            elif args.cmp_rate_decrease == 'exp':
                args.cmp_rate = exp_cmprate(args, iter)
        else:
            if args.cmp_rate_decrease == 'linear':
                args.cmp_rate = linear_cycle_cmprate(args, cycle_count)
            elif args.cmp_rate_decrease == 'exp':
                args.cmp_rate = exp_cycle_cmprate(args, cycle_count)
            cycle_count += 1

        g_locals_selected, loss_locals_selected = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(f"Selected clients: {idxs_users}, Cmp Rate: {args.cmp_rate}")

        for idx in idxs_users:
            if args.verbose:
                print(f"Training client {idx}")
            clients[idx].args.cmp_rate = args.cmp_rate
            clients[idx].spawn_new_local_training()
        
        for idx in idxs_users:
            clients[idx].local_traing_process.join()

        for idx in idxs_users:
            clients[idx].load_local_training_result_if_done()

        for idx in idxs_users:
            g_locals_selected.append(copy.deepcopy(clients[idx].local_g))
            loss_locals_selected.append(clients[idx].local_loss)

            clients[idx].clear_result()

        # update global weights
        g_avg = fl.GradFedAvg(g_locals_selected)
        w_glob = fl.ApplyGrad(w_glob, g_avg)

        save_obj(w_glob, f"miscellaneous/database{args.db_postfix}/w_glob.pkl")
 
        # print loss
        print("Local loss:", loss_locals_selected)
        loss_avg = sum(loss_locals_selected) / len(loss_locals_selected)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))

        acc_train, loss_train = 0.0, 0.0
        
        testing_process = multiprocessing.Process(target=test, args=(args, model_args,))
        testing_process.start()
        testing_process.join()
        
        acc_test, loss_test = load_obj(f"miscellaneous/database{args.db_postfix}/test_result.pkl")

        if args.cycle_cmp_rate is not None:
            if abs(loss_avg - loss_test) >= args.loss_diff_tolerance:
                cycle_count = 0
                print("Cmp rate cycle reset!")


        print("Round {:d}, Testing accuracy & loss: {:.2f} & {:.3f}, Time: {:.2f}".format(iter, acc_test, loss_test, (time.time() - epoch_start_time)/60))

        # Add metrics to store
        ms_acc_train_list.append(acc_train)
        ms_acc_test_list.append(acc_test)
        ms_loss_train_list.append(loss_train)
        ms_loss_test_list.append(loss_test)
        ms_cmp_rate_list.append(args.cmp_rate)

        iter += 1

        if iter % args.checkpoint_every == 0:
            cp.checkpoint(w_glob, ms_acc_train_list, ms_acc_test_list, ms_loss_train_list, ms_loss_test_list, ms_cmp_rate_list, iter, cycle_count)

        

        for g in g_locals_selected:
            del g

        del g_locals_selected, loss_locals_selected, g_avg

    print(f"Total training time: {(time.time() - start_time)/3600}")