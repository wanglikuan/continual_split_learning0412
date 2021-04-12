import os

import torch
import torch.nn as nn

from torchsummary.torchsummary import summary

# from data import get_permute_mnist, get_split_mnist, get_rotate_mnist, get_split_cifar
from models import cifar, mnist

import importlib

import argparse
parser = argparse.ArgumentParser()

# preliminary settings   
parser.add_argument('--method', type=str, default='ewc')
parser.add_argument('--online', type=bool, default=False)
parser.add_argument('--model', type=str, default='ResNet')
parser.add_argument('--dataset', type=str, default='TinyImageNet')
parser.add_argument('--class-incremental', type=bool, default=False)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--num-task', type=int, default=40)
parser.add_argument('--num-gpu', type=int, default=1)

# Hyper-parameter settings 
parser.add_argument('--bsz', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.1) 
parser.add_argument('--decay', type=int, default=100) # number of epochs that learning rate decreases
parser.add_argument('--lr-adjust', type=float, default=1)   #This is for delay factor

parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lam', type=float, default=15)
parser.add_argument('--threshold', type=float, default=0.9) #

args = parser.parse_args()

def model_retrieval():
    if 'cifar' in args.dataset:
        model = {
            'ResNet': cifar.Reduced_ResNet18(100),
            'VGG': cifar.vgg16(100),
            #'LeNet': cifar.LeNet(100),
            #'CNN': cifar.CNN(100),
            'AlexNet': cifar.AlexNet(100)
        }[args.model]
        file_name = '{}_cifar.model'.format(args.model)
    if 'ImageNet' in args.dataset:
        model = {
            'ResNet': cifar.ResNet18(200)
        }[args.model]
        file_name = '{}_ImageNet.model'.format(args.model)
    if 'mnist' in args.dataset:
        model = {
            'ResNet18': mnist.ResNet18(),
            'VGG': mnist.VGG(),
            'LeNet': mnist.LeNet(),
            'CNN': mnist.CNN(),
            'AlexNet': mnist.AlexNet()
        }[args.model]
        file_name = '{}_mnist.model'.format(args.model)
    if not os.path.exists('./models/{}'.format(file_name)):
        torch.save(model.state_dict(), './models/{}'.format(file_name))
    model.load_state_dict(torch.load('./models/{}'.format(file_name)))
    return model

def generate_cut_layer(cut_layer_idx, model):
    input_datasize = (3, 64, 64)
    # input_datasize = (3, 32, 32) if 'cifar' in args.dataset else (1, 28, 28)  
    cut_layer_idx = min(cut_layer_idx, len(summary(model, input_datasize, depth=1, verbose=0).summary_list))
    # print(summary(model, input_datasize, depth=5, verbose=0))

    tot_param_num = 0
    for i, summary_list in enumerate(summary(model, input_datasize, depth=0, verbose=0).summary_list):
        tot_param_num += (summary_list.num_params if i < cut_layer_idx else 0)
    
    for i, param in enumerate(model.parameters()):
        tot_param_num -= len(param.reshape(-1))
        if tot_param_num == 0:
            return  i+1
    return 0

def print_model(model:nn.Module):
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(idx, name)
    print("===================")
    for idx, (key, value) in enumerate(model.state_dict().items()):
        print(idx, key)


if __name__ == '__main__':
    model = model_retrieval()
    cut_idx = generate_cut_layer(args.split, model)
    print('Cut index: {}'.format(cut_idx))
    class_incremental = 'class' in args.dataset
    result_file='./result/2095_t10_{}_{}_{}_{}.txt'.format(args.method, args.dataset, args.split, args.model)

    dataset_module = importlib.import_module('data.' + args.dataset)
    train_loader, test_loader, labels = dataset_module.get_dataset(args.bsz, args.num_task)

    method_module = importlib.import_module("methods." + args.method)
    method_module.process(model, cut_idx, args, train_loader, test_loader, labels, result_file)

    # client_model = cifar.new_ResNet()
    # print(client_model)
