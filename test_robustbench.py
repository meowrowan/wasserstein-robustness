import argparse
import os
import time
import traceback
import sys
import copy


parser = argparse.ArgumentParser(description='Baseline')
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str) 
parser.add_argument('--advloss', type=str)
parser.add_argument('--eps', type=float)
parser.add_argument('--gpu', type=str)
parser.add_argument('--ckpt-path', type=str)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
device = torch.device('cuda')

import torch

from pytorchmodels.resnet import resnet20_cifar10_two, resnet56_cifar10_two
from pytorchmodels.wideresnet import wrn28_10_cifar10_two

from robustbench.eval import benchmark

def main():

    if args.model == "resnet56":
        model = resnet56_cifar10_two()
    elif args.model == "resnet20":
        model = resnet20_cifar10_two()
    else: # model == wideresnet28-10
        model = wrn28_10_cifar10_two()
    
    # Evaluate the Linf robustness of the model using AutoAttack
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    print("model is ready: loaded with robust weight")

    clean_acc, robust_acc = benchmark(model,
                                     dataset=args.dataset,
                                      threat_model='Linf', eps=args.eps, device=device, data_dir="/dataset/cifar10")

if __name__ == '__main__':
	main()