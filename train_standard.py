"""
A standard Training for gaining standard accuracy
"""
from __future__ import print_function

import os
import sys
import time
import argparse
import datetime
import math

arg_parser = argparse.ArgumentParser(description='Standard Training')
arg_parser.add_argument('--gpu', type=str, default='0')
arg_parser.add_argument('--model', type=str, default="wideresnet")
arg_parser.add_argument('--batch-size', type=int, default=128)
arg_parser.add_argument('--epochs', type=int, default=200)
arg_parser.add_argument('--only-eval',action="store_true")
arg_parser.add_argument('--ckpt-path', type=str)

#arg_parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')

args = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


from torch.autograd import Variable
from tqdm import tqdm

from pytorchmodels.resnet import resnet20_cifar10_two, resnet56_cifar10_two
from pytorchmodels.wideresnet import wrn28_10_cifar10_two

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device('cuda')

#start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.epochs, args.batch_size, cf.optim_type
num_epochs, batch_size = args.epochs, args.batch_size
best_acc = 0

# Data Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) 

transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

###########################################################

def train(model, epoch, trainloader, trainset, criterion):
    model.train()
    model.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    tqdm.write('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)  # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        tqdm.write('\r')
        tqdm.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))



def eval(model, epoch, testloader, criterion, only_eval=False):
    global best_acc
    model.eval()
    model.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        tqdm.write("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if (not only_eval) and (acc > best_acc): # save best accuracy model
            tqdm.write('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_name = './checkpoint/'+args.model+"_cifar10_best.pt"
            torch.save(model.state_dict(),
                       os.path.join('./checkpoint/', args.model+"_cifar10_best.pt"))
            best_acc = acc



def main():

    # prepare dataloader
    tqdm.write("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='/dataset/cifar10', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='/dataset/cifar10', train=False, download=False, transform=transform_test)
    num_classes = 10
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # model preparation
    tqdm.write("Preparing Model: {}".format(args.model))
    if args.model == "resnet56":
        model = resnet56_cifar10_two()
    elif args.model == "resnet20":
        model = resnet20_cifar10_two()
    else: # model == wideresnet28-10
        model = wrn28_10_cifar10_two()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    if args.only_eval:
        model.load_state_dict(torch.load(args.ckpt_path))
        eval(model, 1, testloader, criterion, args.only_eval)
        return

    # training model
    tqdm.write('\nTraining model')
    tqdm.write('| Training Epochs = ' + str(num_epochs))
    tqdm.write('| Initial Learning Rate = ' + str(args.lr))

    elapsed_time = 0
    for epoch in tqdm(range(1, 1+num_epochs)):

        train(model, epoch, trainloader, trainset, criterion)
        eval(model, epoch, testloader, criterion)

    tqdm.write('\n* Test results : Acc@1 = %.2f%%' %(best_acc))


    # save final model as checkpoint
    torch.save(model.state_dict(),
                       os.path.join('./checkpoint/', args.model+"_cifar10_last_epoch.pt"))



if __name__ == '__main__':    
    main()