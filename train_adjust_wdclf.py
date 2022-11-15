"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import argparse
import os
import csv

arg_parser = argparse.ArgumentParser(description='Domain adaptation using WDGRL')
arg_parser.add_argument('--batch-size', type=int, default=128)
arg_parser.add_argument('--iterations', type=int, default=500)
arg_parser.add_argument('--k-critic', type=int, default=5)
arg_parser.add_argument('--k-clf', type=int, default=1)
arg_parser.add_argument('--gamma', type=float, default=10)
########################
arg_parser.add_argument('--model', type=str, default="wideresnet")
arg_parser.add_argument('--gpu', type=str, default="0")
arg_parser.add_argument('--ckpt-path', type=str, default="checkpoint_final")
arg_parser.add_argument('--wd-clf', type=float, default=0.001)
arg_parser.add_argument('--epochs', type=int, default=120)
## adversarial training settings
arg_parser.add_argument('--eps', type=float, default=0.031)
arg_parser.add_argument('--step-size', type=float, default=0.007)
arg_parser.add_argument('--steps', type=int, default=10)
## optim settings
arg_parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
arg_parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
arg_parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
arg_parser.add_argument('--trades', action="store_true")
arg_parser.add_argument('--mart', action="store_true")

## loss settings
args = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# TODO: need modification in other local environments
MODEL_FILE = '/nfs/home/dain0823/wasserstein/framework/checkpoint'

import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from pytorchmodels.resnet import resnet20_cifar10_two, resnet56_cifar10_two
from pytorchmodels.wideresnet import wrn28_10_cifar10_two

from utils import loop_iterable, set_requires_grad
from robust_utils.trades import trades_loss
from robust_utils.mart import mart_loss
from robust_utils.utils import generate_perturbed_data
from pgd_function import pgd_test

torch.manual_seed(1)

# Data Preparation
transform_train = Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    ToTensor(),
    #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])
transform_test = Compose([
    ToTensor(),
    #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_k_critic(epoch):
    """increase critic iteration"""
    k = 1
    if epoch >= 60:
        k = 5
    elif epoch >= 30:
        k = 3
    return k

def adjust_wd_clf(epoch):
    wd_clf = 0.001
    if epoch >= 0.75 * args.epochs:
        wd_clf = 0.1
    elif epoch >= 0.5 * args.epochs:
        wd_clf = 0.01
    return wd_clf


def main():
    
    tqdm.write("Preparing Model: {}".format(args.model))
    if args.model == "resnet56":
        clf_model = resnet56_cifar10_two()
        start_dim = 8192
    elif args.model == "resnet20":
        clf_model = resnet20_cifar10_two()
        start_dim = 8192
    else: # model == wideresnet28-10
        clf_model = wrn28_10_cifar10_two()
        start_dim = 8192
    clf_model = clf_model.to(device)
    clf_model.load_state_dict(torch.load(MODEL_FILE+'/{}_cifar10_best.pt'.format(args.model)))

    feature_extractor = clf_model.feature_extractor
    discriminator = clf_model.discriminator

    critic = nn.Sequential(
        nn.Linear(start_dim, 2048), 
        nn.ReLU(),
        nn.Linear(2048, 320), 
        nn.ReLU(),
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    if not os.path.isdir(args.ckpt_path):
        os.mkdir(args.ckpt_path)

    file_name = "./" + args.ckpt_path + "/{}_accuracy.csv".format(args.model)
    f = open(file_name,'a', newline='')   
    wr = csv.writer(f)
    wr.writerow(["epoch", "standard_acc", "pgd_acc"])


    half_batch = args.batch_size // 2
    
    source_set = CIFAR10(root='/datasets/cifar10', train=True, download=True, transform=transform_train)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    test_set = CIFAR10(root='/datasets/cifar10', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    #target_dataset = MNISTM(train=False)
    #target_loader = DataLoader(target_dataset, batch_size=half_batch, drop_last=True,
    #                           shuffle=True, num_workers=0, pin_memory=True)

    # TODO: need hyperparam setting exp
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    #clf_optim = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    clf_optim = torch.optim.SGD(clf_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)   
    clf_criterion = nn.CrossEntropyLoss()

    best_standard = 0
    best_robust = 0
    init_acc, robust_acc = pgd_test(clf_model, len(test_loader.dataset), test_loader)
    print("Training Starts")

    for epoch in tqdm(range(1, args.epochs+1)):
        #batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))
        adjust_learning_rate(clf_optim, epoch)
        #args.k_critic = adjust_k_critic(epoch)
        args.wd_clf = adjust_wd_clf(epoch)

        total_loss = 0
        total_accuracy = 0
        #for _ in trange(args.iterations, leave=False):
        for batch_idx, (source_x, source_y) in enumerate(source_loader):
            #(source_x, source_y), (target_x, _) = next(batch_iterator)
            
            # Train critic
            clf_model.eval()
            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            #source_x, target_x = source_x.to(device), target_x.to(device)
            #source_y = source_y.to(device)
            source_x, source_y = source_x.to(device), source_y.to(device)
            # make x_adv as target domain
            x_adv = generate_perturbed_data(clf_model, source_x, source_y, args.step_size, args.eps, args.steps)
            target_x = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

            target_x = target_x.to(device)

            with torch.no_grad():
                h_s = feature_extractor(source_x).data.view(source_x.shape[0], -1)
                h_t = feature_extractor(target_x).data.view(target_x.shape[0], -1)
            for _ in range(args.k_critic):
                gp = gradient_penalty(critic, h_s, h_t)

                critic_s = critic(h_s)
                critic_t = critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma*gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()

            # TODO: test
            #tqdm.write("only critic has been trained")
            #init_acc, robust_acc = pgd_test(clf_model, len(test_loader.dataset), test_loader)


            # Train classifier
            clf_model.train()
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                # TODO: we have supervised target dataset. can we advance the framework by using this label info?

                source_output, source_features = feature_extractor(source_x, out_feature=True)
                _, target_features = feature_extractor(target_x, out_feature=True)
                wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

                # caution: this CE loss is already implemented in TRADES loss.
                source_preds = discriminator(source_output)
                #print(source_preds.size())
                #print(source_y.size())

                #clf_loss = torch.sum((source_preds - F.one_hot(source_y)) ** 2, dim=-1).mean()
                
                # TODO: the labels of target - same as source, use in cost func
                if args.mart:
                    clf_target_loss = mart_loss(clf_model, source_x, source_y,
                                                clf_optim,
                                                step_size=args.step_size,
                                                epsilon=args.eps,
                                                perturb_steps=args.steps,
                                                beta=6)
                elif args.trades:
                    clf_target_loss = trades_loss(clf_model, source_x, source_y,
                                                clf_optim,
                                                step_size=args.step_size,
                                                epsilon=args.eps,
                                                perturb_steps=args.steps,
                                                beta=6)
                else:
                    tqdm.write("no loss has been configured. get loss flag")

                #loss = clf_loss + args.wd_clf * wasserstein_distance + clf_target_loss                
                loss = args.wd_clf * wasserstein_distance + clf_target_loss
                #print(clf_loss.size())
                #print(clf_target_loss.size())
                #print(wasserstein_distance.size())
                #loss = args.wd_clf * wasserstein_distance + clf_target_loss + clf_loss
                

                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()
            
            #tqdm.write("both has been trained")
            #init_acc, robust_acc = pgd_test(clf_model, len(test_loader.dataset), test_loader)

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')
        # TODO: if you want to test at different eps settings, you have to modify test_pgd also.
        # TODO: modify autoattack settings if you want above modification
        init_acc, robust_acc = pgd_test(clf_model, len(test_loader.dataset), test_loader)

        # save current best model (standard & robust )
        # standard
        if(robust_acc > 60) and (init_acc > best_standard): # save best accuracy model
            tqdm.write('| Saving Standard Best model...\t\t\tTop1 = %.2f%%' %(init_acc))
            torch.save(clf_model.state_dict(),
                       os.path.join('./'+args.ckpt_path+'/', args.model+"_standard_best.pt"))
            best_standard = init_acc
        
        # robust_acc
        if(robust_acc > best_robust): # save best accuracy model
            tqdm.write('| Saving Robust Best model...\t\t\tTop1 = %.2f%%' %(robust_acc))
            torch.save(clf_model.state_dict(),
                       os.path.join('./'+args.ckpt_path+'/', args.model+"_robust_best.pt"))
            best_robust = robust_acc
        #torch.save(clf_model.state_dict(), './checkpoint_final/ws_final_{}.pt'.format(args.model))
        wr.writerow([epoch, init_acc, robust_acc])
    f.close()




if __name__ == '__main__':
    main()