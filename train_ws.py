"""
Implements WDGRL:
Wasserstein Distance Guided Representation Learning, Shen et al. (2017)
"""
import argparse
import os

arg_parser = argparse.ArgumentParser(description='Domain adaptation using WDGRL')
arg_parser.add_argument('MODEL_FILE', help='A model in trained_models')
arg_parser.add_argument('--batch-size', type=int, default=64)
arg_parser.add_argument('--iterations', type=int, default=500)
arg_parser.add_argument('--epochs', type=int, default=5)
arg_parser.add_argument('--k-critic', type=int, default=5)
arg_parser.add_argument('--k-clf', type=int, default=1)
arg_parser.add_argument('--gamma', type=float, default=10)
arg_parser.add_argument('--wd-clf', type=float, default=1)

arg_parser.add_argument('--model', type=str, default="wideresnet")


args = arg_parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange
import torchvision.transforms as transforms

from pytorchmodels.resnet import resnet20_cifar10_two, resnet56_cifar10_two
from pytorchmodels.wideresnet import wrn28_10_cifar10_two

from utils import loop_iterable, set_requires_grad, GrayscaleToRgb

# Data Preparation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
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


def main(args):
    
    tqdm.write("Preparing Model: {}".format(args.model))
    if args.model == "resnet56":
        clf_model = resnet56_cifar10_two()
    elif args.model == "resnet20":
        clf_model = resnet20_cifar10_two()
    else: # model == wideresnet28-10
        clf_model = wrn28_10_cifar10_two()
    clf_model = clf_model.to(device)
    clf_model.load_state_dict(torch.load(args.MODEL_FILE))

    feature_extractor = clf_model.feature_extractor
    discriminator = clf_model.classifier

    critic = nn.Sequential(
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    
    source_set = CIFAR10(root='/dataset/cifar10', train=True, download=True, transform=transform_train)
    source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    #target_dataset = MNISTM(train=False)
    #target_loader = DataLoader(target_dataset, batch_size=half_batch, drop_last=True,
    #                           shuffle=True, num_workers=0, pin_memory=True)

    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    clf_criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        total_loss = 0
        total_accuracy = 0
        for _ in trange(args.iterations, leave=False):
            (source_x, source_y), (target_x, _) = next(batch_iterator)
            # Train critic
            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(critic, requires_grad=True)

            source_x, target_x = source_x.to(device), target_x.to(device)
            source_y = source_y.to(device)

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

            # Train classifier
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(critic, requires_grad=False)
            for _ in range(args.k_clf):
                source_features = feature_extractor(source_x).view(source_x.shape[0], -1)
                target_features = feature_extractor(target_x).view(target_x.shape[0], -1)

                source_preds = discriminator(source_features)
                clf_loss = clf_criterion(source_preds, source_y)
                wasserstein_distance = critic(source_features).mean() - critic(target_features).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write(f'EPOCH {epoch:03d}: critic_loss={mean_loss:.4f}')
        torch.save(clf_model.state_dict(), 'trained_models/wdgrl.pt')


if __name__ == '__main__':
    main()