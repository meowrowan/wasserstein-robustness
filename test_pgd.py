import argparse
import os
import torchattacks
import torch
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from pytorchmodels.resnet import resnet20_cifar10_two, resnet56_cifar10_two
from pytorchmodels.wideresnet import wrn28_10_cifar10_two

__all__ = ['pgd_test']

def pgd_test(model, test_len, test_loader):
	"""
	test robustness
	# sanity checked: returns the same std. accuracy as original test code
	"""

	atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10) #evaluate on same attack as the student
	model.eval()
	iters = len(test_loader)

	correct = 0
	tot_init_correct = 0
	tot_attack_correct = 0

    # Loop over all examples in test set
	for i, data_pair in enumerate(test_loader):
		data, target = data_pair
		data, target = data.cuda(), target.cuda()

		# Set requires_grad attribute of tensor. Important for Attack
		data.requires_grad = True

		# Forward pass the data through the model
		output = model(data)
		init_pred = output.max(1)[1] # get the index of the max log-probability
		init_result = (init_pred == target)
		init_correct = init_result.nonzero() #indices of correct preds

		# Create adversarial examples
		perturbed_data = atk(data, target)

		# Re-classify the perturbed image
		output = model(perturbed_data)

		# Check for success
		final_pred = output.max(1)[1] # get the index of the max log-probability
		attack_result = (final_pred == target)
		attack_correct = attack_result.nonzero()

		tot_init_correct += len(init_correct)
		tot_attack_correct += len(attack_correct)

		
	init_acc = tot_init_correct / float(test_len) * 100.0
	attack_acc = tot_attack_correct / float(test_len) * 100.0

	tqdm.write(
		"PGD test >> network: [acc: %.4f%%] [robust acc: %.4f%%]"
		% (init_acc, attack_acc)
	)
	return init_acc, attack_acc

parser = argparse.ArgumentParser(description='TEST for PGD')                  
parser.add_argument('--gpu', type=str, default="0",
                    help='gpu_num')
parser.add_argument('--model', type=str, default="wideresnet")
parser.add_argument('--ckpt-path', type=str)

args = parser.parse_args()

# settings
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
torch.manual_seed(57)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def main():
    if args.model == "resnet56":
        model = resnet56_cifar10_two()
    elif args.model == "resnet20":
        model = resnet20_cifar10_two()
    else: # model == wideresnet28-10
        model = wrn28_10_cifar10_two()
    print("model: {}".format(args.model))
    
    model.load_state_dict(torch.load(args.ckpt_path))

    test_set = torchvision.datasets.CIFAR10(root='/data/cifar10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    
    model = model.to(device)
    init_acc, robust_acc = pgd_test(model, len(test_loader.dataset), test_loader) 