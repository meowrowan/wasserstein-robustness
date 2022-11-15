import argparse
import os
import torchattacks
import torch
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm


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