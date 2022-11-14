wasserstein-robustness
=============
The purpose of this code is to mitigate the trade-off of robustness and standard accuracy using wasserstein distance & domain adaptation framework.

* * *
## introduction of each folders & files

- in folder 'checkpoint': has checkpoint of models that only contains standard accuracy

    |file name|model name|standard Acc.|
    |---|---|---|
    |resnet20_cifar10_best.pt|ResNet20|92.67%|
    |resnet56_cifar10_best.pt|ResNet56|94.42%|
    |wideresnet_cifar10_best.pt|WideResNet28-10|96.17%|
    other files are checkpoints saved at last evaluation.

- in folder 'checkpoint_final': has checkpoint of models that is the result of our framework

    |file name|model name|standard Acc.| robust Acc.|
    |---|---|---|---|
    |resnet20_cifar10_best.pt|ResNet20|||
    |resnet56_cifar10_best.pt|ResNet56|||
    |wideresnet_cifar10_best.pt|WIdeResNet28-10|||
    other files are checkpoints saved at last evaluation.

- in folder 'pytorchmodels': has model architecture code, all moduled into feature extractor and discriminator
- in folder 'robust_utils': has loss functions used in Adversarial Training (from TRADES, MART paper)
* * *
## Result of our framework
* * *