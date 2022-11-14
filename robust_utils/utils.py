import torch
import torch.nn.functional as F

__all__ = ['generate_perturbed_data']

def generate_perturbed_data(model,
              x_natural,
              y,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10):
    model.eval()
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    # if distance == 'l_inf':
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_ce = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_ce, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv
