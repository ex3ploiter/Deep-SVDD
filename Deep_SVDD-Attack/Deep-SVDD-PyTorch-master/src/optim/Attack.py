import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


upper_limit, lower_limit = 1,0


def fgsm(model, inputs, c, epsilon,objective,R):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    delta.requires_grad = True
    
    outputs = model(inputs+delta)
    
    dist = torch.sum((outputs - c) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist
    
    scores.backward()

    # return inputs+epsilon * delta.grad.detach().sign()
    return epsilon * delta.grad.detach().sign()

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def pgd(model, inputs, c, epsilon, alpha, num_iter,objective,R,norm='l_inf'):

    
    delta = torch.zeros_like(inputs).to(device)
    if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
    delta = clamp(delta, lower_limit-inputs, upper_limit-inputs)
    delta.requires_grad = True
    
    for _ in range(num_iter):
        outputs = model(inputs+delta)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
        scores.backward()        
        
        if norm == "l_inf":
                delta.data = torch.clamp(delta + alpha * torch.sign(delta.grad.data), min=-epsilon, max=epsilon)
        delta.data = clamp(delta, lower_limit - inputs, upper_limit - inputs)
        
        delta.grad.zero_()
    
    return delta.detach()
