import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def fgsm(model, inputs, c, epsilon,objective,R):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    
    
    outputs = model(inputs+delta)
    
    dist = torch.sum((outputs - c) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = dist - R ** 2
    else:
        scores = dist
    
    scores.backward()

    # return inputs+epsilon * delta.grad.detach().sign()
    return epsilon * delta.grad.detach().sign()


def pgd(model, inputs, c, epsilon, alpha, num_iter,objective,R):

    delta = torch.zeros_like(inputs, requires_grad=True).to(device)
    for t in range(num_iter):
        

        outputs = model(inputs+delta)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if objective == 'soft-boundary':
            scores = dist - R ** 2
        else:
            scores = dist
        scores.backward()        
        
        
        
        delta.data = (delta + inputs.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
    return delta.detach()
