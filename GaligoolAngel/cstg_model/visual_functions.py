import torch


def calc_real_weights(hypernetwork_model):
    weights = hypernetwork_model.weights
    real_weights = torch.max(torch.zeros(weights.size()),
                             torch.min(torch.ones(weights.size()), weights))
    return real_weights.detach().numpy()
