import numpy as np
import torch
import time
import torch.optim as optim


def get_optimizer(model, optimizer, batch_size, hyperparameters):
    """This is where users choose their optimizer and define the
       hyperparameter space they'd like to search."""
    np.random.seed(int(time.time()))
    optimizer_class = optimizer
    lr = np.random.choice(np.logspace(-5, 0, base=10))
    momentum = np.random.choice(np.linspace(0.1, .9999))
    if hyperparameters['batch_size']:
        batch_size = int(np.random.choice(np.logspace(1, 5, base=2, dtype=int, num=5)))#2,4,8,16,32

    return optimizer_class(model.parameters(), lr=lr), batch_size


def get_model(model_class, use_cuda, z_dim, device_id, prior_dist, q_dist, hyperparameters):
    if hyperparameters['beta']:
        beta = np.random.choice(range(1, 50))
    else:
        beta = 1

    model = model_class(z_dim=10,
                        use_cuda=use_cuda,
                        prior_dist=prior_dist,
                        q_dist=q_dist,
                        beta=beta,
                        tcvae=True, device=device_id)
    return model


def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, hyper_params,
                        perturb_factors=(1.2, 0.8)):
    """Copy parameters from the better model and the hyperparameters
       and running averages from the corresponding optimizer."""
    # Copy model parameters
    print("Running function exploit_and_explore")
    checkpoint = torch.load(top_checkpoint_path)
    state_dict = checkpoint['model_state_dict']
    hyperparam_state_dict = checkpoint['hyperparam_state_dict']
    optimizer_state_dict = checkpoint['optim_state_dict']
    batch_size = checkpoint['batch_size']
    for hyperparam_name in hyper_params['optimizer']:
        perturb = np.random.choice(perturb_factors)
        for param_group in optimizer_state_dict['param_groups']:
            param_group[hyperparam_name] *= perturb
    if hyper_params['batch_size']:
        perturb = np.random.choice(perturb_factors)
        batch_size = int(np.ceil(perturb * batch_size))
    if hyper_params['beta']:
        perturb = np.random.choice(perturb_factors)
        beta = int(np.ceil(perturb * hyperparam_state_dict['beta']))
        hyperparam_state_dict['beta'] = beta
    checkpoint = dict(model_state_dict=state_dict,
                      hyperparam_state_dict=hyperparam_state_dict,
                      optim_state_dict=optimizer_state_dict,
                      batch_size=batch_size)
    torch.save(checkpoint, bot_checkpoint_path)
