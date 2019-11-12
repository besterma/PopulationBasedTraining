import numpy as np
import torch
import gin


@gin.configurable(blacklist=['random_state'])
def get_init_batch_size(random_state):
    return int(random_state.choice(np.logspace(3, 10, base=2, dtype=int, num=8))) # 8 - 1024


@gin.configurable(blacklist=['random_state'])
def get_init_lr(random_state):
    return random_state.choice(np.logspace(-5, 0, num=30, base=10))


@gin.configurable(blacklist=['random_state'])
def get_init_beta(random_state):
    return int(random_state.choice(np.logspace(1, 15, base=1.5, num=24, dtype=int)[1:]))
    # [1:] because else we would have 1 double


@gin.configurable(whitelist=['ratio'])
def mig_nmig_combination(score_dict, ratio=gin.REQUIRED):
    mig = score_dict.get("discrete_mig", .0)
    nmig = score_dict.get("discrete_mig", .0)
    return np.sum([ratio*mig, (1-ratio) * nmig])


@gin.configurable(blacklist=['mig', 'nmig'])
def mig_nmig_only(mig, nmig):
    return np.mean(mig, nmig)


class TorchIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ground_truth_data, random_seed):
        self.random_state = np.random.RandomState(random_seed)
        self.ground_truth_data = ground_truth_data

    def __iter__(self):
        while True:
            x = self.ground_truth_data.sample_observations(1, self.random_state)[0]
            yield np.moveaxis(x, 2, 0)

    def __len__(self):
        return np.prod(self.ground_truth_data.factors_num_values)


"""
def get_optimizer(model, optimizer, batch_size, hyperparameters, random_state):

    optimizer_class = optimizer
    lr = random_state.choice(np.logspace(-5, 0, num=30, base=10))
    # momentum = random_state.choice(np.linspace(0.1, .9999))
    if hyperparameters['batch_size']:
        batch_size = int(random_state.choice(np.logspace(3, 10, base=2, dtype=int, num=8))) # 8 - 2048

    return optimizer_class(model.parameters(), lr=lr), batch_size


def get_model(model_class, use_cuda, z_dim, device_id, prior_dist, q_dist, hyperparameters, random_state):
    if hyperparameters['beta']:
        beta = int(random_state.choice(np.logspace(1,15,base=1.5, num=24, dtype=int)[1:])) #[1:] because else we would have 1 double
    else:
        beta = 1

    model = model_class(z_dim=10,
                        use_cuda=use_cuda,
                        prior_dist=prior_dist,
                        q_dist=q_dist,
                        beta=beta,
                        tcvae=True,
                        conv=True,
                        device=device_id)
    return model
"""


@gin.configurable(whitelist=['hyper_params', 'perturb_factors'])
def exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, hyper_params, random_state,
                        perturb_factors=(2, 1.2, 0.8, 0.5)):
    """Copy parameters from the better model and the hyperparameters
       and running averages from the corresponding optimizer."""
    # Copy model parameters
    print("Running function exploit_and_explore")
    checkpoint = torch.load(top_checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    hyperparam_state_dict = checkpoint['hyperparam_state_dict']
    optimizer_state_dict = checkpoint['optim_state_dict']
    batch_size = checkpoint['batch_size']
    scores = checkpoint['scores']
    model_random_state = checkpoint['random_state']
    training_params = checkpoint['training_params']
    if 'lr' in hyper_params:
        perturb = random_state.choice(perturb_factors)
        for param_group in optimizer_state_dict['param_groups']:
            param_group['lr'] *= perturb
    if 'batch_size' in hyper_params:
        perturb = random_state.choice(perturb_factors)
        batch_size = int(np.minimum(np.ceil(perturb * batch_size), 1024)) #limit due to memory constraints
    if 'beta' in hyper_params:
        perturb = random_state.choice(perturb_factors)
        beta = int(np.ceil(perturb * hyperparam_state_dict['beta']))
        hyperparam_state_dict['beta'] = beta
    checkpoint = dict(model_state_dict=state_dict,
                      hyperparam_state_dict=hyperparam_state_dict,
                      optim_state_dict=optimizer_state_dict,
                      batch_size=batch_size,
                      training_params=training_params,
                      scores=scores,
                      random_state=model_random_state)
    torch.save(checkpoint, bot_checkpoint_path)
