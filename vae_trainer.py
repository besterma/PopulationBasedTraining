import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
import time

import sys
import gin
sys.path.append('/home/besterma/ETH/Semester_Thesis/Python/beta-tcvae')
import lib.utils as utils
import lib.datasets as dset
from disentanglement_metrics import metrics_shapes
from udr_metric import udr_metric
from elbo_decomposition import elbo_decomposition


class Trainer(object):
    """Abstract class for trainers."""

    def train(self, epoch, train_steps):
        raise NotImplementedError()

    def save_checkpoint(self, checkpoint_path, random_state):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError()

    def eval(self, epoch):
        raise NotImplementedError()

    def release_memory(self):
        raise NotImplementedError()

    def set_id(self, new_id):
        raise NotImplementedError()


@gin.configurable(blacklist=['score_random_state', 'random_state', 'dataset', 'device'])
class UdrVaeTrainer(Trainer):
    def __init__(self, model_class, optimizer_class, device=None,
                 hyper_params=None, dataset=None, mig_active_factors=None, is_test_run=False,
                 score_random_state=None, score_num_labels=None, random_state=None, epoch_train_steps=737280,
                 batch_size=None, batch_size_init_function=None,
                 beta=None, beta_init_function=None,
                 lr=None, lr_init_function=None):

        # Technically only needed for first time model creation, after that they will be overridden during chkpt loading
        beta = beta_init_function(random_state) if 'beta' in hyper_params else beta
        self.batch_size = batch_size_init_function(random_state) if 'batch_size' in hyper_params else batch_size
        lr = lr_init_function(random_state) if 'lr' in hyper_params else lr

        # Other parameters are supplied by gin config
        self.model = model_class(use_cuda=(device != torch.device('cpu') and device != 'cpu'),  # TODO Might fail
                                 device=device,
                                 beta=beta)
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)
        self.task_id = None
        self.device = device
        self.elbo_running_mean = [utils.RunningAverageMeter() for i in range(5)]
        self.training_params = dict()
        self.scores = dict()
        self.hyper_params = hyper_params
        self.dataset = dataset
        self.mig_active_factors = np.array(mig_active_factors) if mig_active_factors is not None else np.array([0, 1, 2, 3])
        self.score_random_state = score_random_state
        self.score_num_labels = score_num_labels
        self.epoch_train_steps = epoch_train_steps
        self.is_test_run = is_test_run

    def set_id(self, new_id):
        self.task_id = new_id

    def release_memory(self):
        self.model.to_device('cpu')
        del self.dataset

    def save_checkpoint(self, checkpoint_path, random_state):
        print(self.task_id, "trying to save checkpoint")
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size,
                          training_params=self.training_params,
                          scores=self.scores,
                          random_state=random_state)
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished saving checkpoint")

    def load_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to load checkpoint")
        self.elbo_running_mean = [utils.RunningAverageMeter() for i in range(5)]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_hyperparam_state_dict(checkpoint['hyperparam_state_dict'])
        self.model.to_device(self.device)
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.training_params = checkpoint['training_params']
        self.scores = checkpoint["scores"]
        print(self.task_id, "finished loading checkpoint")
        return checkpoint['random_state']

    def save_training_params(self, epoch):
        param_dict = dict(epoch=epoch)
        optim_state_dict = self.optimizer.state_dict()
        if 'lr' in self.hyper_params:
            param_dict['lr'] = optim_state_dict['param_groups'][0].get('lr', "empty")
        if 'batch_size' in self.hyper_params:
            param_dict['batch_size'] = self.batch_size
        if 'beta' in self.hyper_params:
            param_dict['beta'] = self.model.beta

        self.training_params[epoch] = param_dict

    def update_scores(self, epoch, final_score, mig_score, new_mig_score, accuracy, elbo, active_units, n_active, elbo_dict):
        score_dict = dict(epoch=epoch, final_score=final_score, mig=mig_score, new_mig=new_mig_score, mse=accuracy,
                          elbo=elbo, active_units=active_units, n_active=n_active, elbo_dict=elbo_dict)
        self.scores[epoch] = score_dict

    def train(self, epoch):

        start = time.time()
        print(self.task_id, "loading data")
        with torch.cuda.device(self.device):
            train_loader = DataLoader(dataset=self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      num_workers=0,
                                      pin_memory=True)
        print(self.task_id, "finished_loading_data")
        dataset_size = len(train_loader.dataset)
        print(self.task_id, "start training with parameters B", self.model.beta, "lr",
              self.optimizer.param_groups[0]["lr"], "batch size", self.batch_size)
        current_train_steps = 0
        with torch.cuda.device(self.device):
            while current_train_steps < self.epoch_train_steps:
                for i, x in enumerate(train_loader):
                    if current_train_steps >= self.epoch_train_steps:
                        break
                    if current_train_steps + x.size(0) > self.epoch_train_steps:
                        x = x[:self.epoch_train_steps - current_train_steps]
                        current_train_steps = self.epoch_train_steps
                    else:
                        current_train_steps += x.size(0)
                    if current_train_steps % 200000 == 0:
                        print("task", self.task_id, "iteration", current_train_steps, "of", self.epoch_train_steps)

                    if self.is_test_run and current_train_steps % 10000 != 0:
                        continue

                    self.model.train()
                    self.optimizer.zero_grad()
                    x = x.to(device=self.device)
                    x = Variable(x)
                    obj = self.model.elbo(x, dataset_size)
                    if self.model.nan_in_objective(obj):
                        raise ValueError('NaN spotted in objective.')
                    self.model.backward(obj)
                    [self.elbo_running_mean[k].update(obj[k][1].mean().item()) for k in range(5)]
                    self.optimizer.step()

        self.optimizer.zero_grad()
        self.save_training_params(epoch=epoch)
        torch.cuda.empty_cache()
        print("finished training in", time.time() - start, "seconds")
        del train_loader

    @torch.no_grad()
    def eval(self, epoch=0, final=False):
        self.model.eval()
        print(self.task_id, "Evaluate Model with B", self.model.beta, "and running_mean elbo", [self.elbo_running_mean[k].val for k in range(5)])
        start = time.time()
        new_mig_metric, mig_metric, full_mig_metric, full_new_mig_metric, _, _ = metrics_shapes(next(self.model.children()),
                                                                                                self.dataset,
                                                                                                self.device,
                                                                                                self.mig_active_factors,
                                                                                                random_state=self.score_random_state,
                                                                                                num_labels=self.score_num_labels,
                                                                                                num_samples=2048)

        udr_score, n_active = udr_metric(self.model, self.dataset, 'mi', self.batch_size,
                                         self.device, self.score_random_state)

        elbo_dict = dict()
        final_score = udr_score
        combined_full = (full_new_mig_metric + full_mig_metric) / 2
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo",
              [self.elbo_running_mean[k].val for k in range(5)], "got MIG", full_mig_metric, "new MIG", full_new_mig_metric, "and UDR", udr_score,
              "final score:", final_score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        self.update_scores(epoch=epoch, final_score=final_score, mig_score=full_mig_metric, new_mig_score=full_new_mig_metric,
                           accuracy=0, elbo=[self.elbo_running_mean[k].val for k in range(5)], active_units=[],
                           n_active=n_active, elbo_dict=elbo_dict)
        if final:
            return final_score, combined_full, 0, [self.elbo_running_mean[k].val for k in range(5)], [], n_active, elbo_dict
        else:
            return final_score
