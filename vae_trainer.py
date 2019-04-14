import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

import sys
sys.path.append('../beta-tcvae')
import lib.utils as utils
import lib.datasets as dset
from disentanglement_metrics import mutual_info_metric_shapes


class VAE_Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None, train_loader=None, hyper_params=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        self.train_loader = None
        self.elbo_running_mean = utils.RunningAverageMeter()
        self.training_params = []
        self.hyper_params = hyper_params

    def set_id(self, num):
        self.task_id = num

    def release_memory(self):
        del self.train_loader
        self.model.to_device('cpu')

    def save_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to save checkpoint")
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size,
                          training_params=self.training_params)
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished saving checkpoint")

    def load_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to load checkpoint")
        self.elbo_running_mean = utils.RunningAverageMeter()
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_hyperparam_state_dict(checkpoint['hyperparam_state_dict'])
        self.model.to_device(self.device)
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.training_params = checkpoint['training_params']
        print(self.task_id, "finished loading checkpoint")

    def update_training_params(self, epoch):
        param_dict = dict(epoch=epoch)
        optim_state_dict = self.optimizer.state_dict()
        for hyperparam_name in self.hyper_params['optimizer']:
            param_dict[hyperparam_name] = optim_state_dict['param_groups'][0][hyperparam_name]
        if self.hyper_params['batch_size']:
            param_dict['batch_size'] = self.batch_size
        if self.hyper_params['beta']:
            param_dict['beta'] = self.model.beta

        self.training_params.append(param_dict)

    def train(self, epoch, num_subepochs=3):
        print(self.task_id, "loading data")
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        with np.load(loc, encoding='latin1') as dataset_zip:
            dataset = torch.from_numpy(dataset_zip['imgs']).float()
        self.train_loader = DataLoader(dataset=dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       pin_memory=True)
        print(self.task_id, "finished_loading_data")
        dataset_size = len(self.train_loader.dataset)
        print(self.task_id, "start training with parameters B", self.model.beta, "lr",
              self.optimizer.param_groups[0]["lr"], "and dataset_size: ", dataset_size)
        iteration = 0
        start = time.time()
        num_iterations = num_subepochs * dataset_size
        while iteration < num_iterations:
            for i, x in enumerate(self.train_loader):
                if iteration % 100000 == 0:
                    print("task", self.task_id, "iteration", iteration, "of", dataset_size)
                #print("iteration", iteration, "of", dataset_size)
                #if iteration % 10 != 0:
                 #   iteration += x.size(0)
                  #  continue
                self.model.train()
                self.optimizer.zero_grad()
                #self.anneal_kl('shapes', self.model, iteration + epoch * dataset_size)
                x = x.to(device=self.device)
                x = Variable(x)
                obj, elbo = self.model.elbo(x, dataset_size)
                if utils.isnan(obj).any():
                    raise ValueError('NaN spotted in objective.')
                obj.mean().mul(-1).backward()
                self.elbo_running_mean.update(elbo.mean().item())
                self.optimizer.step()
                iteration += x.size(0)
        self.update_training_params(epoch=epoch)
        print("finished training in", time.time() - start, "seconds")

    def anneal_kl(self, dataset, vae, iteration):
        if dataset == 'shapes':
            warmup_iter = 7000
        elif dataset == 'faces':
            warmup_iter = 2500
        else:
            warmup_iter = 5000

        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)


    def eval(self):
        """
        #Evaluate model on the provided validation or test set.
        self.model.eval()
        dataloader = tqdm.tqdm(DataLoader(self.train_data, self.batch_size, True),
                               desc='Eval (task {})'.format(self.task_id),
                               ncols=80, leave=True)
        correct = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            pred = output.argmax(1)
            correct += pred.eq(y).sum().item()
        accuracy = 100. * correct / (len(dataloader) * self.batch_size)
        return accuracy
        """
        print(self.task_id, "Evaluate Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val)
        start = time.time()
        score, _, _ = mutual_info_metric_shapes(self.model, self.train_loader.dataset, self.device)
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val, "got MIG", score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        return score.to('cpu').numpy()

