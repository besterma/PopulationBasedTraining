import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tqdm

import sys
sys.path.append('../beta-tcvae')
import lib.utils as utils
import lib.datasets as dset
from disentanglement_metrics import mutual_info_metric_shapes


class VAE_Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None, train_loader=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        if (train_loader == 'Shapes'):
            self.train_loader = dset.Shapes()
        else:
            self.train_loader = dset.Shapes()
        self.elbo_running_mean = utils.RunningAverageMeter()

    def set_id(self, num):
        self.task_id = num

    def save_checkpoint(self, checkpoint_path):
        print("trying to save checkpoint")
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size)
        torch.save(checkpoint, checkpoint_path)
        print("finished saving checkpoint")

    def load_checkpoint(self, checkpoint_path):
        print("trying to load checkpoint")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_hyperparam_state_dict(checkpoint['hyperparam_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']
        print("finished loading checkpoint")

    def train(self, epoch, device):
        dataset_size = len(self.train_loader)
        print("start training with parameters B", self.model.beta, "lr",
              self.optimizer.param_groups[0]["lr"], "and dataset_size: ", dataset_size)
        iteration = 0 + epoch*dataset_size
        for i, x in enumerate(self.train_loader):
            iteration += 1
            if iteration % 20000 == 0:
                print("iteration", iteration, "of", dataset_size)
            if iteration % 10 != 0:
                continue
            self.model.train()
            self.optimizer.zero_grad()
            self.anneal_kl('shapes', self.model, iteration)
            x = x.to(device=device)
            x = Variable(x)
            obj, elbo = self.model.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            self.elbo_running_mean.update(elbo.mean().item())
            self.optimizer.step()
        print("finished training")

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
        print("Evaluate Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val)
        score, _, _ = mutual_info_metric_shapes(self.model, self.train_loader)
        print("Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val, "got MIG", score)
        return self.elbo_running_mean.val

