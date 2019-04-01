import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import tqdm

import sys
sys.path.append('../beta-tcvae')
import lib.utils as utils


class VAE_Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None, train_loader=None):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        self.train_loader = train_loader
        self.elbo_running_mean = utils.RunningAverageMeter()

    def set_id(self, num):
        self.task_id = num

    def save_checkpoint(self, checkpoint_path):
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size)
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']

    def train(self):
        dataset_size = len(self.train_loader.dataset)
        for i, x in enumerate(self.train_loader):
            self.model.train()
            #anneal_kl(args, vae, iteration)
            self.optimizer.zero_grad()
            x = x.cuda()
            x = Variable(x)
            obj, elbo = self.model.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            obj.mean().mul(-1).backward()
            self.elbo_running_mean.update(elbo.mean().item())
            self.optimizer.step()


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
        #TODO evaluate on test dataset
        return self.elbo_running_mean.val

