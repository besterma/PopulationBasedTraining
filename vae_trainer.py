import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
import time
from torch.nn.modules.loss import BCEWithLogitsLoss

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
        self.elbo_running_mean = utils.RunningAverageMeter()
        self.training_params = dict()
        self.hyper_params = hyper_params
        self.loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'


    def set_id(self, num):
        self.task_id = num

    def release_memory(self):
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
            param_dict[hyperparam_name] = optim_state_dict['param_groups'][0].get(hyperparam_name, "empty")
        if self.hyper_params['batch_size']:
            param_dict['batch_size'] = self.batch_size
        if self.hyper_params['beta']:
            param_dict['beta'] = self.model.beta

        self.training_params[epoch] = param_dict

    def get_dataset(self):
        with np.load(self.loc, encoding='latin1') as dataset_zip:
            dataset = torch.from_numpy(dataset_zip['imgs']).float()
        return dataset

    def train(self, epoch, num_subepochs=3):
        print(self.task_id, "loading data")
        dataset = self.get_dataset()
        with torch.cuda.device(self.device):
            train_loader = DataLoader(dataset=dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=True)
        print(self.task_id, "finished_loading_data")
        dataset_size = len(train_loader.dataset)
        print(self.task_id, "start training with parameters B", self.model.beta, "lr",
              self.optimizer.param_groups[0]["lr"], "and dataset_size: ", dataset_size)
        iteration = 0
        start = time.time()
        num_iterations = num_subepochs * dataset_size
        while iteration < num_iterations:
            for i, x in enumerate(train_loader):
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
        train_loader = None
        del train_loader
        dataset = None
        del dataset

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
        accuracy = self.crossEntropyLoss()
        print(self.task_id, "got accuracy", accuracy)
        score, _, _ = mutual_info_metric_shapes(self.model, self.get_dataset(), self.device)
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val, "got MIG", score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        return score.to('cpu').numpy()

    def crossEntropyLoss(self, num_samples = 2048):
        accuracy = 0
        with torch.cuda.device(self.device):
            with torch.no_grad():
                randomSampler = RandomSampler(self.get_dataset(), num_samples = 4096)
                dataLoader = DataLoader(self.get_dataset(), batch_size=64, shuffle=False, num_workers=0,
                                        pin_memory=True, sampler=randomSampler)
                data_size = len(randomSampler)
                for i, x in enumerate(dataLoader):
                    batch_size = x.size(0)
                    x = x.view(batch_size, 1, 64, 64).to(self.device)
                    xs, _, _, _ = self.model.reconstruct_img(x)
                    acc_temp = BCEWithLogitsLoss(xs, x)
                    accuracy += acc_temp * batch_size / data_size
        return accuracy