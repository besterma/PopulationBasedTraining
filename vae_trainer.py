import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
import time
from deprecated import deprecated

import sys
import gin
from utils import TorchIterableDataset
sys.path.append('/home/besterma/ETH/Semester_Thesis/Python/beta-tcvae')
sys.path.append('/home/disentanglement/Python/beta-tcvae')
sys.path.append('/home/disentanglement/Python/disentanglement_lib')
import lib.utils as utils
import lib.datasets as dset
from disentanglement_metrics import metrics_shapes
from udr_metric import udr_metric
from elbo_decomposition import elbo_decomposition

from disentanglement_lib.evaluation.metrics.nmig import compute_nmig
from disentanglement_lib.evaluation.metrics.dci import compute_dci

class Trainer(object):
    """Abstract class for trainers."""

    def train(self, epoch, dataset_iterator, random_seed):
        raise NotImplementedError()

    def save_checkpoint(self, checkpoint_path, random_state):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError()

    def export_best_model(self, checkpoint_path, dataset):
        raise NotImplementedError()

    def eval(self, epoch, dataset_iterator, random_seed):
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

    def export_best_model(self, checkpoint_path, dataset):
        """
        export best model out of th UDRVAE according to UDR metric
        :type checkpoint_path: string
        :type dataset: ground_truth_data.GroundTruthData
        """
        print(self.task_id, "trying to export best model")
        udr_scores, _ = udr_metric(models=self.model,
                                   dataset=dataset,
                                   corr_function='mi',
                                   batch_size=128,
                                   device=self.device,
                                   summarize_results=False)
        print(udr_scores)
        udr_scores = udr_scores[~np.eye(udr_scores.shape[0], dtype=bool)].reshape(udr_scores.shape[0], -1)
        print(udr_scores)
        best_model_id = np.argmax(np.median(udr_scores, axis=1))

        for name, module in self.model.named_children():
            if name == str(best_model_id):
                export_model = module
                print("Best model:", name, "type", type(module))
                break

        checkpoint = dict(model_state_dict=export_model.state_dict(),
                          hyperparam_state_dict=export_model.get_hyperparam_state_dict(),
                          batch_size=self.batch_size,
                          training_params=self.training_params,
                          scores=self.scores)
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "saved best model", best_model_id)

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

    @deprecated()
    def update_scores(self, epoch, final_score, mig_score, new_mig_score, accuracy, elbo, active_units, n_active, elbo_dict):
        score_dict = dict(epoch=epoch, final_score=final_score, mig=mig_score, new_mig=new_mig_score, mse=accuracy,
                          elbo=elbo, active_units=active_units, n_active=n_active, elbo_dict=elbo_dict)
        self.scores[epoch] = score_dict

    def train(self, epoch, dataset_iterator, random_seed):
        self.dataset = TorchIterableDataset(dataset_iterator, random_seed)

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
                    [self.elbo_running_mean[k].update(obj[k][1].mean().item()) for k in range(self.model.num_models)]
                    self.optimizer.step()

        self.optimizer.zero_grad()
        self.save_training_params(epoch=epoch)
        torch.cuda.empty_cache()
        print("finished training in", time.time() - start, "seconds")
        del train_loader

    @torch.no_grad()
    def eval(self, epoch=0, dataset_iterator=None, random_seed=7, final=False):
        self.dataset = TorchIterableDataset(dataset_iterator, random_seed)
        self.model.eval()
        print(self.task_id, "Evaluate Model with B", self.model.beta, "and running_mean elbo", [self.elbo_running_mean[k].val for k in range(self.model.num_models)])
        start = time.time()
        udr_score, n_active = udr_metric(self.model, self.dataset, 'mi', 128,
                                         self.device)

        final_score = udr_score
        score_dict = dict(final_score=udr_score, udr_score=udr_score, n_active=n_active, epoch=epoch)
        score_dict['elbo'] = [self.elbo_running_mean[k].val for k in range(self.model.num_models)]
        self.scores[epoch] = score_dict
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo",
              [self.elbo_running_mean[k].val for k in range(self.model.num_models)], "and UDR", udr_score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")

        if final:
            return final_score, [self.elbo_running_mean[k].val for k in range(self.model.num_models)], n_active
        else:
            return final_score


@gin.configurable(blacklist=['score_random_state', 'random_state', 'dataset', 'device'])
class VaeTrainer(Trainer):
    def __init__(self, model_class, optimizer_class, device=None, eval_function=None, eval_combine_function=None,
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
        self.elbo_running_mean = utils.RunningAverageMeter()
        self.training_params = dict()
        self.scores = dict()
        self.hyper_params = hyper_params
        self.dataset = dataset
        self.mig_active_factors = np.array(mig_active_factors) if mig_active_factors is not None else np.array([0, 1, 2, 3])
        self.score_random_state = score_random_state
        self.score_num_labels = score_num_labels
        self.epoch_train_steps = epoch_train_steps
        self.is_test_run = is_test_run
        self.eval_function = eval_function
        self.eval_combine_function = eval_combine_function


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
        self.elbo_running_mean = utils.RunningAverageMeter()
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

    def export_best_model(self, checkpoint_path, dataset):
        print(self.task_id, "trying to export best model")
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
                          batch_size=self.batch_size,
                          training_params=self.training_params,
                          scores=self.scores)
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished exporting best model")

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

    @deprecated()
    def update_scores(self, epoch, final_score, score_dict, accuracy, elbo, active_units, n_active, elbo_dict):
        score_dict = dict(epoch=epoch, final_score=final_score, score_dict=score_dict, mse=accuracy,
                          elbo=elbo, active_units=active_units, n_active=n_active, elbo_dict=elbo_dict)
        self.scores[epoch] = score_dict

    def train(self, epoch, dataset_iterator, random_seed):
        self.dataset = TorchIterableDataset(dataset_iterator, random_seed)

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
                    self.elbo_running_mean.update(obj[1].mean().item())
                    self.optimizer.step()

        self.optimizer.zero_grad()
        self.save_training_params(epoch=epoch)
        torch.cuda.empty_cache()
        print("finished training in", time.time() - start, "seconds")
        del train_loader

    @torch.no_grad()
    def eval(self, epoch=0, dataset_iterator=None, random_seed=7, final=False):
        self.model.eval()
        print(self.task_id, "Evaluate Model with B", self.model.beta, "and running_mean elbo", self.elbo_running_mean.val)
        start = time.time()

        random_state = np.random.RandomState(random_seed)
        def _representation_function(x):
            """Computes representation vector for input images."""
            x = np.moveaxis(x, 3, 1)
            x = torch.from_numpy(x).to(0)
            zs, zs_params = self.model.encode(x)

            return zs_params[:, :, 0].cpu().detach().numpy()  # mean
            # return zs.cpu().numpy()                # if we want a sample from the distribution

        score_dict = self.eval_function(dataset_iterator,
                                        _representation_function,
                                        random_state)
        n_active = 0
        print(self.task_id, score_dict)


        final_score = self.eval_combine_function(score_dict)
        score_dict['final_score'] = final_score
        score_dict['elbo'] = self.elbo_running_mean.val
        score_dict['epoch'] = epoch
        metrics = ['compute_nmig, compute_dci']
        for metric in [compute_nmig, compute_dci]:
            metric_dict = metric(dataset_iterator,_representation_function,random_state)
            for el in metric_dict.keys():
                score_dict[el] = metric_dict[el]
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo",
              self.elbo_running_mean.val, "and Score", final_score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        self.scores[epoch] = score_dict
        if final:
            return final_score, self.elbo_running_mean.val, n_active
        else:
            return final_score
