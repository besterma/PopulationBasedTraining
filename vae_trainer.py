import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler
from torch.autograd import Variable
import time
from torch.nn.modules.loss import MSELoss

import sys
sys.path.append('../beta-tcvae')
import lib.utils as utils
import lib.datasets as dset
from disentanglement_metrics import metrics_shapes
from udr_metric import udr_metric
from elbo_decomposition import elbo_decomposition


class VAE_Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None, train_loader=None,
                 hyper_params=None, dataset=None, mig_active_factors=np.array([0, 1, 2, 3]),
                 torch_random_state=None, score_num_labels=None):

        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        self.elbo_running_mean = [utils.RunningAverageMeter() for i in range(5)]
        self.training_params = dict()
        self.scores = dict()
        self.hyper_params = hyper_params
        self.dataset = dataset
        self.mig_active_factors = mig_active_factors
        self.torch_random_state = torch_random_state
        self.score_num_labels = score_num_labels

    def set_id(self, num):
        self.task_id = num

    def release_memory(self):
        self.model.to_device('cpu')
        del self.dataset

    def save_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to save checkpoint")
        checkpoint = dict(model_state_dict=self.model.state_dict(),
                          hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
                          optim_state_dict=self.optimizer.state_dict(),
                          batch_size=self.batch_size,
                          training_params=self.training_params,
                          scores=self.scores)
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished saving checkpoint")

    def load_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to load checkpoint")
        self.elbo_running_mean = [utils.RunningAverageMeter() for i in range(5)]
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_hyperparam_state_dict(checkpoint['hyperparam_state_dict'])
        self.model.to_device(self.device)
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
        self.batch_size = checkpoint['batch_size']
        self.training_params = checkpoint['training_params']
        self.scores = checkpoint["scores"]
        print(self.task_id, "finished loading checkpoint")

    def save_training_params(self, epoch):
        param_dict = dict(epoch=epoch)
        optim_state_dict = self.optimizer.state_dict()
        for hyperparam_name in self.hyper_params['optimizer']:
            param_dict[hyperparam_name] = optim_state_dict['param_groups'][0].get(hyperparam_name, "empty")
        if self.hyper_params['batch_size']:
            param_dict['batch_size'] = self.batch_size
        if self.hyper_params['beta']:
            param_dict['beta'] = self.model.beta

        self.training_params[epoch] = param_dict

    def update_scores(self, epoch, final_score, mig_score, new_mig_score, accuracy, elbo, active_units, n_active, elbo_dict):
        score_dict = dict(epoch=epoch, final_score=final_score, mig=mig_score, new_mig=new_mig_score, mse=accuracy,
                          elbo=elbo, active_units=active_units, n_active=n_active, elbo_dict=elbo_dict)
        self.scores[epoch] = score_dict

    def train(self, epoch, num_subepochs=1):



        start = time.time()
        print(self.task_id, "loading data")
        with torch.cuda.device(self.device):
            train_loader = DataLoader(dataset=self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      pin_memory=True)
        print(self.task_id, "finished_loading_data")
        dataset_size = len(train_loader.dataset)
        print(self.task_id, "start training with parameters B", self.model.beta, "lr",
              self.optimizer.param_groups[0]["lr"], "batch size", self.batch_size)
        subepoch = 0
        with torch.cuda.device(self.device):
            while subepoch < num_subepochs:
                print("task", self.task_id, "subepoch", subepoch)
                self.training_iteration(dataset_size, train_loader)
                subepoch += 1
        self.save_training_params(epoch=epoch)
        torch.cuda.empty_cache()
        print("finished training in", time.time() - start, "seconds")
        del train_loader

    def training_iteration(self, dataset_size, train_loader):
        iteration = 0
        for i, x in enumerate(train_loader):
            iteration += x.size(0)

            #if iteration % 10000 != 0:
            #    continue

            if iteration % 200000 == 0:
                print("task", self.task_id, "iteration", iteration, "of", dataset_size)
            # print("iteration", iteration, "of", dataset_size)

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
            for part_obj, elbo in obj:
                del part_obj
                del elbo
        self.optimizer.zero_grad()

    @staticmethod
    def anneal_kl(self, dataset, vae, iteration):
        if dataset == 'shapes':
            warmup_iter = 7000
        elif dataset == 'faces':
            warmup_iter = 2500
        else:
            warmup_iter = 5000

        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration) # 1 --> 0

    @staticmethod
    def anneal_beta(self, dataset, vae, batch_size, iteration, epoch, orig_beta, dataset_size):
        if dataset == 'shapes':
            warmup_iter_0 = 3 * dataset_size
            warmup_iter_orig = 7 * dataset_size
        elif dataset == 'faces':
            warmup_iter = 2500

        current_overall_iter = iteration + epoch * dataset_size
        if current_overall_iter < warmup_iter_0:
            vae.beta = min(orig_beta, 1 / warmup_iter_0 * current_overall_iter)  # 0 --> 1
        elif current_overall_iter < warmup_iter_orig:
            vae.beta = min(orig_beta, orig_beta / warmup_iter_orig * current_overall_iter)  # 1 --> orig_beta
        else:
            vae.beta = orig_beta

    @torch.no_grad()
    def eval(self, epoch=0, final=False):
        self.model.eval()
        print(self.task_id, "Evaluate Model with B", self.model.beta, "and running_mean elbo", [self.elbo_running_mean[k].val for k in range(5)])
        start = time.time()
        new_mig_metric, mig_metric, full_mig_metric, full_new_mig_metric, _, _ = metrics_shapes(next(self.model.children()),
                                                                                                self.dataset,
                                                                                                self.device,
                                                                                                self.mig_active_factors,
                                                                                                random_state=self.torch_random_state,
                                                                                                num_labels=self.score_num_labels,
                                                                                                num_samples=2048)

        udr_score, n_active = udr_metric(self.model, self.dataset, 'mi', self.batch_size, self.device)

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

    @torch.no_grad()
    def reconstructionError(self, num_samples = 2048):
        VAR_THRESHOLD = 1e-2
        accuracy = 0
        with torch.cuda.device(self.device):
            randomSampler = RandomSampler(self.dataset, replacement=True, num_samples = 2**18) # 262144
            dataLoader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=0,
                                    pin_memory=True, sampler=randomSampler)
            loss = MSELoss()
            data_size = len(randomSampler)
            N = len(dataLoader.dataset)
            K = self.model.z_dim
            Z = self.model.q_dist.nparams
            qz_params = torch.Tensor(N, K, Z).to(self.device)
            for i, x in enumerate(dataLoader):
                batch_size = x.size(0)
                n = dataLoader.batch_size * i
                x = x.view(batch_size, 1, 64, 64).to(self.device)
                xs, _, _, _ = self.model.reconstruct_img(x)
                qz_params[n:n + batch_size] = self.model.encoder.forward(x).\
                                                view(batch_size, K, Z).\
                                                data.to(self.device)
                xs = xs.view(batch_size, -1)
                x = x.view(batch_size, -1)
                acc_temp = loss(xs, x)
                accuracy += acc_temp * batch_size / data_size

            qz_means = qz_params[:,:,0]
            var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
            active_units = torch.arange(0, K)[var > VAR_THRESHOLD].to("cpu").numpy()
            n_active = len(active_units)


        return accuracy.to('cpu').numpy(), active_units, n_active

    @torch.no_grad()
    def elbo_decomp(self):
        randomSampler = RandomSampler(self.dataset, replacement=True, num_samples=2 ** 18, )  # 65536
        dataLoader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=0,
                                pin_memory=True, sampler=randomSampler)
        logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = elbo_decomposition(self.model, dataLoader, self.device)

        return dict(logpx=logpx.to('cpu').numpy(),
                    dependence=dependence.to('cpu').numpy(),
                    information=information.to('cpu').numpy(),
                    dimwise_kl=dimwise_kl.to('cpu').numpy(),
                    analytical_cond_kl=analytical_cond_kl.to('cpu').numpy(),
                    marginal_entropies=marginal_entropies.to('cpu').numpy(),
                    joint_entropy=joint_entropy.to('cpu').numpy(),
                    estimated_elbo=(logpx - analytical_cond_kl).to('cpu').numpy())