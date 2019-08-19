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
from disentanglement_metrics import mutual_info_metric_shapes,\
                                    mutual_info_metric_shapes_reduced_y,\
                                    mutual_info_metric_shapes_reduced_scale, \
                                    mutual_info_metric_shapes_reduced_orientation, \
                                    mutual_info_metric_shapes_reduced_y_scale, \
                                    mutual_info_metric_shapes_x_only, \
                                    mutual_info_metric_shapes_scale_only, \
                                    mutual_info_metric_shapes_orientation_only, \
                                    mutual_info_metric_shapes_y_only, \
                                    advanced_mutual_info_metric_shapes, \
                                    combined_mutual_info_metric_shapes
from elbo_decomposition import elbo_decomposition


class VAE_Trainer:

    def __init__(self, model, optimizer, loss_fn=None, train_data=None,
                 test_data=None, batch_size=None, device=None, train_loader=None,
                 hyper_params=None, dataset=None, mig_active_factors=np.array([0, 1, 2, 3]),
                 torch_random_state=None, score_num_labels = 1000):
        """Note: Trainer objects don't know about the database."""

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.task_id = None
        self.device = device
        self.elbo_running_mean = utils.RunningAverageMeter()
        self.training_params = dict()
        self.scores = dict()
        self.hyper_params = hyper_params
        self.dataset = dataset
        self.loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.mig_active_factors = mig_active_factors
        self.torch_random_state = torch_random_state
        self.score_num_labels = score_num_labels


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
                          training_params=self.training_params,
                          scores=self.scores)
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
        original_beta = self.model.beta
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
        iteration = 0
        subepoch = 0
        num_iterations = num_subepochs * dataset_size
        with torch.cuda.device(self.device):
            while iteration < num_iterations:
                print("task", self.task_id, "subepoch", subepoch)
                for i, x in enumerate(train_loader):
                    if iteration % 500000 == 0:
                        print("task", self.task_id, "iteration", iteration, "of", dataset_size)
                    #print("iteration", iteration, "of", dataset_size)
                    #if iteration % 10000 != 0:
                        #iteration += x.size(0)
                        #continue
                    self.model.train()
                    self.optimizer.zero_grad()
                    #self.anneal_kl('shapes', self.model, iteration + epoch * dataset_size)
                    # self.anneal_beta(dataset='shapes',
                    #                  vae=self.model,
                    #                  batch_size=self.batch_size,
                    #                  iteration=iteration,
                    #                  epoch=epoch,
                    #                  orig_beta=original_beta,
                    #                  dataset_size=dataset_size)
                    x = x.to(device=self.device)
                    x = Variable(x)
                    obj, elbo = self.model.elbo(x, dataset_size)
                    if utils.isnan(obj).any():
                        raise ValueError('NaN spotted in objective.')
                    obj.mean().mul(-1).backward()
                    self.elbo_running_mean.update(elbo.mean().item())
                    self.optimizer.step()
                    iteration += x.size(0)
                subepoch += 1
        self.save_training_params(epoch=epoch)
        self.model.beta = original_beta
        torch.cuda.empty_cache()
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

        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration) # 1 --> 0

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

    def eval(self, epoch=0, final = False):
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
        accuracy, active_units, n_active = self.reconstructionError()
        print(self.task_id, "Finished reconstrution + active units")
        new_mig_metric, mig_metric, full_mig_metric, full_new_mig_metric, _, _ = combined_mutual_info_metric_shapes(self.model,
                                                                                                            self.dataset,
                                                                                                            self.device,
                                                                                                            self.mig_active_factors,
                                                                                                            random_state=self.torch_random_state,
                                                                                                            num_labels=self.score_num_labels,
                                                                                                            num_samples=2048)
        new_mig_metric = new_mig_metric.to('cpu').numpy()
        mig_metric = mig_metric.to('cpu').numpy()
        full_mig_metric = full_mig_metric.to('cpu').numpy()
        full_new_mig_metric = full_new_mig_metric.to('cpu').numpy()


        elbo_dict = self.elbo_decomp()
        final_score = (new_mig_metric + mig_metric) / 2 #mig_score + 0.375 * (1 - accuracy * 100)
        combined_full = (full_new_mig_metric + full_mig_metric) / 2
        print(self.task_id, "Model with B", self.model.beta, "and running_mean elbo",
              self.elbo_running_mean.val, "got MIG", full_mig_metric, "new MIG", full_new_mig_metric, "and RL", accuracy,
              "final score:", final_score)
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        self.update_scores(epoch=epoch, final_score=final_score, mig_score=full_mig_metric, new_mig_score=full_new_mig_metric,
                           accuracy=accuracy, elbo=self.elbo_running_mean.val, active_units=active_units,
                           n_active=n_active, elbo_dict=elbo_dict)
        if final:
            return final_score, combined_full, accuracy, self.elbo_running_mean.val, active_units, n_active, elbo_dict
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
        randomSampler = RandomSampler(self.dataset, replacement=True, num_samples=2 ** 18)  # 65536
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