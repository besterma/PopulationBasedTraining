import pickle
from shutil import copyfile
import numpy as np
import random
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
import torch.optim as optim
from vae_trainer import VAE_Trainer
from utils import get_optimizer, exploit_and_explore, get_model

import sys
sys.path.append('../beta-tcvae')
from vae_quant import VAE
import lib.dist as dist
from plot_latent_vs_true import plot_vs_gt_shapes

import time

mp = _mp.get_context('spawn')

class Explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks, hyper_params,
                 device_id, result_dict, dataset, random_states):
        print("Init Explorer")
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyper_params = hyper_params
        self.device_id = device_id
        self.result_dict = result_dict
        self.dataset = dataset
        self.random_state = np.random.RandomState()

        if 'scores' not in self.result_dict:
            self.result_dict['scores'] = dict()
        if 'parameters' not in self.result_dict:
            self.result_dict['parameters'] = dict()

        self.set_rng_states(random_states)
        self.latent_variable_plotter = LatentVariablePlotter(device_id+1 % torch.cuda.device_count(),
                                                             hyper_params, dataset, self.random_state)

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch.value)
        start = time.time()
        with torch.cuda.device(self.device_id):
            while True:

                if self.population.empty() and self.finish_tasks.full():
                    print("One epoch took", time.time() - start, "seconds")
                    print("Exploit and explore")
                    time.sleep(1)   #Bug of not having all tasks in finish_tasks
                    tasks = []
                    while not self.finish_tasks.empty():
                        tasks.append(self.finish_tasks.get())
                    tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                    print("Total #tasks:", len(tasks))
                    print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                    print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                    best_model_path = "checkpoints/task-{:03d}.pth".format(tasks[0]['id'])
                    try:
                        self.exportScores(tasks=tasks)
                        self.exportBestModel(best_model_path)
                        self.latent_variable_plotter.plotLatentBestModel(best_model_path, self.epoch.value, tasks[0]['id'])
                        self.saveModelParameters(tasks=tasks)
                        self.exportBestModelParameters(best_model_path, tasks[0])
                        fraction = 0.2
                        cutoff = int(np.ceil(fraction * len(tasks)))
                        tops = tasks[:cutoff]
                        bottoms = tasks[len(tasks) - cutoff:]
                    except RuntimeError as err:
                        print("Runtime Error in Explorer:", err)
                        torch.cuda.empty_cache()
                        continue

                    for bottom in bottoms:
                        top = self.random_state.choice(tops)
                        top_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                        bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                        exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, self.hyper_params, self.random_state)

                    with self.epoch.get_lock():
                        self.epoch.value += 1
                        print("New epoch: ", self.epoch.value)
                    for task in tasks:
                        print("Put task", task, "in queue")
                        self.population.put(task)

                    torch.cuda.empty_cache()
                    start = time.time()
                if self.epoch.value > self.max_epoch:
                    print("Reached max_epoch in explorer")
                    time.sleep(1)
                    break
                time.sleep(1)

    def exportScores(self, tasks):
        print("Explorer export scores")
        with open('scores.txt', 'a+') as f:
            f.write(str(self.epoch.value) + '. Epoch Scores:')
            for task in tasks:
                f.write('\n\tId: ' + str(task['id']) + ' - Score: ' + str(task['score']))
            f.write('\n')

            self.result_dict['scores'][self.epoch.value] = tasks

    def exportBestModel(self, top_checkpoint_path):
        copyfile(top_checkpoint_path, "bestmodels/model_epoch-{:03d}.pth".format(self.epoch.value))

    def exportBestModelParameters(self, top_checkpoint_path, task):
        print("Explorer export best model parameters")
        checkpoint = torch.load(top_checkpoint_path)
        with open('best_parameters.txt', 'a+') as f:
            f.write("\n\n" + str(self.epoch.value) + ". Epoch: Score of " + str(task['score']) + " for task " + str(task['id']) +
                    " achieved with following parameters:")
            for i in range(self.epoch.value):
                f.write("\n" + str(checkpoint['training_params'][i]) + str(checkpoint['scores'][i]))

    def saveModelParameters(self, tasks):
        print("Explorer save model parameters")
        temp_dict = dict()
        for task in tasks:
            checkpoint = torch.load("checkpoints/task-{:03d}.pth".format(task['id']))
            checkpoint_dict = dict()
            checkpoint_dict['training_params'] = checkpoint['training_params']
            checkpoint_dict['scores'] = checkpoint['scores']
            temp_dict[task['id']] = checkpoint_dict

        self.result_dict['parameters'][self.epoch.value] = temp_dict

        pickle_out = open("parameters/parameters-{:03d}.pickle".format(self.epoch.value), "wb")
        pickle.dump(self.result_dict, pickle_out)
        pickle_out.close()

    def set_rng_states(self, rng_states):
        numpy_rng_state, random_rng_state, torch_cpu_rng_state, torch_gpu_rng_state = rng_states
        self.random_state.set_state(numpy_rng_state)
        random.setstate(random_rng_state)
        torch.cuda.set_rng_state(torch_gpu_rng_state, device=self.device_id)
        torch.random.set_rng_state(torch_cpu_rng_state)

class LatentVariablePlotter(object):
    def __init__(self, device_id, hyper_params, dataset, random_state):
        self.loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.device_id = device_id
        self.hyper_params = hyper_params
        self.dataset = dataset
        self.random_state = random_state

    def get_trainer(self):
        model = get_model(model_class=VAE,
                          use_cuda=True,
                          z_dim=10,
                          device_id=self.device_id,
                          prior_dist=dist.Normal(),
                          q_dist=dist.Normal(),
                          hyperparameters=self.hyper_params,
                          random_state=self.random_state)
        optimizer, _ = get_optimizer(model, optim.Adam, 16, self.hyper_params, self.random_state)
        trainer = VAE_Trainer(model=model,
                              optimizer=optimizer,
                              loss_fn=nn.CrossEntropyLoss(),
                              batch_size=16,
                              device=self.device_id,
                              hyper_params=self.hyper_params)
        return trainer

    def plotLatentBestModel(self, top_checkpoint_name, epoch, task_id):
        with torch.cuda.device(self.device_id):
            print("Plot latents of best model")
            trainer = self.get_trainer()
            trainer.load_checkpoint(top_checkpoint_name)
            plot_vs_gt_shapes(trainer.model, self.dataset, "latentVariables/best_epoch_{:03d}_task_{:03d}.png".format(epoch, task_id), range(trainer.model.z_dim), self.device_id)
            del trainer
            torch.cuda.empty_cache()
