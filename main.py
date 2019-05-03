import argparse
import os
import pickle
from shutil import copyfile
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from trainer import Trainer
from vae_trainer import VAE_Trainer
from utils import get_optimizer, exploit_and_explore, get_model

import sys
sys.path.append('../beta-tcvae')
from vae_quant import VAE
import lib.datasets as dset
import lib.dist as dist
from plot_latent_vs_true import plot_vs_gt_shapes

import time



mp = _mp.get_context('spawn')

np.random.seed(13)


class Worker(mp.Process):
    def __init__(self, batch_size, epoch, max_epoch, population, finish_tasks,
                 device, worker_id, hyperparameters, result_dict):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyperparameters=hyperparameters
        self.orig_batch_size = batch_size
        self.result_dict = result_dict
        if (device != 'cpu'):
            if (torch.cuda.device_count() > 1):
                #self.device_id = (worker_id) % (torch.cuda.device_count() - 1) + 1 #-1 because we dont want to use card #0 as it is weaker
                self.device_id = (worker_id) % torch.cuda.device_count()
            else:
                self.device_id = 0

    def run(self):
        with torch.cuda.device(self.device_id):
            while True:
                print("Running in loop of worker in epoch ", self.epoch.value, "on gpu", self.device_id)
                if self.epoch.value > self.max_epoch:
                    print("Reached max_epoch in worker")
                    break
                task = self.population.get() # should be blocking for new epoch
                print("working on task", task['id'])
                checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
                model = get_model(model_class=VAE,
                                  use_cuda=True,
                                  z_dim=10,
                                  device_id=self.device_id,
                                  prior_dist=dist.Normal(),
                                  q_dist=dist.Normal(),
                                  hyperparameters=self.hyperparameters)
                optimizer, batch_size = get_optimizer(model, optim.Adam, self.orig_batch_size, self.hyperparameters)
                trainer = VAE_Trainer(model=model,
                                           optimizer=optimizer,
                                           loss_fn=nn.CrossEntropyLoss(),
                                           batch_size=batch_size,
                                           device=self.device_id,
                                           hyper_params=self.hyperparameters)
                trainer.set_id(task['id'])             # double on purpose to have right id as early as possible (for logging)
                if os.path.isfile(checkpoint_path):
                    trainer.load_checkpoint(checkpoint_path)
                # Train
                trainer.set_id(task['id'])
                try:
                    trainer.train(self.epoch.value)
                    score, mig, accuracy, elbo, active_units, n_active = trainer.eval(epoch=self.epoch.value, final=True)
                    trainer.save_checkpoint(checkpoint_path)
                    self.finish_tasks.put(dict(id=task['id'], score=score, mig=mig, accuracy=accuracy, elbo=elbo, active_units=active_units, n_active=n_active))
                    trainer = None
                    del trainer
                    torch.cuda.empty_cache()
                    print("Worker finished one loop")
                except KeyboardInterrupt:
                    break
                except ValueError as err:
                    print("Encountered ValueError, restarting")
                    print("Error: ", err)
                    trainer = None
                    del trainer
                    torch.cuda.empty_cache()
                    self.population.put(task)
                    continue


class Explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks, hyper_params, device_id, result_dict):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyper_params = hyper_params
        self.latent_variable_plotter = LatentVariablePlotter(device_id+1, hyper_params)
        self.device_id = device_id
        self.result_dict = result_dict

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch.value)
        self.result_dict['scores'] = dict()
        self.result_dict['parameters'] = dict()
        start = time.time()
        with torch.cuda.device(self.device_id):
            while True:
                if self.epoch.value > self.max_epoch:
                    print("Reached max_epoch in explorer")
                    break
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
                    self.exportScores(tasks=tasks)
                    self.exportBestModel(best_model_path)
                    self.latent_variable_plotter.plotLatentBestModel(best_model_path, self.epoch.value, tasks[0]['id'])
                    self.saveModelParameters(tasks=tasks)
                    self.exportBestModelParameters(best_model_path, tasks[0])
                    fraction = 0.2
                    cutoff = int(np.ceil(fraction * len(tasks)))
                    tops = tasks[:cutoff]
                    bottoms = tasks[len(tasks) - cutoff:]
                    for bottom in bottoms:
                        top = np.random.choice(tops)
                        top_checkpoint_path = "checkpoints/task-%03d.pth" % top['id']
                        bot_checkpoint_path = "checkpoints/task-%03d.pth" % bottom['id']
                        exploit_and_explore(top_checkpoint_path, bot_checkpoint_path, self.hyper_params)
                    with self.epoch.get_lock():
                        self.epoch.value += 1
                        print("New epoch: ", self.epoch.value)
                    for task in tasks:
                        print("Put task", task, "in queue")
                        self.population.put(task)
                    torch.cuda.empty_cache()
                    start = time.time()
                time.sleep(1)

    def exportScores(self, tasks):
        with open('scores.txt', 'a+') as f:
            f.write(str(self.epoch.value) + '. Epoch Scores:')
            for task in tasks:
                f.write('\n\tId: ' + str(task['id']) + ' - Score: ' + str(task['score']))
            f.write('\n')

            self.result_dict['scores'][self.epoch.value] = tasks

    def exportBestModel(self, top_checkpoint_path):
        copyfile(top_checkpoint_path, "bestmodels/model_epoch-{:03d}.pth".format(self.epoch.value))

    def exportBestModelParameters(self, top_checkpoint_path, task):
        checkpoint = torch.load(top_checkpoint_path)
        with open('best_parameters.txt', 'a+') as f:
            f.write("\n\n" + str(self.epoch.value) + ". Epoch: Score of " + str(task['score']) + " for task " + str(task['id']) +
                    " achieved with following parameters:")
            for i in range(self.epoch.value):
                f.write("\n" + str(checkpoint['training_params'][i]) + str(checkpoint['scores'][i]))

    def saveModelParameters(self, tasks):
        temp_dict = dict()
        for task in tasks:
            checkpoint = torch.load("checkpoints/task-{:03d}.pth".format(task['id']))
            checkpoint_dict = dict()
            checkpoint_dict['training_params'] = checkpoint['training_params']
            checkpoint_dict['scores'] = checkpoint['scores']
            temp_dict[task['id']] = checkpoint_dict

        self.result_dict[self.epoch.value] = temp_dict

        pickle_out = open("parameters-{:03d}.pickle".format(self.epoch.value), "wb")
        pickle.dump(self.result_dict, pickle_out)
        pickle_out.close()


class LatentVariablePlotter(object):
    def __init__(self, device_id, hyper_params):
        self.loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.device_id = device_id
        self.hyper_params = hyper_params

    def get_trainer(self):
        model = get_model(model_class=VAE,
                          use_cuda=True,
                          z_dim=10,
                          device_id=self.device_id,
                          prior_dist=dist.Normal(),
                          q_dist=dist.Normal(),
                          hyperparameters=self.hyper_params)
        optimizer, _ = get_optimizer(model, optim.Adam, 16, self.hyper_params)
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
            with np.load(self.loc, encoding='latin1') as dataset_zip:
                dataset = torch.from_numpy(dataset_zip['imgs']).float()
            plot_vs_gt_shapes(trainer.model, dataset, "latentVariables/best_epoch_{:03d}_task_{:03d}.png".format(epoch, task_id), range(trainer.model.z_dim), self.device_id)
            del dataset
            del trainer
            torch.cuda.empty_cache()



if __name__ == "__main__":
    print("Lets go!")
    start = time.time()
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--device", type=str, default='cuda',
                        help="")
    parser.add_argument("--population_size", type=int, default=10,
                        help="")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="")
    parser.add_argument("--worker_size", type=int, default=3,
                        help="number of worker threads, should be a multiple of #graphics cards")
    parser.add_argument("--max_epoch", type=int, default=8,
                        help="")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="define start epoch when continuing training")

    args = parser.parse_args()
    # mp.set_start_method("spawn")
    mp = mp.get_context('forkserver')
    device = args.device
    if not torch.cuda.is_available():
        device = 'cpu'
    population_size = args.population_size
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    worker_size = args.worker_size

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_str = "checkpoints/task-%03d.pth"
    print("Create mp queues")
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)
    epoch = mp.Value('i', args.start_epoch)
    results = dict()
    for i in range(population_size):
        population.put(dict(id=i, score=0, mig=0, accuracy=0, elbo=0, active_units=[], n_active=0))
    hyper_params = {'optimizer': ["lr"], "batch_size": False, "beta": True}
    train_data_path = test_data_path = './data'
    print("Create workers")
    workers = [Worker(batch_size, epoch, max_epoch, population, finish_tasks, device, i, hyper_params)
               for i in range(worker_size)]
    workers.append(Explorer(epoch, max_epoch, population, finish_tasks, hyper_params, workers[0].device_id))
    [w.start() for w in workers]
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
    best_model_path = "checkpoints/task-%03d.pth" % task[0]['id']
    workers[-1].exportScores(task)
    workers[-1].exportBestModel(best_model_path, epoch.value+1)
    workers[-1].exportBestModelParameters(best_model_path, task[0])
    workers[-1].latent_variable_plotter.plotLatentBestModel(best_model_path)
    end = time.time()

    print('Total execution time:', start-end)
