import argparse
import os
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

import time



mp = _mp.get_context('spawn')


class Worker(mp.Process):
    def __init__(self, batch_size, epoch, max_epoch, population, finish_tasks,
                 device, worker_id, hyperparameters):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.device = device
        self.hyperparameters=hyperparameters
        if (device != 'cpu'):
            if (torch.cuda.device_count() > 1):
                #self.device_id = (worker_id) % (torch.cuda.device_count() - 1) + 1 #-1 because we dont want to use card #0 as it is weaker
                self.device_id = (worker_id) % torch.cuda.device_count()
            else:
                self.device_id = 0
        self.model = get_model(model_class=VAE,
                               use_cuda=True,
                               z_dim=10,
                               device_id=self.device_id,
                               prior_dist=dist.Normal(),
                               q_dist=dist.Normal(),
                               hyperparameters=self.hyperparameters)
        self.optimizer, self.batch_size = get_optimizer(self.model, optim.Adam)
        self.trainer = VAE_Trainer(model=self.model,
                                   optimizer=self.optimizer,
                                   loss_fn=nn.CrossEntropyLoss(),
                                   batch_size=self.batch_size,
                                   device=self.device)

    def run(self):
        while True:
            print("Running in loop of worker in epoch ", self.epoch.value, "on gpu", self.device_id)
            if self.epoch.value > self.max_epoch:
                print("Reached max_epoch in worker")
                break
            # Train
            task = self.population.get()
            self.trainer.set_id(task['id'])
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            if os.path.isfile(checkpoint_path):
                self.trainer.load_checkpoint(checkpoint_path)
            try:
                self.trainer.train(self.epoch.value, self.device_id)
                score = self.trainer.eval()
                self.trainer.save_checkpoint(checkpoint_path)
                self.finish_tasks.put(dict(id=task['id'], score=score))
                print("Worker finished one loop")
            except KeyboardInterrupt:
                break
            except ValueError:
                self.population.put(task)
                print("Encountered ValueError, restarting")
                self.model = get_model(model_class=VAE,
                                       use_cuda=True,
                                       z_dim=10,
                                       device_id=self.device_id,
                                       prior_dist=dist.Normal(),
                                       q_dist=dist.Normal(),
                                       hyperparameters=self.hyperparameters)
                self.optimizer, self.batch_size = get_optimizer(self.model, optim.Adam)
                self.trainer = VAE_Trainer(model=self.model,
                                           optimizer=self.optimizer,
                                           loss_fn=nn.CrossEntropyLoss(),
                                           batch_size=self.batch_size,
                                           device=self.device)
                continue


class Explorer(mp.Process):
    def __init__(self, epoch, max_epoch, population, finish_tasks, hyper_params):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyper_params = hyper_params

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch.value)
        while True:
            if self.epoch.value > self.max_epoch:
                print("Reached max_epoch in explorer")
                break
            if self.population.empty() and self.finish_tasks.full():
                print("Exploit and explore")
                tasks = []
                while not self.finish_tasks.empty():
                    tasks.append(self.finish_tasks.get())
                tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
                print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
                print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
                self.exportScores(tasks=tasks)
                self.exportBestModel("task-%03d.pth" % tasks[0]['id'], self.epoch.value)
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
                for task in tasks:
                    self.population.put(task)
            time.sleep(1)

    def exportScores(self, tasks):
        with open('scores.txt', 'a+') as f:
            f.write(self.epoch.value + '. Epoch Scores: \n')
            for task in tasks:
                f.write('\tId: ' + task['id'] + ' - Score: ' + task['score'])

    def exportBestModel(self, top_checkpoint_name, id):
        copyfile("checkpoints/"+top_checkpoint_name, "bestmodels/model_epoch-%03d.pth" % id)


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
    epoch = mp.Value('i', 0)
    for i in range(population_size):
        population.put(dict(id=i, score=0))
    hyper_params = {'optimizer': ["lr"], "batch_size": False, "beta": True}
    train_data_path = test_data_path = './data'
    print("Create workers")
    workers = [Worker(batch_size, epoch, max_epoch, population, finish_tasks, device, i, hyper_params)
               for i in range(worker_size)]
    workers.append(Explorer(epoch, max_epoch, population, finish_tasks, hyper_params))
    [w.start() for w in workers]
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
    workers[-1].exportScores(task)
    workers[-1].exportBestModel("task-%03d.pth" % task[0]['id'], epoch.value+1)
    end = time.time()

    print('Total execution time:', start-end)
