
import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as _mp
import torch.optim as optim
from vae_trainer import VAE_Trainer
from utils import get_optimizer, get_model

import sys
sys.path.append('../beta-tcvae')
from vae_quant import VAE
import lib.dist as dist

mp = _mp.get_context('spawn')

np.random.seed(13)

class Worker(mp.Process):
    def __init__(self, batch_size, epoch, max_epoch, population, finish_tasks,
                 device, worker_id, hyperparameters, dataset):
        super().__init__()
        self.epoch = epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.hyperparameters = hyperparameters
        self.orig_batch_size = batch_size
        self.dataset = dataset
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
                if self.epoch.value > self.max_epoch:
                    print("Reached max_epoch in worker")
                    self.population.put(task)
                    break
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
                                           hyper_params=self.hyperparameters,
                                           dataset=self.dataset)
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
                except RuntimeError as err:
                    print("Error:", err)
                    trainer = None
                    del trainer
                    torch.cuda.empty_cache()
                    self.population.put(task)
                    continue