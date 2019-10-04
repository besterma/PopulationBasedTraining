import pickle
from shutil import copyfile
import numpy as np
import random
import torch
import torch.multiprocessing as _mp
import os
import time
import gin
import sys
import vae_trainer
import utils
from torch.optim import Adam
from utils import TorchIterableDataset

sys.path.append('../beta-tcvae')
from plot_latent_vs_true import plot_vs_gt_shapes

mp = _mp.get_context('spawn')


@gin.configurable('explorer',
                  whitelist=['start_epoch', 'max_epoch', 'trainer_class', 'exploit_and_explore_func', 'cutoff'])
class Explorer(mp.Process):
    def __init__(self, population, finish_tasks, device_id, result_dict, dataset, random_states, start_epoch, gin_string,
                 model_dir, max_epoch=gin.REQUIRED, exploit_and_explore_func=gin.REQUIRED,
                 trainer_class=gin.REQUIRED, cutoff=0.2):
        print("Init Explorer")
        super().__init__()
        self.epoch = start_epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.max_epoch = max_epoch
        self.device_id = device_id
        self.result_dict = result_dict
        self.random_state = np.random.RandomState()
        self.exploit_and_explore_func = exploit_and_explore_func
        self.epoch_start_time = 0
        self.trainer_class = trainer_class
        self.gin_config = gin_string
        self.cutoff = cutoff
        self.model_dir = model_dir

        if 'scores' not in self.result_dict:
            self.result_dict['scores'] = dict()
        if 'parameters' not in self.result_dict:
            self.result_dict['parameters'] = dict()

        self.set_rng_states(random_states)
        self.dataset_iterator = TorchIterableDataset(dataset, self.random_state.randint(2**32))
        self.latent_variable_plotter = LatentVariablePlotter(0, self.dataset_iterator, self.random_state,
                                                             self.trainer_class, model_dir)

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch.value, "on gpu", self.device_id)
        gin.external_configurable(Adam, module='torch')
        gin.parse_config(self.gin_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        with torch.cuda.device(0):
            self.epoch_start_time = time.time()
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Explorer finishing")
        return

    @torch.no_grad()
    def main_loop(self):
        if self.population.empty() and self.finish_tasks.full():
            print("One epoch took", time.time() - self.epoch_start_time, "seconds")
            print("Exploit and explore")
            time.sleep(1)  # Bug of not having all tasks in finish_tasks
            tasks = []
            while not self.finish_tasks.empty():
                tasks.append(self.finish_tasks.get())
            tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
            print("Total #tasks:", len(tasks))
            print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
            print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
            best_model_path = os.path.join(self.model_dir, "checkpoints/task-{:03d}.pth".format(tasks[0]['id']))
            try:
                self.exportScores(tasks=tasks)
                self.exportBestModel(best_model_path)
                #self.latent_variable_plotter.plotLatentBestModel(best_model_path, self.epoch.value, tasks[0]['id'])
                self.saveModelParameters(tasks=tasks)
                self.exportBestModelParameters(best_model_path, tasks[0])

            except RuntimeError as err:
                print("Runtime Error in Explorer:", err)
                torch.cuda.empty_cache()
                return 0

            fraction = self.cutoff
            cutoff = int(np.ceil(fraction * len(tasks)))
            tops = tasks[:cutoff]
            bottoms = tasks[len(tasks) - cutoff:]

            for bottom in bottoms:
                top = self.random_state.choice(tops)
                top_checkpoint_path = os.path.join(self.model_dir, "checkpoints/task-%03d.pth" % top['id'])
                bot_checkpoint_path = os.path.join(self.model_dir, "checkpoints/task-%03d.pth" % bottom['id'])
                self.exploit_and_explore_func(top_checkpoint_path=top_checkpoint_path,
                                              bot_checkpoint_path=bot_checkpoint_path,
                                              random_state=self.random_state)
            with self.epoch.get_lock():
                self.epoch.value += 1
                print("New epoch: ", self.epoch.value)
            for task in tasks:
                score = task.get('score', -1)
                mig = task.get('mig', -1)
                elbo = task.get('elbo', [])
                active_units = task.get('active_units', [])
                n_active = task.get('n_active', -1)
                print("Put task", task['id'], "in queue with score", score,
                      "mig", mig,
                      "n_active", n_active)
                self.population.put(task)

            torch.cuda.empty_cache()
            self.epoch_start_time = time.time()
        else:
            print("Already finished:", self.finish_tasks.qsize(), "remaining:", self.population.qsize())
        if self.epoch.value > self.max_epoch:
            print("Reached max_epoch in explorer")
            time.sleep(1)
            return 1
        time.sleep(20)
        return 0

    def exportScores(self, tasks):
        print("Explorer export scores")
        with open('scores.txt', 'a+') as f:
            f.write(str(self.epoch.value) + '. Epoch Scores:')
            for task in tasks:
                f.write('\n\tId: ' + str(task['id']) + ' - Score: ' + str(task['score']))
            f.write('\n')

            self.result_dict['scores'][self.epoch.value] = tasks

    def exportBestModel(self, top_checkpoint_path):
        copyfile(top_checkpoint_path,
                 os.path.join(self.model_dir, "bestmodels/model_epoch-{:03d}.pth".format(self.epoch.value)))

    def exportBestModelParameters(self, top_checkpoint_path, task):
        print("Explorer export best model parameters")
        checkpoint = torch.load(top_checkpoint_path, map_location=torch.device('cpu'))
        with open(os.path.join(self.model_dir, 'best_parameters.txt'), 'a+') as f:
            f.write("\n\n" + str(self.epoch.value) + ". Epoch: Score of " + str(task['score']) + " for task " + str(
                task['id']) +
                    " achieved with following parameters:")
            for i in range(self.epoch.value):
                f.write("\n" + str(checkpoint['training_params'][i]) + str(checkpoint['scores'][i]))

    def saveModelParameters(self, tasks):
        print("Explorer save model parameters")
        temp_dict = dict()
        for task in tasks:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoints/task-{:03d}.pth".format(task['id'])),
                                    map_location=torch.device('cpu'))
            checkpoint_dict = dict()
            checkpoint_dict['training_params'] = checkpoint['training_params']
            checkpoint_dict['scores'] = checkpoint['scores']
            temp_dict[task['id']] = checkpoint_dict

        self.result_dict['parameters'][self.epoch.value] = temp_dict

        pickle_out = open(os.path.join(self.model_dir, "parameters/parameters-{:03d}.pickle".format(self.epoch.value)),
                          "wb")
        pickle.dump(self.result_dict, pickle_out)
        pickle_out.close()

    def set_rng_states(self, rng_states):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        numpy_rng_state, random_rng_state, torch_cpu_rng_state, torch_gpu_rng_state = rng_states
        self.random_state.set_state(numpy_rng_state)
        random.setstate(random_rng_state)
        torch.cuda.set_rng_state(torch_gpu_rng_state, device=0)
        torch.random.set_rng_state(torch_cpu_rng_state)


class LatentVariablePlotter(object):
    def __init__(self, device_id, dataset, random_state, trainer_class, model_dir):
        self.device_id = device_id
        self.dataset = dataset
        self.random_state = random_state
        self.trainer_class = trainer_class
        self.model_dir = model_dir

    def get_trainer(self):
        trainer = self.trainer_class(device=0,
                                     dataset=self.dataset,
                                     random_state=self.random_state)
        return trainer

    @torch.no_grad()
    def plotLatentBestModel(self, top_checkpoint_name, epoch, task_id):
        with torch.cuda.device(self.device_id):
            print("Plot latents of best model")
            trainer = self.get_trainer()
            trainer.load_checkpoint(top_checkpoint_name)
            for i, model in enumerate(trainer.model.children()):
                plot_vs_gt_shapes(model, self.dataset,
                                  os.path.join(self.model_dir,
                                               "latentVariables/"
                                               "best_epoch_{:03d}_task_{:03d}_model_{:03d}.png".format(epoch,
                                                                                                       task_id, i)),
                                  range(trainer.model.z_dim), self.device_id)
            del trainer
            torch.cuda.empty_cache()
