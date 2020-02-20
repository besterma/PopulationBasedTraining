import pickle
from shutil import copyfile
import numpy as np
import random
import torch
import torch.multiprocessing as _mp
import os
import time
from functools import partial
import gin
import sys
import vae_trainer
import utils
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils import TorchIterableDataset
from scipy import signal

from worker import Worker

import sys
sys.path.append("../disentanglement_lib/disentanglement_lib")
sys.path.append("../beta-tcvae")
from disentanglement_lib.data.ground_truth import named_data
from vae_quant import VAE

mp = _mp.get_context('spawn')


@gin.configurable('explorer',
                  whitelist=['start_epoch', 'max_epoch', 'trainer_class', 'exploit_and_explore_func', 'cutoff'])
class Explorer(mp.Process):
    def __init__(self, population, finish_tasks, device_id, result_dict, random_states, start_epoch, gin_string,
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
        self.dataset_iterator = None
        self.dataset = None

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch.value, "on gpu", self.device_id)
        gin.external_configurable(Adam, module='torch')
        gin.parse_config(self.gin_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        with torch.cuda.device(0):
            self.dataset = named_data.get_named_ground_truth_data()
            self.dataset_iterator = TorchIterableDataset(self.dataset, self.random_state.randint(2**32)) # obsolete?
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
            if self.epoch.value > self.max_epoch:
                trainer = self.trainer_class(device=0,
                                             dataset=None,
                                             random_state=self.random_state,
                                             score_random_seed=7)
                trainer.load_checkpoint(best_model_path)
                trainer.export_best_model(os.path.join(self.model_dir,
                                                       "bestmodels/model.pth"),
                                          dataset=self.dataset)

            for task in tasks:
                score = task.get('score', -1)
                mig = task.get('mig', -1)
                n_active = task.get('n_active', -1)
                print("Put task", task['id'], "in queue with score", score,
                      "mig", mig,
                      "n_active", n_active)
                if self.epoch.value > self.max_epoch:
                    task = {'id': task['id'], 'score': task['score'], 'mig': task['mig']}
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
        with open(os.path.join(self.model_dir, 'parameters/scores.txt'), 'a+') as f:
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

@gin.configurable('simple_explorer',
                  whitelist=['trainer_class', 'exploit_and_explore_func', 'cutoff'])
class SimpleExplorer(mp.Process):
    def __init__(self, is_stop_requested, train_queue, score_queue, finished_queue, gpu_id,
                 result_dict, gin_string, model_dir, random_states, trainer_class,
                 exploit_and_explore_func=gin.REQUIRED,
                 cutoff=gin.REQUIRED):

        print("Init Explorer")
        super().__init__()
        self.is_stop_requested = is_stop_requested
        self.train_queue = train_queue
        self.score_queue = score_queue
        self.finished_queue = finished_queue
        self.gpu_id = gpu_id
        self.result_dict = result_dict
        self.random_state = np.random.RandomState()
        self.gin_config = gin_string
        self.model_dir = model_dir
        self.dataset_path = None
        self.epoch = 0
        self.iteration = 0
        self.exploit_and_explore_func = exploit_and_explore_func
        self.epoch_start_time = 0
        self.trainer_class = trainer_class
        self.cutoff = cutoff


        if 'scores' not in self.result_dict:
            self.result_dict['scores'] = dict()
        if 'parameters' not in self.result_dict:
            self.result_dict['parameters'] = dict()

        self.set_rng_states(random_states)
        self.dataset = None

    def run(self):
        print("Running in loop of explorer in epoch ", self.epoch, "on gpu", self.gpu_id)
        gin.external_configurable(Adam, module='torch')
        gin.parse_config(self.gin_config)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.cuda.device(0):
            print("Explorer init dataset")
            DSPRITES_PATH = os.path.join(
                os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
                "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
            if self.dataset_path is None:
                data = np.load(DSPRITES_PATH, encoding="latin1", allow_pickle=True)
                self.dataset = np.expand_dims(np.array(data["imgs"]).astype(np.float32), axis=1)
                self.dataset_path = os.path.join(self.model_dir, "datasets", "dataset_iteration-000.npy")
                np.save(self.dataset_path, self.dataset)
            else:
                self.dataset = np.load(self.dataset_path)
            print("Explorer dataset initialized")
            while not self.finished_queue.empty():
                self.train_queue.put(self.finished_queue.get())
            self.epoch_start_time = time.time()
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Explorer finishing")
        return

    @torch.no_grad()
    def main_loop(self):
        if self.train_queue.empty() and self.score_queue.empty() and self.finished_queue.full():
            print("One epoch took", time.time() - self.epoch_start_time, "seconds")
            print("Exploit and explore")
            time.sleep(1)  # Bug of not having all tasks in finished_queue
            tasks = []
            while not self.finished_queue.empty():
                tasks.append(self.finished_queue.get())
            tasks = sorted(tasks, key=lambda x: x['score'], reverse=True)
            print("Total #tasks:", len(tasks))
            print('Best score on', tasks[0]['id'], 'is', tasks[0]['score'])
            print('Worst score on', tasks[-1]['id'], 'is', tasks[-1]['score'])
            best_model_path = tasks[0]['model_path']
            try:
                self.exportScores(tasks=tasks)
                self.exportBestModel(best_model_path)
                self.saveModelParameters(tasks=tasks)
                #self.exportBestModelParameters(best_model_path, tasks[0])

            except RuntimeError as err:
                print("Runtime Error in Explorer:", err)
                torch.cuda.empty_cache()
                return 0

            if tasks[0]['score'] > 0.87:
                trainer = self.trainer_class(device=0,
                                             random_state=self.random_state,
                                             score_random_seed=self.random_state.randint(0, 2**32))
                trainer.load_checkpoint(best_model_path)
                export_model_path = os.path.join(self.model_dir,
                                                 "bestmodels",
                                                 "model_iteration{:03d}.pth".format(self.iteration))
                trainer.export_best_model(export_model_path,
                                          dataset=self.dataset)
                if self.iteration == 0:
                    self.reduce_dataset("/home/disentanglement/Python/beta-tcvae/Example_Models/udr_model_xy.pth")
                else:
                    self.reduce_dataset(export_model_path)
                for task in tasks:
                    task['dataset_path'] = self.dataset_path
                self.reset_population(tasks)
                self.epoch = 0
                self.iteration += 1
                print("Explorer: New Iteration", self.iteration)

            elif self.epoch > 1:
                print("Explorer: Reached exit condition")
                trainer = self.trainer_class(device=0,
                                             random_state=self.random_state,
                                             score_random_seed=self.random_state.randint(0, 2**32))
                trainer.load_checkpoint(best_model_path)
                trainer.export_best_model(os.path.join(self.model_dir,
                                                       "bestmodels/model.pth"),
                                          dataset=self.dataset)
                for task in tasks:
                    self.finished_queue.put(task)
                time.sleep(1)
                self.is_stop_requested.value = True
                return 1
            else:
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
                self.epoch += 1
                print("New epoch: ", self.epoch)


            for task in tasks:
                score = task.get('score', -1)
                print("Put task", task['id'], "in queue with score", score)
                self.train_queue.put(task)

            torch.cuda.empty_cache()
            self.epoch_start_time = time.time()
        else:
            print("Already finished:", self.finished_queue.qsize(),
                  "remaining:", str(self.train_queue.qsize() + self.score_queue.qsize()))
        time.sleep(20)
        return 0

    def reduce_dataset(self, model_path):
        print("Explorer reducing dataset")
        self.dataset_path = os.path.join(self.model_dir, "datasets",
                                         "dataset_iteration-{:03d}.npy".format(self.iteration + 1))

        model = VAE(device=0)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        model_function = partial(self._representation_function, model=model)

        batch_size = 32
        train_loader = DataLoader(dataset=self.dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)
        num_samples = len(self.dataset)
        labels = np.zeros((num_samples, model.z_dim))

        kl_divs = np.zeros((len(train_loader), model.z_dim))

        for i, x in enumerate(train_loader):
            means, kl = model_function(x)
            labels[i * batch_size:(i + 1) * batch_size] = means
            kl_divs[i] = kl
        print("")
        kl_means = np.mean(kl_divs, axis=0)

        top_latent = np.argmax(kl_means)
        labels_sorted = np.sort(labels[:, top_latent])
        indices_sorted = np.argsort(labels[:, top_latent])

        derivatives = labels_sorted[1:] - labels_sorted[:-1]
        edge_cutoff = len(derivatives) // 10
        distance = len(derivatives) // 100
        derivatives = derivatives[edge_cutoff:-edge_cutoff]

        peaks, _ = signal.find_peaks(derivatives, height=np.median(derivatives) + np.std(derivatives) * 4.5,
                                     distance=distance)
        if len(peaks) == 0:
            print("No peaks in label derivatives found, this is bad!")
            print("Not updating dataset")
        else:
            inter_peak_median = np.zeros((len(peaks) - 1))
            for j in range(len(peaks) - 1):
                inter_peak_median[j] = np.median(derivatives[peaks[j]:peaks[j + 1]])

            peaks += edge_cutoff
            best_peak = np.argmin(inter_peak_median)
            other_best_peak = np.argmax(np.diff(peaks))

            new_indices = indices_sorted[peaks[other_best_peak]:peaks[other_best_peak+1]]
            self.dataset = self.dataset[new_indices]



        np.save(self.dataset_path, self.dataset)

    def reset_population(self, tasks):
        print("Explorer resetting population")
        for task in tasks:
            checkpoint_path = os.path.join(self.model_dir, "checkpoints/task-%03d.pth" % task['id'])
            os.remove(checkpoint_path)

    def exportScores(self, tasks):
        print("Explorer export scores")
        with open(os.path.join(self.model_dir, 'parameters/scores.txt'), 'a+') as f:
            f.write(str(self.epoch) + '. Epoch Scores:')
            for task in tasks:
                f.write('\n\tId: ' + str(task['id']) + ' - Score: ' + str(task['score']))
            f.write('\n')

            self.result_dict['scores'][self.epoch] = tasks

    def exportBestModel(self, top_checkpoint_path):
        copyfile(top_checkpoint_path,
                 os.path.join(self.model_dir,
                              "bestmodels/model_iteration-{:03d}_epoch-{:03d}.pth".format(self.iteration, self.epoch)))



    def exportBestModelParameters(self, top_checkpoint_path, task):
        print("Explorer export best model parameters")
        checkpoint = torch.load(top_checkpoint_path, map_location=torch.device('cpu'))
        with open(os.path.join(self.model_dir, 'best_parameters.txt'), 'a+') as f:
            f.write("\n\n" + str(self.epoch) + ". Epoch: Score of " + str(task['score']) + " for task " + str(
                task['id']) +
                    " achieved with following parameters:")
            for i in range(self.epoch):
                f.write("\n" + str(checkpoint['training_params'][i]) + str(checkpoint['scores'][i]))

    def saveModelParameters(self, tasks):
        print("Explorer save model parameters")
        temp_dict = dict()
        for task in tasks:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoints/task-{:03d}.pth".format(task['id'])),
                                    map_location=torch.device('cpu'))
            checkpoint_dict = dict()
            checkpoint_dict['training_params'] = checkpoint.get('training_params', None)
            checkpoint_dict['scores'] = checkpoint.get('scores', None)
            temp_dict[task['id']] = checkpoint_dict

        self.result_dict['parameters'][self.epoch] = temp_dict

        pickle_out = open(os.path.join(self.model_dir,
                                       "parameters/parameters-{:03d}-{:03d}.pickle".format(self.iteration, self.epoch)),
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

    @staticmethod
    def _representation_function(x, model):
        """Computes representation vector for input images."""
        # x = np.moveaxis(x, 3, 1)
        x = x.to(0)
        zs, zs_params = model.encode(x)

        means = zs_params[:, :, 0].cpu().detach().numpy()

        def compute_gaussian_kl(z_mean, z_logvar):
            return np.mean(
                0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1),
                axis=0)

        logvars = np.abs(zs_params[:, :, 1].cpu().detach().numpy().mean(axis=0))

        return means, compute_gaussian_kl(means, logvars)
        # return zs.cpu().numpy()                # if we want a sample from the distribution



