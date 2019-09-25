import os
import time
import numpy as np
import random
import torch
import torch.multiprocessing as _mp
import gin
import vae_trainer
import utils

mp = _mp.get_context('spawn')


@gin.configurable('worker', whitelist=['max_epoch', 'trainer_class'])
class Worker(mp.Process):
    def __init__(self, population, finish_tasks, device, worker_id, dataset, start_epoch,
                 max_epoch=gin.REQUIRED, trainer_class=gin.REQUIRED, score_random_state=None):
        super().__init__()
        print("Init Worker")
        self.epoch = start_epoch
        self.max_epoch = max_epoch
        self.population = population
        self.finish_tasks = finish_tasks
        self.dataset = dataset
        self.worker_id = worker_id
        self.score_random_state = score_random_state
        self.random_state = np.random.RandomState()
        self.trainer_class = trainer_class

        np.random.seed(worker_id)
        if device != 'cpu':
            if torch.cuda.device_count() > 1:
                # self.device_id = (worker_id) % (torch.cuda.device_count() - 1) + 1 #-1 because we dont want to use card #0 as it is weaker
                self.device_id = worker_id % torch.cuda.device_count()
            else:
                self.device_id = 0
        else:
            self.device_id = 'cpu'

    def run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with torch.cuda.device(0):
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Worker", self.worker_id, "finishing")
        return

    def main_loop(self):
        print("Worker", self.worker_id, "Running in loop of worker in epoch ", self.epoch.value, "on gpu",
              self.device_id)
        print(self.population.qsize(), "tasks left until new epoch")
        if self.epoch.value > self.max_epoch:
            print("Worker", self.worker_id, "Reached max_epoch in worker")
            return 1
        task = self.population.get()  # should be blocking for new epoch
        if self.epoch.value > self.max_epoch:
            print("Worker", self.worker_id, "Reached max_epoch in worker")
            self.population.put(task)
            return 1
        print("Worker", self.worker_id, "working on task", task['id'])
        try:
            checkpoint_path = "checkpoints/task-%03d.pth" % task['id']
            self.set_rng_states(task["random_states"])

            trainer = self.trainer_class(device=0,
                                         dataset=self.dataset,
                                         random_state=self.random_state,
                                         score_random_state=self.score_random_state)
            trainer.set_id(task['id'])  # double on purpose to have right id as early as possible (for logging)
            if os.path.isfile(checkpoint_path):
                random_states = trainer.load_checkpoint(checkpoint_path)
                self.set_rng_states(random_states)

            # Train
            trainer.set_id(task['id'])
            trainer.train(self.epoch.value)
            score, mig, accuracy, elbo, active_units, n_active, elbo_dict = trainer.eval(epoch=self.epoch.value,
                                                                                         final=True)
            trainer.save_checkpoint(checkpoint_path, self.get_rng_states())
            self.finish_tasks.put(dict(id=task['id'], score=score, mig=mig, accuracy=accuracy,
                                       elbo=elbo, active_units=active_units, n_active=n_active,
                                       random_states=self.get_rng_states()))
            print("Worker", self.worker_id, "finished task", task['id'])
            trainer.release_memory()
            torch.cuda.empty_cache()
            return 0

        except KeyboardInterrupt:
            return -1

        except ValueError as err:
            nr_value_errors = task.get('nr_value_errors', 0)
            nr_value_errors += 1

            if nr_value_errors >= 10:
                print("Worker", self.worker_id, "Task", task['id'], "Encountered ValueError", nr_value_errors,
                      ", giving up, evaluating")
                try:
                    score, mig, accuracy, elbo, active_units, n_active, elbo_dict = trainer.eval(
                        epoch=self.epoch.value, final=True)
                except Exception:
                    score = task.get('score', -1)
                    mig = task.get('mig', 0)
                    accuracy = task.get('accuracy', 0)
                    elbo = task.get('elbo', 0)
                    active_units = task.get('active_units', [])
                    n_active = task.get('n_active', 0)
                random_states = task.get('random_states', [])
                trainer.save_checkpoint(checkpoint_path, self.get_rng_states())
                self.finish_tasks.put(dict(id=task['id'], score=score, mig=mig, accuracy=accuracy, elbo=elbo,
                                           active_units=active_units, n_active=n_active,
                                           random_states=random_states))
                print(task['id'], "with too many ValueErrors finished")
            else:
                print("Worker", self.worker_id, "Task", task['id'], "Encountered ValueError", nr_value_errors,
                      ", restarting")
                task["nr_value_errors"] = nr_value_errors
                task['random_states'] = self.get_rng_states()
                self.population.put(task)

                print("Worker", self.worker_id, "put task", task['id'], 'back into population')
            trainer.release_memory()
            torch.cuda.empty_cache()
            return 0

        except RuntimeError as err:
            print("Worker", self.worker_id, "Runtime Error:", err)
            if trainer is not None:
                trainer.release_memory()
            torch.cuda.empty_cache()
            self.population.put(task)
            print("Worker", self.worker_id, "put task", task['id'], 'back into population')
            time.sleep(10)
            return 0

    def set_rng_states(self, rng_states):
        numpy_rng_state, random_rng_state, torch_cpu_rng_state, torch_gpu_rng_state = rng_states
        self.random_state.set_state(numpy_rng_state)
        random.setstate(random_rng_state)
        torch.cuda.set_rng_state(torch_gpu_rng_state, device=0)
        torch.random.set_rng_state(torch_cpu_rng_state)

    def get_rng_states(self):
        numpy_rng_state = self.random_state.get_state()
        random_rng_state = random.getstate()
        torch_cpu_rng_state = torch.random.get_rng_state()
        torch_gpu_rng_state = torch.cuda.get_rng_state()
        return [numpy_rng_state, random_rng_state, torch_cpu_rng_state, torch_gpu_rng_state]
