import argparse
import pathlib
import numpy as np
import torch
import torch.multiprocessing as _mp
from torch.optim import Adam
from worker import Worker
from explorer import Explorer
import time
import random
import pickle
import gin
import os
import sys
import utils
from vae_trainer import UdrVaeTrainer, VaeTrainer

sys.path.append('../beta-tcvae')
#from vae_quant import VAE, UDRVAE
#import lib.dist as dist

mp = _mp.get_context('spawn')


@gin.configurable(blacklist=['dataset'])
def pbt_main(model_dir, device='cpu', population_size=24, worker_size=8, start_epoch=0,
             existing_parameter_dict=None, random_seed=7, dataset=None):
    print("Lets go!")
    start = time.time()

    # mp.set_start_method("spawn")
    mp = _mp.get_context('forkserver') # Maybe doesnt work
    if not torch.cuda.is_available() and device != 'cpu':
        print("Cuda not available, switching to cpu")
        device = 'cpu'
    population_size = population_size
    worker_size = worker_size
    print("Population Size:", population_size)
    print("Worker_size:", worker_size)
    print("Start_epoch", start_epoch)
    print("Existing_parameter_dict", existing_parameter_dict)
    print("Random_seed", random_seed)

    random_seed = random_seed

    init_random_state(random_seed)
    torch_limited_labels_rng_seed = np.random.randint(2**32)

    if dataset is None:
        with np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1') as dataset_zip:
            dataset = torch.from_numpy(dataset_zip['imgs']).float()

    pathlib.Path(os.path.join(model_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(model_dir, 'bestmodels')).mkdir(parents=True, exist_ok=True)
    # pathlib.Path(os.path.join(model_dir, 'latentVariables')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(model_dir, 'parameters')).mkdir(parents=True, exist_ok=True)

    checkpoint_str = "checkpoints/task-%03d.pth"
    print("Create mp queues")
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)
    epoch = mp.Value('i', start_epoch)
    if existing_parameter_dict is None:
        results = dict()
    else:
        with open(existing_parameter_dict, "rb") as pickle_in:
            results = pickle.load(pickle_in)
    for i in range(population_size):
        population.put(dict(id=i, score=0, mig=0, accuracy=0,
                            elbo=0, active_units=[], n_active=0,
                            random_states=generate_random_states()))
    train_data_path = test_data_path = './data'
    print("Create workers")
    if str(gin.query_parameter("worker.trainer_class")) == "@vae_trainer.VaeTrainer":
        trainer_class = VaeTrainer
    elif str(gin.query_parameter("worker.trainer_class")) == "@vae_trainer.UdrVaeTrainer":
        trainer_class = UdrVaeTrainer
    else:
        trainer_class = VaeTrainer

    workers = [Worker(population, finish_tasks, device, i, dataset, gin_string=gin.config_str(), model_dir=model_dir,
                      score_random_state=torch_limited_labels_rng_seed, start_epoch=epoch, trainer_class=trainer_class)
               for i in range(worker_size)]
    workers.append(Explorer(population, finish_tasks, workers[0].device_id, results, dataset, generate_random_states(),
                            model_dir=model_dir, gin_string=gin.config_str(), start_epoch=epoch,
                            trainer_class=trainer_class))
    print("Start workers")
    [w.start() for w in workers]
    print("Wait for workers to finish")
    [w.join() for w in workers]
    print("Workers and Explorer finished")
    task = []
    time.sleep(1)
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
    end = time.time()
    print('Total execution time:', end-start)
    try:
        if str(gin.query_parameter("worker.trainer_class")) == "@vae_trainer.VaeTrainer":
            score_name = str(gin.query_parameter("vae_trainer.VaeTrainer.eval_function"))
        elif str(gin.query_parameter("worker.trainer_class")) == "@vae_trainer.UdrVaeTrainer":
            score_name = "udr_score"
        if score_name[0] == '@':
            score_name = score_name[1:]
    except ValueError:
        print("score name not found from config file, using default")
        score_name = 'score'


    score_dict = {score_name: task[0]['score'], 'mig': task[0]['mig']}
    return task[0]['id'], score_dict


def generate_random_states():
    random_seed = np.random.randint(low=1, high=2**32-1)
    random_state = np.random.RandomState(random_seed)
    numpy_rng_state = random_state.get_state()
    random.seed(random_seed)
    random_rng_state = random.getstate()
    torch.cuda.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch_cpu_rng_state = torch.random.get_rng_state()
    torch_gpu_rng_state = torch.cuda.get_rng_state()
    return [numpy_rng_state, random_rng_state, torch_cpu_rng_state, torch_gpu_rng_state]


def init_argparser():
    parser = argparse.ArgumentParser(description="Population Based Training")
    parser.add_argument("--gin_config", type=str, default=None, required=True, help='Path to gin config file')
    return parser


def init_random_state(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    torch.random.manual_seed(random_seed)


def init_gin(path_to_config):
    print('Using gin config from', args.gin_config)
    # gin.external_configurable(VAE, module='vae_quant')
    # gin.external_configurable(UDRVAE, module='vae_quant')
    gin.external_configurable(Adam, module='torch')
    gin.parse_config_file(args.gin_config)
    print(gin.operative_config_str())




if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    if os.path.isfile(args.gin_config):
        init_gin(args.gin_config)
        pbt_main()
    else:
        print('Error, gin config', args.gin_config, 'not found')



