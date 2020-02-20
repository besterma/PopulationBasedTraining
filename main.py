import argparse
import pathlib
import numpy as np
import torch
import torch.multiprocessing as _mp
from torch.optim import Adam

from explorer import SimpleExplorer
from worker import SimpleWorker
import time
import random
import pickle
import gin
import os
import sys
import utils
from vae_trainer import UdrVaeTrainer, VaeTrainer, DummyTrainer, SimpleUdrVaeTrainer

sys.path.append('../beta-tcvae')
#from vae_quant import VAE, UDRVAE
#import lib.dist as dist

mp = _mp.get_context('spawn')


@gin.configurable()
def pbt_main(model_dir, device='cpu', population_size=24, worker_size=8, start_epoch=0,
             existing_parameter_dict=None, random_seed=7):
    print("Lets go!")
    start = time.time()

    # mp.set_start_method("spawn")
    #mp = _mp.get_context('forkserver') # Maybe doesnt work
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
    score_random_seed = np.random.randint(2**32)


    pathlib.Path(os.path.join(model_dir, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(model_dir, 'bestmodels')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(model_dir, 'datasets')).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(model_dir, 'parameters')).mkdir(parents=True, exist_ok=True)

    checkpoint_str = "checkpoints/task-%03d.pth"
    init_dataset_path = os.path.join(model_dir, "datasets", "dataset_iteration-000.npy")
    print("Create mp queues")
    train_queue = mp.Queue(maxsize=population_size)
    score_queue = mp.Queue(maxsize=population_size)
    finished_queue = mp.Queue(maxsize=population_size)
    is_stop_requested = mp.Value('i', False, lock=False)
    if existing_parameter_dict is None:
        results = dict()
    else:
        with open(existing_parameter_dict, "rb") as pickle_in:
            results = pickle.load(pickle_in)
    for i in range(population_size):
        finished_queue.put(dict(id=i,
                                dataset_path=init_dataset_path,
                                model_path=os.path.join(model_dir, "checkpoints", "task-{:03d}.pth".format(i)),
                                random_states=generate_random_states()))
    print("Create workers")
    if str(gin.query_parameter("simple_worker.trainer_class")) == "@vae_trainer.VaeTrainer":
        trainer_class = VaeTrainer
    elif str(gin.query_parameter("simple_worker.trainer_class")) == "@vae_trainer.UdrVaeTrainer":
        trainer_class = UdrVaeTrainer
    elif str(gin.query_parameter("simple_worker.trainer_class")) == "@vae_trainer.SimpleUdrVaeTrainer":
        trainer_class = SimpleUdrVaeTrainer
    else:
        trainer_class = VaeTrainer

    allowed_gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    all_ids = list(map(lambda id: str(id), range(torch.cuda.device_count())))
    allowed_gpu_ids = all_ids if allowed_gpu_ids is None else allowed_gpu_ids.split(",")

    workers = [SimpleWorker(is_stop_requested,
                            train_queue,
                            score_queue,
                            finished_queue,
                            gpu_id=allowed_gpu_ids[i % len(allowed_gpu_ids)],
                            worker_id=i,
                            gin_string=gin.config_str(),
                            trainer_class=trainer_class)
               for i in range(worker_size)]
    workers.append(SimpleExplorer(is_stop_requested,
                                  train_queue,
                                  score_queue,
                                  finished_queue,
                                  gpu_id=allowed_gpu_ids[0],
                                  result_dict=results,
                                  gin_string=gin.config_str(),
                                  model_dir=model_dir,
                                  random_states=generate_random_states(),
                                  trainer_class=trainer_class))

    print("Start workers")
    [w.start() for w in workers]
    print("Wait for workers to finish")
    [w.join() for w in workers]
    print("Workers and Explorer finished")
    task = []
    time.sleep(1)
    while not finished_queue.empty():
        task.append(finished_queue.get())
    while not train_queue.empty():
        task.append(train_queue.get())
    while not score_queue.empty():
        task.append(score_queue.get())
    finished_queue.close()
    train_queue.close()
    score_queue.close()
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

    score_dict = {score_name: task[0]['score']}
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



