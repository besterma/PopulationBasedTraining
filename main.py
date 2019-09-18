import argparse
import pathlib
import numpy as np
import torch
import torch.multiprocessing as _mp
from worker import Worker
from explorer import Explorer
import time
import random
import pickle
import gin
import os

mp = _mp.get_context('spawn')


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
    parser.add_argument("--device", type=str, default='cuda',
                        help="")
    parser.add_argument("--population_size", type=int, default=24,
                        help="")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="")
    parser.add_argument("--worker_size", type=int, default=8,
                        help="number of worker threads, should be a multiple of #graphics cards")
    parser.add_argument("--max_epoch", type=int, default=8,
                        help="")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="define start epoch when continuing training")
    parser.add_argument("--existing_parameter_dict", type=str, default=None,
                        help="Load existing parameter dict to properly extend")
    parser.add_argument("--exp_bonus", action="store_true",
                        help="Give bonus for new number of latent variables")
    parser.add_argument("--partial_mig", type=int, default=15,
                        help="What parts of the mig to use in binary")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="How many labels to use for reduced sample score")
    parser.add_argument("--random_seed", type=int, default=7,
                        help="Initialize random seed")
    parser.add_argument("--gin_config", type=str, default=None)
    return parser


def init_random_state(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    torch.random.manual_seed(random_seed)

@gin.configurable()
def pbt_main(device='cpu', population_size=24, batch_size=20, worker_size=8, max_epoch=10, start_epoch=0,
             existing_parameter_dict=None, partial_mig=15, num_labels=None, random_seed=7):
    print("Lets go!")
    start = time.time()


    # mp.set_start_method("spawn")
    mp = _mp.get_context('forkserver') # Maybe doesnt work
    if not torch.cuda.is_available() and device != 'cpu':
        print("Cuda not available, switching to cpu")
        device = 'cpu'
    population_size = population_size
    batch_size = batch_size
    max_epoch = max_epoch
    worker_size = worker_size
    assert 0 < partial_mig < 16, "partial mig outside range"
    mig_active_factors_binary = [int(x) for x in list('{0:04b}'.format(partial_mig))]
    mig_active_factors = np.array([x for x in range(4) if mig_active_factors_binary[x] == 1])
    print("Using MIG Factors", mig_active_factors, "for this training")
    print("Using", num_labels, "labels for mig estimation")
    print("Population Size:", population_size)
    print("Batch_size:", batch_size)
    print("Worker_size:", worker_size)
    print("Max_epoch", max_epoch)
    print("Start_epoch", start_epoch)
    print("Existing_parameter_dict", existing_parameter_dict)
    print("Random_seed", random_seed)

    random_seed = random_seed

    init_random_state(random_seed)
    torch_limited_labels_rng_state = torch.random.get_rng_state()

    with np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1') as dataset_zip:
        dataset = torch.from_numpy(dataset_zip['imgs']).float()

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
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
    hyper_params = {'optimizer': ["lr"], "batch_size": True, "beta": True}
    train_data_path = test_data_path = './data'
    print("Create workers")
    workers = [Worker(batch_size, epoch, max_epoch, population, finish_tasks, device, i, hyper_params, dataset,
                      mig_active_factors, torch_limited_labels_rng_state, num_labels)
               for i in range(worker_size)]
    workers.append(Explorer(epoch, max_epoch, population, finish_tasks, hyper_params, workers[0].device_id,
                            results, dataset, generate_random_states()))
    print("Start workers")
    [w.start() for w in workers]
    print("Wait for workers to finish")
    [w.join() for w in workers]
    task = []
    while not finish_tasks.empty():
        task.append(finish_tasks.get())
    while not population.empty():
        task.append(population.get())
    task = sorted(task, key=lambda x: x['score'], reverse=True)
    print('best score on', task[0]['id'], 'is', task[0]['score'])
    end = time.time()
    print('Total execution time:', end-start)



if __name__ == "__main__":
    parser = init_argparser()
    args = parser.parse_args()
    if args.gin_config is not None and os.path.isfile(args.gin_config):
        print('Using gin config from', args.gin_config)
        gin.parse_config_file(args.gin_config)
        pbt_main()
    else:
        pbt_main(args.device, args.population_size, args.batch_size, args.worker_size,
                 args.max_epoch, args.start_epoch, args.existing_parameter_dict,
                 args.partial_mig, args.num_labels, args.random_seed)


