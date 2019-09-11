import argparse
import pathlib
import numpy as np
import torch
import torch.multiprocessing as _mp
from worker import Worker
from explorer import Explorer
import time
import random

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
    return parser


def init_random_state(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(random_seed)
    torch.random.manual_seed(random_seed)


if __name__ == "__main__":
    print("Lets go!")
    start = time.time()
    parser = init_argparser()
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
    assert 0 < args.partial_mig < 16, "partial mig outside range"
    mig_active_factors_binary = [int(x) for x in list('{0:04b}'.format(args.partial_mig))]
    mig_active_factors = np.array([x for x in range(4) if mig_active_factors_binary[x] == 1])
    print("Using MIG Factors", mig_active_factors, "for this training")
    print("Using", args.num_labels, "labels for mig estimation")
    print("Population Size:", args.population_size)
    print("Batch_size:", args.batch_size)
    print("Worker_size:", args.worker_size)
    print("Max_epoch", args.max_epoch)
    print("Start_epoch", args.start_epoch)
    print("Existing_parameter_dict", args.existing_parameter_dict)
    print("Random_seed", args.random_seed)

    random_seed = args.random_seed

    init_random_state(random_seed)
    torch_limited_labels_rng_state = torch.random.get_rng_state()

    with np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1') as dataset_zip:
        dataset = torch.from_numpy(dataset_zip['imgs']).float()

    pathlib.Path('checkpoints').mkdir(exist_ok=True)
    checkpoint_str = "checkpoints/task-%03d.pth"
    print("Create mp queues")
    population = mp.Queue(maxsize=population_size)
    finish_tasks = mp.Queue(maxsize=population_size)
    epoch = mp.Value('i', args.start_epoch)
    if args.existing_parameter_dict is None:
        results = dict()
    else:
        results = np.load(args.existing_parameter_dict, allow_pickle=True)
    for i in range(population_size):
        population.put(dict(id=i, score=0, mig=0, accuracy=0,
                            elbo=0, active_units=[], n_active=0,
                            random_states=generate_random_states()))
    hyper_params = {'optimizer': ["lr"], "batch_size": True, "beta": True}
    train_data_path = test_data_path = './data'
    print("Create workers")
    workers = [Worker(batch_size, epoch, max_epoch, population, finish_tasks, device, i, hyper_params, dataset,
                      mig_active_factors, torch_limited_labels_rng_state, args.num_labels)
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


