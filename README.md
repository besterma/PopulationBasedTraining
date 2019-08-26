### PBT: Population Based Training

[Population Based Training of Neural Networks, Jaderberg et al. @ DeepMind](https://arxiv.org/abs/1711.09846)

A simple PyTorch implementation of PBT.

### What this code is for

Finding a good hyperparameter schedule.

### How does PBT work?
PBT trains each model partially and assesses them on the validation set. It then transfers the parameters and hyperparameters from the top performing models to the bottom performing models (exploitation). After transferring the hyperparameters, PBT perturbs them (exploration). Each model is then trained some more, and the process repeats. This allows PBT to learn a hyperparameter schedule instead of only a fixed hyperparameter configuration. PBT can be used with different selection methods (e.g. different ways of defining "top" and "bottom" (e.g. top 5, top 5%, etc.)).

For more information, see [the paper](https://arxiv.org/abs/1711.09846) or [blog post](https://deepmind.com/blog/population-based-training-neural-networks/).

### Requirements
- PyTorch >= 1.0.0

### Usage
`$ unbuffer python3 main.py --population_size 80 --worker_size 8 --max_epoch 10 --start_epoch 0 --partial_mig 15 --num_labels 100 2>&1 | tee results34_execution_4_ep_80_pop_conv_random_7.log`
