from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env
from model import ActorCritic
from test import test
from train import train

import gym


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')

# デフォルト値はこちらに記載、基本的にはデフォルトのままで学習可能です
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--num-processes', type=int, default=16,
                    help='how many training processes to use (default: 16)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')

# ハイパーパラメータ系、学習が上手くいったら実験で変えてみましょう
# 学習率(learning rate)
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
# 学習率(learning rate)
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
# added from here
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy bonus coefficient (default: 0.01)')
# added till here
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # 追加！

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = create_atari_env(args.env_name)
    # print(f"env.observation_space.shape[0]: {env.observation_space.shape[0]}")
    print(f"env.observation_space.shape: {env.observation_space.shape}")
    print(f"env action meanings: {env.unwrapped.get_action_meanings()}")
    print(f"num of processes: {args.num_processes}")

    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    #optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer = my_optim.SharedRMSprop(shared_model.parameters(), lr=args.lr) # RMSPropのηを指定
    optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
