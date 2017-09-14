#!/usr/bin/env python
import os, sys
sys.path.append(os.path.abspath(".."))
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
from baselines import logger
import sys
from helper import ei
import tensorflow as tf


def train(env_id, num_timesteps, vis, seed, diff, load_model,fixed_var):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1).__enter__()
    #set_global_seeds(seed)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=300, num_hid_layers=2,gaussian_fixed_var=fixed_var,atoms=11)
    env = ei(vis,seed,diff)
    pposgd_simple.learn(sess, load_model,fixed_var,env,policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=16,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=1e-5, optim_batchsize=1,
            gamma=0.99, lam=0.95, schedule='linear',atoms=11
        )

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='osim')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--vis', help='visualize', action='store_true', default=False)
    parser.add_argument('--fixed_var', help='gaussian_fixed_var', action='store_true', default=False)
    parser.add_argument('--diff', help='difficulty', type=int, default=0)
    parser.add_argument('--load_model', help='load latest model', action='store_true', default=False)
 
    args = parser.parse_args()
    train(args.env, num_timesteps=1e6, vis=args.vis, seed=args.seed, diff=args.diff, load_model=args.load_model,fixed_var=args.fixed_var)


if __name__ == '__main__':
    main()
