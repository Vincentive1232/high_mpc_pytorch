import os
import datetime
import argparse
import numpy as np
from functools import partial

import matplotlib.pyplot as plt

# A Gym style environment
from simulation.dynamic_gap import Dynamic_Gap
from mpc.high_mpc import High_MPC
from simulation.animation import SimVisual
from common import logger_pytorch
from common import utils as U
from policy import high_policy


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics"
                        )
    parser.add_argument('--seed', type=int, default=2349, help="Random seed")
    parser.add_argument('--beta', type=float, default=3.0, help="beta")
    return parser


def main():
    args = arg_parser().parse_args()

    plan_T = 2.0
    plan_dt = 0.04
    so_path = "./mpc/saved/high_mpc.so"

    high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = Dynamic_Gap(high_mpc, plan_T, plan_dt)

    U.set_global_seed(seed=args.seed)


    wml_params = dict(
        sigma0=100,
        max_iter=20,
        n_samples=20,
        beta0=args.beta
    )

    save_dir = U.get_dir(args.save_dir + "/saved_policy")
    save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("highmpc-%m-%d-%H-%M-%S"))

    logger = logger_pytorch.configure(dir=save_dir)
    logger.log("***********************Log & Store Hyper-parameters***********************")
    logger.log("weighted maximum likelihood params")
    logger.log(wml_params)
    logger.log("***************************************************************************")
    high_policy.run_wml(env=env, logger=logger, save_dir=save_dir, **wml_params)


if __name__=="__main__":
    main()



