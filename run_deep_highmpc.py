import os
import time
import datetime
import argparse
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import meshcat

from simulation.dynamic_gap import Dynamic_Gap
from mpc.high_mpc import High_MPC
from simulation.animation import SimVisual
from simulation.animation_meshcat import Sim_Visual_Meshcat
from common import logger_pytorch
from common import utils as U
from policy import deep_high_policy


#
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, default=0, help="0 - Data collection; 1 - train the deep high-level policy; 2 - test the trained policy.")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    parser.add_argument('--load_dir', type=str, help="Directory where to load weights")
    return parser


def run_deep_high_mpc(env, actor_params, load_dir):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = deep_high_policy.Actor(obs_dim, act_dim)
    actor.load_weights(load_dir)
    actor.eval()

    ep_len, ep_reward = 0, 0
    for i in range(10):
        obs = env.reset()
        t = 0
        while t < env.sim_T:
            t += env.sim_dt

            obs_tmp = np.reshape(obs, (1, -1))
            obs_tensor = torch.from_numpy(obs_tmp).float()

            act_tensor = actor(obs_tensor)
            act = act_tensor.detach().cpu().numpy()[0]
            # act = actor(obs_tmp).numpy()[0]

            # execute action
            next_obs, reward, _, info = env.step(act)

            #
            obs = next_obs
            ep_reward += reward

            #
            ep_len += 1

            #
            update = False
            if t >= env.sim_T:
                update = True
            yield [info, t, update]


def run_deep_high_mpc_meshcat(env, actor_params, load_dir, vis_env):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = deep_high_policy.Actor(obs_dim, act_dim)
    actor.load_weights(load_dir)
    actor.eval()

    ep_len, ep_reward = 0, 0
    for i in range(10):
        obs = env.reset()
        t = 0
        while t < env.sim_T:
            t += env.sim_dt

            obs_tmp = np.reshape(obs, (1, -1))
            obs_tensor = torch.from_numpy(obs_tmp).float()

            act_tensor = actor(obs_tensor)
            act = act_tensor.detach().cpu().numpy()[0]
            # act = actor(obs_tmp).numpy()[0]

            # execute action
            next_obs, reward, _, info = env.step(act)

            #
            obs = next_obs
            ep_reward += reward
            vis_env.update(env.pend_state[0], env.quad_state[0:3], env.quad_state[3:7])

            #
            ep_len += 1

            #
            update = False
            if t >= env.sim_T:
                update = True
            yield [info, t, update]


def main():
    args = arg_parser().parse_args()

    plan_T = 2.0    # Prediction horizon for MPC and local planner
    plan_dt = 0.04  # Prediction horizon for MPC and local planner
    so_path = "./mpc/saved/high_mpc.so"  # saved high mpc model (casadi code generation)

    #
    high_mpc = High_MPC(T=plan_T, dt=plan_dt, so_path=so_path)
    env = Dynamic_Gap(high_mpc, plan_T, plan_dt)

    #
    actor_params = dict(
        hidden_units = [32, 32],
        learning_rate = 1e-4,
        activation = 'relu',
        train_epoch = 1000,
        batch_size = 128
    )

    #
    training_params = dict(
        max_samples = 5000,
        max_wml_iter = 15,
        beta0 = 3.0,
        n_samples = 15
    )

    # if in training mode, create new dir to save the model and checkpoints

    if args.option == 0: # collection data
        #
        save_dir = U.get_dir(args.save_dir + "/Dataset")
        save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("deep_highmpc-%m-%d-%H-%M-%S"))
        logger = logger_pytorch.configure(dir=save_dir)

        logger.log("**********************************Log & Store Hyper-parameters**********************************")
        logger.log("actor params")
        logger.log(actor_params)
        logger.log("training params")
        logger.log(training_params)
        logger.log("************************************************************************************************")

        #
        deep_high_policy.data_collection(env=env, logger=logger, save_dir=save_dir, **training_params)
    elif args.option == 1: # train the policy
        data_dir = args.save_dir + "/Dataset/deep_highmpc-06-04-17-30-08/dataset"
        save_weights_dir = args.save_dir + "/Weights"
        deep_high_policy.train(env, data_dir=data_dir, save_weights_dir=save_weights_dir, **actor_params)
    elif args.option == 2: # evaluate the policy
        load_dir = args.save_dir + "/Weights/act_net/weight_900.pt"
        sim_visual = SimVisual(env)
        run_frame = partial(run_deep_high_mpc, env, actor_params, load_dir)
        ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                      init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

        #
        if args.save_video:
            writer = animation.writers["ffmpeg"]
            writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1000)
            ani.save("output.mp4", writer=writer)

        plt.tight_layout()
        plt.show()
    elif args.option == 3:
        load_dir = args.save_dir + "/Weights/act_net/weight_900.pt"
        vis = meshcat.Visualizer().open()
        vis_env = Sim_Visual_Meshcat(vis)
        for info, t, _ in run_deep_high_mpc_meshcat(env, actor_params, load_dir, vis_env):
            time.sleep(0.01)

if __name__=="__main__":
    main()

