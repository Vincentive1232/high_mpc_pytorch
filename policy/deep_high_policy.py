"""
Training a Policy to Select Hyperparameter by using a MLP
"""

import os
import numpy as np
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset, DataLoader

from policy.high_policy import GaussianPolicy
from common import logger_pytorch


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, learning_rate=3e-4, hidden_units=[32, 32], activation=nn.ReLU):
        super(Actor, self).__init__()

        act_cls = None
        if type(activation) == str:
            activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'leaky_relu': nn.LeakyReLU,
            }
            act_cls = activation_map.get(activation.lower(), nn.ReLU)
        else:
            act_cls = activation

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_units[0]),
            act_cls(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            act_cls(),
            nn.Linear(hidden_units[1], act_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x):
        return self.net(x)

    def train_batch(self, obs_batch, act_batch):
        self.train()
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device)

        pred = self(obs_batch)
        loss = 0.5 * nn.functional.mse_loss(pred, act_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_weights(self, save_dir, iter):
        os.makedirs(os.path.join(save_dir, "act_net"), exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "act_net", f"weight_{iter}.pt"))

    def load_weights(self, file_path):
        self.load_state_dict(torch.load(file_path))
        self.eval()


# Dataset class for storing generated training data
class Dataset(object):
    def __init__(self, obs_dim, act_dim, max_size=int(1e6)):
        self._obs_buf = np.zeros(shape=(max_size, obs_dim), dtype=np.float32)
        self._act_buf = np.zeros(shape=(max_size, act_dim), dtype=np.float32)
        self._max_size = max_size
        self._ptr = 0
        self._size = 0

    def add(self, obs, act):
        self._obs_buf[self._ptr] = obs
        self._act_buf[self._ptr] = act
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self._size, size=batch_size)
        return [self._obs_buf[index], self._act_buf[index]]

    def save(self, save_dir, n_iter):
        save_dir = os.path.join(save_dir, "dataset")
        os.makedirs(save_dir, exist_ok=True)
        data_path = os.path.join(save_dir, f"data_{n_iter}.npz")
        np.savez(data_path, obs=self._obs_buf[:self._size], act=self._act_buf[:self._size])

    def get_data(self):
        return self._obs_buf[:self._size], self._act_buf[:self._size]


# Load training data into network(define by using Pytorch)
class MemoryDataset(TorchDataset):
    def __init__(self, data_dir):
        obs_list, act_list = [], []
        for file in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
            data = np.load(file)
            obs_list.append(data["obs"])
            act_list.append(data["act"])
        obs_array = np.vstack(obs_list)
        act_array = np.vstack(act_list)
        self.obs = torch.tensor(obs_array, dtype=torch.float32)
        self.act = torch.tensor(act_array, dtype=torch.float32)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx]


def data_collection(env, logger, save_dir, max_samples, max_wml_iter, beta0, n_samples):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_low = env.action_space.low
    act_high = env.action_space.high

    clip_value = [act_low, act_high]

    dataset = Dataset(obs_dim, act_dim)

    # Data Collection
    obs, done = env.reset(), True
    save_iter = 100

    for n in range(max_samples):
        if done:
            obs = env.reset()

        # # # # # # # # # # # # # # # # # #
        # ---- Weighted Maximum Likelihood
        # # # # # # # # # # # # # # # # # #
        pi = GaussianPolicy(act_dim, clip_value=clip_value)
        # online optimization
        opt_success, pass_index = True, 0
        if obs[0] >= 0.2:
            act = np.array([4.0])
        else:
            # weighted, maximum likelihood optimization
            for i in range(max_wml_iter):
                rewards = np.zeros(n_samples)
                Actions = np.zeros(shape=(n_samples, pi.act_dim))
                for j in range(n_samples):
                    act = pi.sample()

                    rewards[j], pass_index = env.episode(act)

                    Actions[j, :] = act

                if not (np.max(rewards) - np.min(rewards)) <= 1e-10:
                    # compute weights
                    beta = beta0 / (np.max(rewards) - np.min(rewards))
                    weights = np.exp(beta * (rewards - np.max(rewards)))
                    Weights = weights / np.mean(weights)

                    pi.fit(Weights, Actions)
                    opt_success = True
                else:
                    opt_success = False

                logger.log("********** Sample %i ************" % n)
                logger.log("---------- Wml_iter %i ----------" % i)
                logger.log("---------- Reward %f ----------" % np.mean(rewards))
                logger.log("---------- Pass index %i ----------" % pass_index)
                if abs(np.mean(rewards)) <= 0.001:
                    logger.log("---------- Converged %f ----------" % np.mean(rewards))
                    break

            # take the optimal value
            if opt_success:
                act = pi()
                logger.log("---------- Success {0} ----------".format(act))
            else:
                act = np.array([(pass_index + 1) * env.plan_dt])
                logger.log("---------- Fail {0} ----------".format(act))

        # collect optimal value
        dataset.add(obs, act)

        # # # # # # # # # # # # # # # # # #
        # ---- Execute the action --------
        # # # # # # # # # # # # # # # # # #
        obs, _, done, _ = env.step(act)
        logger.log("---------- Obs{0}  ----------".format(obs))
        logger.log("                          ")

        #
        if (n+1) % save_iter == 0:
            dataset.save(save_dir, n)


def train(env, data_dir, save_weights_dir, hidden_units, learning_rate, activation, train_epoch, batch_size):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    dataset = MemoryDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    '''
    train_obs, train_act = None, None
    for i, data_file in enumerate(glob.glob(os.path.join(data_dir, ".npz"))):
        size = int(data_file.split("/")[-1].split("_")[-1].split(".")[0])
        np_file = np.load(data_file)
        obs_array = np_file["obs"][:size, :]
        act_array = np_file["act"][:size, :]

        if i == 0:
            train_obs = obs_array
            train_act = act_array
        else:
            train_obs = np.append(train_obs, obs_array, axis=0)
            train_act = np.append(train_act, act_array, axis=0)

        dataset = MemoryDataset(train_obs, train_act)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    '''
    actor = Actor(obs_dim, act_dim, learning_rate=learning_rate, hidden_units=hidden_units, activation=activation)

    for epoch in range(train_epoch):
        epoch_loss = []
        for obs_batch, act_batch in dataloader:
            loss = actor.train_batch(obs_batch, act_batch)
            epoch_loss.append(loss)

        if epoch % 100 == 0:
            avg_loss = np.mean(epoch_loss)
            print(f"Epoch {epoch:03d}: Loss: {avg_loss:.3f}")
            actor.save_weights(save_weights_dir, iter=epoch)