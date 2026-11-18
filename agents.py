"""The paper's three agents (A2C, PPO, DDPG), with an optional Fuzzy Tiling Activation (FTA) in the critic.

FTA (Pan, Banman & White, ICLR 2021) turns each unit of a layer into k = 20 sparse values.
As in the FTA paper, it replaces the activation of the critic's last hidden layer, with tiles
over [-20, 20] and delta = eta = 2. Everything else stays identical between the two variants.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy

AGENTS = ['a2c', 'ppo', 'ddpg']
TIMESTEPS = {'a2c': 10_000, 'ppo': 50_000, 'ddpg': 20_000}   # the paper's training steps per quarter


class FTA(nn.Module):
    """Fuzzy Tiling Activation: phi(z) = 1 - I_eta(max(c - z, 0) + max(z - delta - c, 0))."""
    k = 20

    def __init__(self, low=-20.0, high=20.0, delta=2.0, eta=2.0):
        super().__init__()
        self.register_buffer('c', torch.arange(low, high, delta))   # left edge of each of the k tiles
        self.delta, self.eta = delta, eta

    def forward(self, z):
        z = z.unsqueeze(-1)
        dist = torch.relu(self.c - z) + torch.relu(z - self.delta - self.c)   # distance from z to each tile
        fuzzy = torch.where(dist < self.eta, dist, torch.ones_like(dist))     # I_eta: x if x < eta, else 1
        return (1 - fuzzy).flatten(-2)


class FTAActorCriticPolicy(ActorCriticPolicy):
    """A2C / PPO: FTA on the last hidden layer of the value network."""

    def _build_mlp_extractor(self):
        super()._build_mlp_extractor()
        self.mlp_extractor.value_net[-1] = FTA()
        self.mlp_extractor.latent_dim_vf *= FTA.k


class FTATD3Policy(TD3Policy):
    """DDPG: FTA on the last hidden layer of the Q-network (and its target copy)."""

    def make_critic(self, features_extractor=None):
        critic = super().make_critic(features_extractor)
        for q in critic.q_networks:          # [Linear, ReLU, Linear, ReLU, Linear(64 -> 1)]
            q[-2] = FTA()
            q[-1] = nn.Linear(q[-1].in_features * FTA.k, 1)
        return critic


def train_agent(name, env, fta, seed, steps=None):
    """Train one agent with the paper's settings (stable-baselines defaults, 64x64 networks) for `steps`
    (default: the paper's training steps)."""
    torch.set_num_threads(1)
    kw = dict(seed=seed, verbose=0, device='cpu')
    if name == 'a2c':
        model = A2C(FTAActorCriticPolicy if fta else 'MlpPolicy', env, **kw)
    elif name == 'ppo':
        model = PPO(FTAActorCriticPolicy if fta else 'MlpPolicy', env, **kw)
    else:
        noise = OrnsteinUhlenbeckActionNoise(np.zeros(30), 0.5 * np.ones(30))   # paper: OU noise, sigma 0.5
        model = DDPG(FTATD3Policy if fta else 'MlpPolicy', env, action_noise=noise, buffer_size=50_000,
                     policy_kwargs={'net_arch': [64, 64]}, **kw)
    return model.learn(steps or TIMESTEPS[name])


def load_agent(path):
    name = path.split('_')[-1].removesuffix('.zip')
    return {'a2c': A2C, 'ppo': PPO, 'ddpg': DDPG}[name].load(path, device='cpu')
