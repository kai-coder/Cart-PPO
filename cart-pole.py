"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


class CartPoleVectorEnv(VectorEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
        "autoreset_mode": AutoresetMode.NEXT_STEP,
    }

    def __init__(
        self,
        num_envs: int = 1,
        max_episode_steps: int = 500,
        render_mode: str | None = None,
        sutton_barto_reward: bool = False,
    ):
        self._sutton_barto_reward = sutton_barto_reward

        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.state = None

        self.steps = np.zeros(num_envs, dtype=np.int32)
        self.prev_done = np.zeros(num_envs, dtype=np.bool_)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.inf,
                np.inf,
                np.inf,
            ],
            dtype=np.float32,
        )

        self.low = -1
        self.high = 1

        self.single_action_space = spaces.Discrete(2)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)

        self.screen_width = 600
        self.screen_height = 400
        self.screens = None
        self.surf = None
        self.clock = None

        self.steps_beyond_terminated = None

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."

        x, x_dot, theta, theta_dot = self.state
        force = np.sign(action - 0.5) * self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.stack((x, x_dot, theta, theta_dot))

        terminated: np.ndarray = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
        )

        self.steps += 1

        truncated = self.steps >= self.max_episode_steps

        if self._sutton_barto_reward:
            reward = -np.array(terminated, dtype=np.float32)
        else:
            reward = np.cos(theta) > np.cos(12/180*np.pi)
            reward = reward.astype(np.float32)
            reward[np.cos(theta) < np.cos(12/180*np.pi)] = -0.1
            reward[terminated] = -1000


        # Reset all environments which terminated or were truncated in the last step
        self.state[:, self.prev_done] = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.prev_done.sum())
        )
        self.state[2, self.prev_done] += np.pi
        self.steps[self.prev_done] = 0
        reward[self.prev_done] = 0.0
        terminated[self.prev_done] = False
        truncated[self.prev_done] = False

        self.prev_done = np.logical_or(terminated, truncated)
        cos_vals = np.cos(self.state[None, 2, :])
        state2 = np.concatenate((self.state, cos_vals), axis=0)
        state2[2, :] = np.sin(self.state[2, :])

        return state2.T.astype(np.float32), reward, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # -0.05 and 0.05 is the default low and high bounds
        self.low, self.high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(
            low=self.low, high=self.high, size=(4, self.num_envs)
        )
        self.state[2, :] += np.pi
        self.steps_beyond_terminated = None
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_done = np.zeros(self.num_envs, dtype=np.bool_)

        cos_vals = np.cos(self.state[None, 2, :])
        state2 = np.concatenate((self.state, cos_vals), axis=0)
        state2[2, :] = np.sin(self.state[2, :])

        #if self.render_mode == "human":
        #    self.render()
        return state2.T.astype(np.float32), {}

    def render(self, state):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screens is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screens = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screens = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]

        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        color = (202, 152, 101) if np.cos(x[2]) < np.cos(12/180*np.pi) else (255, 0, 0)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, color)

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screens.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screens)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screens is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

class CartNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x2 = self.encoder2(x)
        x = self.encoder(x)
        return self.actor(x2), self.critic(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

%%time
num_envs = 100
num_timesteps = 1000
num_iter = 100
env = CartPoleVectorEnv(num_envs=num_envs, max_episode_steps = 500, render_mode=None)
observation_arr = [_] * (int(num_iter/10) + 1)
actorcritic = CartNet().to(DEVICE)

optimizer = torch.optim.Adam(actorcritic.parameters(), lr=1e-4)
#scheduler = CosineAnnealingLR(optimizer, T_max=300)
n = 0
index = np.arange(num_envs)

for iteration in range(num_iter):
    saved=False
    observation, info = env.reset()
    rewards_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    value_t = torch.zeros((num_timesteps + 1, num_envs), device=DEVICE)
    terminated_t = torch.zeros((num_timesteps, num_envs), dtype=bool, device=DEVICE)
    log_probs_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    actions_t = torch.zeros((num_timesteps, num_envs), dtype=int, device=DEVICE)
    states_t = torch.zeros((num_timesteps, num_envs, 5), device=DEVICE)
    advantages_t = torch.zeros((num_timesteps, num_envs), device=DEVICE)
    t = torch.zeros(num_envs, dtype=int, device=DEVICE)
    last_t = torch.zeros(num_envs, dtype=int, device=DEVICE)
    best_t = [0,0]
    best_index = 0
    best_reward = -100000
    beta = 0.9
    
    for _ in range(num_timesteps):
        observation[:, 0] /= 2.4
        observation[:, 3] /= 4
        actorcritic.eval()
        with torch.no_grad():
            actions, values = actorcritic(torch.from_numpy(observation).to(DEVICE))
            
        states_t[t, index] = torch.from_numpy(observation).to(DEVICE)
        
        m = torch.distributions.categorical.Categorical(logits=actions)
        action = m.sample()
        
        observation, reward, terminated, truncated, info = env.step(np.array(action.cpu()))
        terminate = torch.from_numpy(terminated | truncated)
        value_t[t, index] = values.squeeze(1)
        rewards_t[t, index] = torch.from_numpy(reward)
        terminated_t[t, index] = terminate
        if terminate.any():
            for ind in np.where(terminate)[0]:
                this_reward = rewards_t[last_t[ind] : t[ind] + 1, ind].sum()
                if this_reward > best_reward:
                    best_reward = this_reward
                    best_t = [last_t[ind] + 1, t[ind] - 1]
                    best_index = ind
            last_t[np.where(terminate)[0]] = t[np.where(terminate)[0]] + 1
        log_probs_t[t, index] = m.log_prob(action)
        actions_t[t, index] = action
        
        t += 1
    if iteration % 10 == 0 or iteration == 99:
        observation_arr[n] = states_t[best_t[0]:best_t[1], best_index]
        n+=1
    observation[:, 0] /= 2.4
    observation[:, 3] /= 4
    
    with torch.no_grad():
        actions, values = actorcritic(torch.from_numpy(observation).to(DEVICE))
    
    value_t[t[terminate], index[terminate]] = values.squeeze(1)[terminate]
    terminated_t[-1, index] = 1
    value_t[-1, index] = values.squeeze(1)
    
    for i in range(num_timesteps-1, -1, -1):
        delta_t = rewards_t[i] + beta * value_t[i + 1] - value_t[i]
        advantages_t[i, terminated_t[i]] = delta_t[terminated_t[i]]
        if i == num_timesteps - 1:
            continue
        advantages_t[i, ~terminated_t[i]] = delta_t[~terminated_t[i]] + beta * advantages_t[i+1, ~terminated_t[i]]
    print(best_reward.item(), np.array(rewards_t>0.5).mean(),np.array(terminated_t).mean(), np.array(rewards_t).mean())
    terminated_t[0, index] = 1
    for i in range(1):
        terminated_t[1:] = terminated_t[1:] | terminated_t[:-1]
    for i in range(30):
        terminated_t[:-1] = terminated_t[1:] | terminated_t[:-1]
    
    terminated_t = terminated_t.transpose(1, 0).flatten()
    value_t = value_t[:-1].transpose(1, 0).flatten()[~terminated_t]
    rewards_t = rewards_t.transpose(1, 0).flatten()[~terminated_t]
    log_probs_t = log_probs_t.transpose(1, 0).flatten()[~terminated_t]
    actions_t = actions_t.transpose(1, 0).flatten()[~terminated_t]
    states_t = states_t.transpose(1, 0).reshape(-1, 5)[~terminated_t]
    advantages_t = advantages_t.transpose(1, 0).flatten()[~terminated_t]
    
    if (len(value_t) < 128):
        continue
    
    
    advantages_data_loader = DataLoader(
                    TensorDataset(value_t, rewards_t, log_probs_t, 
                                  actions_t, states_t, advantages_t),
                    batch_size=128, shuffle=True,drop_last=True)
    
    actorcritic.train()
    for epoch in range(3):
        loss_mean = 0
        loss_mean2 = 0
        loss_mean3 = 0
        t = 0
        for i in advantages_data_loader:
            
            t += 1
            optimizer.zero_grad()
            
            v, r, p, a, s, ad = i
    
            actions, values = actorcritic(s)
            m = torch.distributions.categorical.Categorical(logits=actions)
            ratio = torch.exp(m.log_prob(a) - p)
            val = ad + v
    
            policy_loss_1 = ad * ratio
            alpha = 0.2 * (1- iteration/num_iter)
            policy_loss_2 = ad * torch.clamp(ratio, 1 - alpha, 1 + alpha)
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()
    
            policy_loss += 0.5 * torch.square(values - val).mean()
            policy_loss -= 0.01 * m.entropy().mean()
            policy_loss.backward()
            optimizer.step()