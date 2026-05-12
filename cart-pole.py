import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import CartNet
from ModifiedCartPoleVectorEnv import ModifiedCartPoleVectorEnv
from blah import fillFrame
import cv2

# Training hyperparameters
num_envs = 1024
num_iter = 40
num_timesteps = 256
discount_factor = 0.99
GAE_param = 0.95
clip_factor = 0.2
lr = 1e-4

# Use Updated Cart Pole Environment
env = ModifiedCartPoleVectorEnv(num_envs=num_envs, max_episode_steps=500, render_mode=None)

# Normalize and clip observations
env = gym.wrappers.vector.NormalizeObservation(env)
env = gym.wrappers.vector.TransformObservation(env, lambda i: np.clip(i, -5, 5))

# Set number of observations and actions
observation, info = env.reset()
num_actions = 2
num_observations = observation.shape[1]

# Setup model and optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = CartNet(num_observations, num_actions).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Stabilize observation normalization
for _ in range(3):
    observation, info = env.reset()
    for t in range(num_timesteps + 1):
        env.step(env.action_space.sample())

frames = np.empty((num_timesteps * 5, 400, 600, 3))

fillFrame(0, frames, 0, model, env, num_timesteps, num_envs, DEVICE)

for iteration in range(num_iter):
    data = {
        "State": torch.zeros((num_timesteps, num_envs, num_observations), device=DEVICE),
        "Value": torch.zeros((num_timesteps + 1, num_envs), device=DEVICE),
        "Reward": torch.zeros((num_timesteps, num_envs), device=DEVICE),
        "Log_Probs": torch.zeros((num_timesteps, num_envs), device=DEVICE),
        "Terminated": torch.zeros((num_timesteps, num_envs), dtype=torch.bool, device=DEVICE),
        "Actions": torch.zeros((num_timesteps, num_envs), device=DEVICE),
        "Advantage": torch.zeros((num_timesteps, num_envs), device=DEVICE)
    }
    track_rewards = torch.zeros((num_timesteps, num_envs), device=DEVICE)

    observation, info = env.reset()
    observation = torch.from_numpy(observation).float().to(DEVICE)

    model.eval()
    for t in range(num_timesteps + 1):
        with torch.no_grad():
            actions, log_probs, values = model(observation)

        if t == num_timesteps:
            data["Value"][t] = values.squeeze(1)
            data["Terminated"][-1] = True
            continue

        next_observation, reward, terminated, truncated, info = env.step(np.array(actions.cpu()))

        data["State"][t] = observation
        data["Value"][t] = values.squeeze(1)
        data["Reward"][t] = torch.from_numpy(reward)
        data["Log_Probs"][t] = log_probs
        data["Terminated"][t] = torch.from_numpy(terminated | truncated)
        data["Actions"][t] = actions

        track_rewards[t] = torch.from_numpy(reward)
        observation = torch.from_numpy(next_observation).float().to(DEVICE)

    print("iteration: " + str(iteration + 1))
    print("    score: " + str(track_rewards.mean().item()))

    # Calculate advantage
    for i in range(num_timesteps - 1, -1, -1):
        delta = data["Reward"][i] + discount_factor * data["Value"][i + 1] - data["Value"][i]

        # If terminated set as delta
        data["Advantage"][i] = delta

        if i == num_timesteps - 1:
            continue

        # If not just terminated add discounted next advantage
        data["Advantage"][i, ~data["Terminated"][i]] += (GAE_param * discount_factor *
                                                         data["Advantage"][i + 1, ~data["Terminated"][i]])

    data["Terminated"] = data["Terminated"].transpose(1, 0).flatten()

    # Get rid of all episodes where terminated
    for key, val in data.items():
        skip = True
        match key:
            case "Actions":
                data[key] = val.transpose(1, 0).reshape(-1)[~data["Terminated"]]
            case "State":
                data[key] = val.transpose(1, 0).reshape(-1, num_observations)[~data["Terminated"]]
            case "Terminated":
                pass
            case _:
                skip = False

        if skip:
            continue

        data[key] = val[:num_timesteps].transpose(1, 0).flatten()[~data["Terminated"]]

    # Terminated values no longer needed
    del data["Terminated"]

    data_loader = DataLoader(
        TensorDataset(*data.values()),
        batch_size=2048, shuffle=True, drop_last=True)

    # train model using PPO algorithm
    model.train()
    for epoch in range(10):
        for batch in data_loader:
            state, values, reward, log_probs, actions, advantages = batch

            curr_log_probs, curr_entropy, curr_values = model(state, actions)

            ratio = torch.exp(curr_log_probs - log_probs)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_factor, 1 + clip_factor)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            target_values = advantages + values
            policy_loss += 0.5 * torch.square(curr_values - target_values).mean()

            policy_loss -= 0.01 * curr_entropy.mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
    if (iteration + 1) % 10 == 0:
        fillFrame(iteration + 1, frames, int((iteration + 1) / 10) , model, env, num_timesteps, num_envs, DEVICE)

# Convert frames to video
fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('training.mp4', fourcc, 60.0, (600, 400))
frames = frames.astype(np.uint8)
for frame in frames:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)

out.release()