import numpy as np
import torch
def fillFrame(epoch, frames, amnt, model, env, num_timesteps, num_envs, DEVICE):
    env.env._update_running_mean = False
    model.eval()
    np.random.seed(0)
    torch.manual_seed(0)
    observation, info = env.reset(seed=0)
    all_rewards = np.zeros((num_timesteps, num_envs))
    for t in range(num_timesteps):
        observation = torch.from_numpy(observation).float().to(DEVICE)
        actions, log_probs, values = model(observation, inference=True)

        observation, reward, terminated, truncated, info = env.step(np.array(actions.cpu()))
        all_rewards[t] = reward

    sorted = np.argsort(all_rewards.sum(axis=0))
    np.random.seed(0)
    torch.manual_seed(0)
    observation, info = env.reset(seed=0)
    for t in range(num_timesteps):
        observation = torch.from_numpy(observation).float().to(DEVICE)
        actions, log_probs, values = model(observation, inference=True)

        observation, reward, terminated, truncated, info = env.step(np.array(actions.cpu()))
        out = env.env.env.render(epoch, sorted)
        frames[amnt * num_timesteps + t] = out
    env.env._update_running_mean = True