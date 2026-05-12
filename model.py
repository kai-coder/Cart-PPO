from torch import nn
import torch
from typing import Type

# Creates sequential network of linear layers with sizes specified
# by layer_sizes and activation function specified by activation_funct
def create_sequential(layer_sizes: list[int], activation_funct: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    modules = []

    for i in range(len(layer_sizes) - 2):
        modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        modules.append(activation_funct())
    modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    return nn.Sequential(*modules)


class CartNet(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()

        actor_layers = [observation_size, 64, 64, action_size]
        actor_activation = nn.ReLU
        self.actor = create_sequential(actor_layers, actor_activation)

        critic_layers = [observation_size, 64, 64, 1]
        critic_activation = nn.ReLU
        self.critic = create_sequential(critic_layers, critic_activation)

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None, inference: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If inference return best predicted action
        if inference:
            prediction = self.actor(x)
            actions = prediction[:, 1] > prediction[:, 0]
            return actions, torch.ones_like(actions), self.critic(x)

        distribution = torch.distributions.categorical.Categorical(logits=self.actor(x))

        # Sample actions based on current policy
        if actions is None:
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            return actions, log_probs, self.critic(x)

        # Return log_probs and entropy based on current policy
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return log_probs, entropy, self.critic(x)