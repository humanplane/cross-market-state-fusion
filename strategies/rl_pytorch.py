#!/usr/bin/env python3
"""
PyTorch-based PPO (Proximal Policy Optimization) strategy.

Drop-in replacement for rl_mlx.py that works on Windows/Linux/Mac.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from .base import Strategy, MarketState, Action


@dataclass
class Experience:
    """Single experience tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class Actor(nn.Module):
    """Policy network: state -> action probabilities."""

    def __init__(self, input_dim: int = 18, hidden_size: int = 128, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns action probabilities."""
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        logits = self.fc3(h)
        probs = torch.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    """Value network: state -> expected return."""

    def __init__(self, input_dim: int = 18, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns value estimate."""
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        value = self.fc3(h)
        return value


class RLStrategy(Strategy):
    """PPO-based strategy with actor-critic architecture using PyTorch."""

    def __init__(
        self,
        input_dim: int = 18,
        hidden_size: int = 128,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,  # Shorter horizon for 15-min markets
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.10,  # Higher entropy to prevent premature convergence
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        buffer_size: int = 512,  # Faster updates (4x more frequent)
        batch_size: int = 64,  # Smaller batches for smaller buffer
        n_epochs: int = 10,
        target_kl: float = 0.02,
        device: str = None,
    ):
        super().__init__("rl")
        
        # Set device (CPU on Windows, CUDA if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.output_dim = 3  # BUY, HOLD, SELL (simplified)

        # Hyperparameters
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.target_kl = target_kl

        # Networks
        self.actor = Actor(input_dim, hidden_size, self.output_dim).to(self.device)
        self.critic = Critic(input_dim, hidden_size).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Experience buffer
        self.experiences: List[Experience] = []

        # Running stats for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # For storing last action's log prob and value
        self._last_log_prob = 0.0
        self._last_value = 0.0

    def act(self, state: MarketState) -> Action:
        """Select action using current policy."""
        features = state.to_features()
        features_torch = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get action probabilities and value
            probs = self.actor(features_torch)
            value = self.critic(features_torch)

            probs_np = probs.cpu().numpy()[0]
            value_np = float(value.cpu().numpy()[0, 0])

        if self.training:
            # Sample from distribution
            action_idx = np.random.choice(self.output_dim, p=probs_np)
        else:
            # Greedy
            action_idx = int(np.argmax(probs_np))

        # Store for experience collection
        self._last_log_prob = float(np.log(probs_np[action_idx] + 1e-8))
        self._last_value = value_np

        return Action(action_idx)

    def store(self, state: MarketState, action: Action, reward: float,
              next_state: MarketState, done: bool):
        """Store experience for training."""
        # Update running reward stats for normalization
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        self.reward_std = np.sqrt(
            ((self.reward_count - 1) * self.reward_std**2 + delta * (reward - self.reward_mean))
            / max(1, self.reward_count)
        )

        # Normalize reward
        norm_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

        exp = Experience(
            state=state.to_features(),
            action=action.value,
            reward=norm_reward,
            next_state=next_state.to_features(),
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
        )
        self.experiences.append(exp)

        # Limit buffer size
        if len(self.experiences) > self.buffer_size:
            self.experiences = self.experiences[-self.buffer_size:]

    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                     dones: np.ndarray, next_value: float) -> tuple:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update(self) -> Optional[Dict[str, float]]:
        """Update policy using PPO with PyTorch autograd."""
        if len(self.experiences) < self.buffer_size:
            return None

        # Convert experiences to arrays
        states = np.array([e.state for e in self.experiences], dtype=np.float32)
        actions = np.array([e.action for e in self.experiences], dtype=np.int64)
        rewards = np.array([e.reward for e in self.experiences], dtype=np.float32)
        dones = np.array([e.done for e in self.experiences], dtype=np.float32)
        old_log_probs = np.array([e.log_prob for e in self.experiences], dtype=np.float32)
        old_values = np.array([e.value for e in self.experiences], dtype=np.float32)

        # Compute next value for GAE
        with torch.no_grad():
            next_state_torch = torch.FloatTensor(self.experiences[-1].next_state).unsqueeze(0).to(self.device)
            next_value = float(self.critic(next_state_torch).cpu().numpy()[0, 0])

        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, old_values, dones, next_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to PyTorch tensors
        states_torch = torch.FloatTensor(states).to(self.device)
        actions_torch = torch.LongTensor(actions).to(self.device)
        old_log_probs_torch = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_torch = torch.FloatTensor(advantages).to(self.device)
        returns_torch = torch.FloatTensor(returns).to(self.device)
        old_values_torch = torch.FloatTensor(old_values).to(self.device)

        n_samples = len(self.experiences)
        all_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }

        # Multiple epochs over the data
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)

            epoch_kl = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                batch_states = states_torch[batch_idx]
                batch_actions = actions_torch[batch_idx]
                batch_old_log_probs = old_log_probs_torch[batch_idx]
                batch_advantages = advantages_torch[batch_idx]
                batch_returns = returns_torch[batch_idx]
                batch_old_values = old_values_torch[batch_idx]

                # ====== Update Actor ======
                self.actor_optimizer.zero_grad()
                
                probs = self.actor(batch_states)
                
                # Get log probs for taken actions
                selected_probs = probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                log_probs = torch.log(selected_probs + 1e-8)

                # PPO clipped objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus (encourages exploration)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy

                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean()
                    clip_frac = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean()

                # ====== Update Critic ======
                self.critic_optimizer.zero_grad()
                
                values = self.critic(batch_states).squeeze()

                # Value loss with clipping (PPO2 style)
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (batch_returns - values) ** 2
                value_loss2 = (batch_returns - values_clipped) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()

                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # Record metrics
                all_metrics["policy_loss"].append(float(policy_loss.item()))
                all_metrics["value_loss"].append(float(value_loss.item()))
                all_metrics["entropy"].append(float(entropy.item()))
                all_metrics["approx_kl"].append(float(approx_kl.item()))
                all_metrics["clip_fraction"].append(float(clip_frac.item()))

                epoch_kl += float(approx_kl.item())
                n_batches += 1

            # Early stopping on KL divergence
            avg_kl = epoch_kl / max(1, n_batches)
            if avg_kl > self.target_kl:
                print(f"  [RL] Early stop epoch {epoch}, KL={avg_kl:.4f}")
                break

        # Clear buffer after update
        self.experiences.clear()

        # Compute explained variance
        y_pred = old_values
        y_true = returns
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0.0

        return {
            "policy_loss": np.mean(all_metrics["policy_loss"]),
            "value_loss": np.mean(all_metrics["value_loss"]),
            "entropy": np.mean(all_metrics["entropy"]),
            "approx_kl": np.mean(all_metrics["approx_kl"]),
            "clip_fraction": np.mean(all_metrics["clip_fraction"]),
            "explained_variance": explained_var,
        }

    def reset(self):
        """Clear experience buffer."""
        self.experiences.clear()

    def save(self, path: str):
        """Save model and training state."""
        save_path = path.replace(".npz", "") + ".pth"
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count,
        }, save_path)
        print(f"  [RL] Model saved to {save_path}")

    def load(self, path: str):
        """Load model and training state."""
        load_path = path.replace(".npz", "").replace(".safetensors", "") + ".pth"
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.reward_mean = checkpoint['reward_mean']
        self.reward_std = checkpoint['reward_std']
        self.reward_count = checkpoint['reward_count']
        
        print(f"  [RL] Model loaded from {load_path}")

