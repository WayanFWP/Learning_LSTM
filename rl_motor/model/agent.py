import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Because tau is in [-1, 1]
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

        self.gamma = 0.99
        self.tau = 0.005

        # Sync target networks
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).detach().cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Critic loss
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, target_actions)
            y = rewards + self.gamma * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

    def _soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.tau) + s.data * self.tau)

    def _hard_update(self, target, source):
        target.load_state_dict(source.state_dict())
