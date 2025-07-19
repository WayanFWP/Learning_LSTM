import gym
import numpy as np
from env.actuator_env import ActuatorEnv
from model.agent import DDPGAgent
from utils.replay_buffer import ReplayBuffer

env = ActuatorEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer()

num_episodes = 1000
max_steps = 100

for ep in range(num_episodes):
    state = env.reset()
    ep_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        agent.train(replay_buffer)

        state = next_state
        ep_reward += reward

        if done:
            break

    print(f"Episode {ep} | Total Reward: {ep_reward:.2f}")
