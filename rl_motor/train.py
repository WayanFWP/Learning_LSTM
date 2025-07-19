import gym
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from env.actuator_env import ActuatorEnv
from model.agent import DDPGAgent
from utils.replay_buffer import ReplayBuffer

# Create directories for saving
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

env = ActuatorEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = DDPGAgent(state_dim, action_dim)
replay_buffer = ReplayBuffer()

num_episodes = 1000
max_steps = 100

# Training metrics
episode_rewards = []
episode_lengths = []
save_interval = 100  # Save model every 100 episodes

print("Starting training...")
print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
print(f"Training for {num_episodes} episodes")
print("-" * 50)

for ep in range(num_episodes):
    state = env.reset()
    ep_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        # Only train if we have enough samples
        if len(replay_buffer) > 64:
            agent.train(replay_buffer)

        state = next_state
        ep_reward += reward

        if done:
            break

    # Store metrics
    episode_rewards.append(ep_reward)
    episode_lengths.append(step + 1)

    # Print progress
    if (ep + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        avg_length = np.mean(episode_lengths[-10:])
        print(f"Episode {ep + 1:4d} | Avg Reward (last 10): {avg_reward:7.2f} | Avg Length: {avg_length:5.1f}")

    # Save model checkpoint
    if (ep + 1) % save_interval == 0:
        checkpoint_path = f"checkpoints/ddpg_model_ep{ep + 1}.pth"
        torch.save({
            'episode': ep + 1,
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'target_actor_state_dict': agent.target_actor.state_dict(),
            'target_critic_state_dict': agent.target_critic.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }, checkpoint_path)
        print(f"    Checkpoint saved: {checkpoint_path}")

# Save final model
final_model_path = "checkpoints/ddpg_model.pth"
torch.save({
    'episode': num_episodes,
    'actor_state_dict': agent.actor.state_dict(),
    'critic_state_dict': agent.critic.state_dict(),
    'target_actor_state_dict': agent.target_actor.state_dict(),
    'target_critic_state_dict': agent.target_critic.state_dict(),
    'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
    'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
    'episode_rewards': episode_rewards,
    'episode_lengths': episode_lengths
}, final_model_path)

print("\n" + "=" * 50)
print("TRAINING COMPLETED!")
print(f"Final model saved: {final_model_path}")
print(f"Total episodes: {num_episodes}")
print(f"Final average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
print("=" * 50)

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
# Plot moving average
window = 50
if len(episode_rewards) >= window:
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', label=f'Moving Average ({window})')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress - Rewards')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(episode_lengths, alpha=0.3, label='Episode Length')
if len(episode_lengths) >= window:
    moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(episode_lengths)), moving_avg_len, 'r-', label=f'Moving Average ({window})')
plt.xlabel('Episode')
plt.ylabel('Episode Length')
plt.title('Training Progress - Episode Lengths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plots/training_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Training plots saved to: training_plots/training_progress.png")
