import torch
import numpy as np
import matplotlib.pyplot as plt
from env.pybullet_env import PybulletActuatorEnv, PybulletRobotArmEnv
from model.agent import DDPGAgent
from utils.replay_buffer import ReplayBuffer
import os
from datetime import datetime

def train_pybullet_agent(env_type='single_joint', render=False, num_episodes=500):
    """
    Train DDPG agent in PyBullet environment
    
    Args:
        env_type: 'single_joint' or 'robot_arm'
        render: Whether to show PyBullet GUI during training
        num_episodes: Number of training episodes
    """
    print("=" * 60)
    print(f"TRAINING DDPG AGENT IN PYBULLET - {env_type.upper()}")
    print("=" * 60)
    
    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("pybullet_plots", exist_ok=True)
    
    # Create environment
    if env_type == 'single_joint':
        env = PybulletActuatorEnv(render=render)
        max_steps_per_episode = 500
    else:
        env = PybulletRobotArmEnv(render=render, num_joints=3)
        max_steps_per_episode = 800
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    
    # Create agent and replay buffer
    agent = DDPGAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # Training parameters
    batch_size = 64
    save_interval = 50
    eval_interval = 25
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    evaluation_scores = []
    
    print(f"Training for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Set random target for variety
        if env_type == 'single_joint':
            target = np.random.uniform(-1.0, 1.0)
            env.set_target_position(target)
        else:
            targets = np.random.uniform(-1.0, 1.0, 3)
            env.target_positions = targets
        
        for step in range(max_steps_per_episode):
            # Get action with exploration noise
            action = agent.act(state)
            
            # Add exploration noise (decreasing over time)
            noise_scale = max(0.1, 1.0 - episode / (num_episodes * 0.7))
            noise = np.random.normal(0, noise_scale, action.shape)
            action = np.clip(action + noise, env.action_space.low, env.action_space.high)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train agent
            if len(replay_buffer) > batch_size:
                agent.train(replay_buffer, batch_size)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg Reward (10): {avg_reward:8.2f} | "
                  f"Length: {episode_length:4d} | "
                  f"Avg Length: {avg_length:6.1f}")
        
        # Evaluation episodes (without exploration noise)
        if (episode + 1) % eval_interval == 0:
            eval_score = evaluate_agent(env, agent, env_type, num_eval_episodes=3)
            evaluation_scores.append(eval_score)
            print(f"    Evaluation Score: {eval_score:.2f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = f"checkpoints/pybullet_{env_type}_ep{episode + 1}.pth"
            save_model(agent, checkpoint_path, episode + 1, episode_rewards, episode_lengths, env_type)
            print(f"    Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = f"checkpoints/pybullet_{env_type}_model.pth"
    save_model(agent, final_path, num_episodes, episode_rewards, episode_lengths, env_type)
    
    # Final evaluation
    final_score = evaluate_agent(env, agent, env_type, num_eval_episodes=10)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Final model saved: {final_path}")
    print(f"Total episodes: {num_episodes}")
    print(f"Final evaluation score: {final_score:.2f}")
    print(f"Final average reward (last 50): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"Final average episode length: {np.mean(episode_lengths[-50:]):.1f}")
    
    # Create training plots
    create_training_plots(episode_rewards, episode_lengths, evaluation_scores, env_type)
    
    # Close environment
    env.close()
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'evaluation_scores': evaluation_scores,
        'final_score': final_score
    }

def evaluate_agent(env, agent, env_type, num_eval_episodes=5):
    """Evaluate agent performance without exploration noise"""
    total_reward = 0
    
    for _ in range(num_eval_episodes):
        state = env.reset()
        episode_reward = 0
        
        # Set random target
        if env_type == 'single_joint':
            target = np.random.uniform(-1.0, 1.0)
            env.set_target_position(target)
        else:
            targets = np.random.uniform(-1.0, 1.0, 3)
            env.target_positions = targets
        
        for _ in range(500):  # Max steps for evaluation
            action = agent.act(state)  # No exploration noise
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_eval_episodes

def save_model(agent, path, episode, rewards, lengths, env_type):
    """Save model with training metadata"""
    torch.save({
        'episode': episode,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_state_dict': agent.critic.state_dict(),
        'target_actor_state_dict': agent.target_actor.state_dict(),
        'target_critic_state_dict': agent.target_critic.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'episode_rewards': rewards,
        'episode_lengths': lengths,
        'env_type': env_type,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, path)

def create_training_plots(rewards, lengths, eval_scores, env_type):
    """Create comprehensive training plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'PyBullet {env_type.replace("_", " ").title()} Training Progress', fontsize=16)
    
    episodes = range(len(rewards))
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) >= 50:
        # Moving average
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 'red', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Episode Lengths
    axes[0, 1].plot(episodes, lengths, alpha=0.3, color='green', label='Episode Length')
    if len(lengths) >= 50:
        moving_avg_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(lengths)), moving_avg_len, 'orange', linewidth=2, label=f'Moving Avg ({window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Evaluation Scores
    if eval_scores:
        eval_episodes = range(0, len(rewards), len(rewards)//len(eval_scores))[:len(eval_scores)]
        axes[1, 0].plot(eval_episodes, eval_scores, 'bo-', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Evaluation Score')
        axes[1, 0].set_title('Evaluation Performance')
        axes[1, 0].grid(True)
    
    # Plot 4: Learning Progress (last 100 episodes)
    if len(rewards) > 100:
        recent_rewards = rewards[-100:]
        recent_episodes = range(len(rewards)-100, len(rewards))
        axes[1, 1].plot(recent_episodes, recent_rewards, 'purple', alpha=0.7)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Recent Performance (Last 100 Episodes)')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"pybullet_plots/training_{env_type}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training plots saved to: {plot_path}")

def main():
    """Main training function"""
    print("PyBullet DDPG Training")
    print("=" * 30)
    
    # Choose environment
    print("Select environment:")
    print("1. Single Joint Actuator")
    print("2. Multi-Joint Robot Arm")
    env_choice = input("Enter choice (1 or 2): ").strip()
    env_type = 'single_joint' if env_choice == '1' else 'robot_arm'
    
    # Training parameters
    num_episodes = int(input("Number of training episodes (default 500): ") or "500")
    render = input("Show PyBullet GUI during training? (y/n): ").lower() == 'y'
    
    if render:
        print("WARNING: GUI rendering will slow down training significantly!")
        confirm = input("Continue with GUI? (y/n): ").lower() == 'y'
        if not confirm:
            render = False
    
    print(f"\nStarting training...")
    print(f"Environment: {env_type}")
    print(f"Episodes: {num_episodes}")
    print(f"GUI Rendering: {render}")
    
    # Start training
    results = train_pybullet_agent(env_type, render, num_episodes)
    
    print("\nTraining completed successfully!")
    print(f"You can now run 'python pybullet_simulate.py' to test your trained model.")

if __name__ == "__main__":
    main()
