import torch
import numpy as np
import matplotlib.pyplot as plt
from env.actuator_env import ActuatorEnv
from model.agent import DDPGAgent
import os
import json
from datetime import datetime

class ModelSimulator:
    def __init__(self, model_path=None):
        """
        Initialize the simulator
        
        Args:
            model_path: Path to saved model checkpoint (optional)
        """
        self.env = ActuatorEnv()
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        self.agent = DDPGAgent(self.state_dim, self.action_dim)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Using untrained model (random weights)")
    
    def save_model(self, path):
        """Save the current model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'target_actor_state_dict': self.agent.target_actor.state_dict(),
            'target_critic_state_dict': self.agent.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.agent.critic_optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model state from checkpoint"""
        if not os.path.exists(path):
            print(f"Model file {path} not found!")
            return False
        
        checkpoint = torch.load(path)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.agent.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        return True
    
    def simulate_episode(self, max_steps=100, initial_state=None, add_noise=False, noise_std=0.1):
        """
        Simulate a single episode
        
        Args:
            max_steps: Maximum steps in episode
            initial_state: Custom initial state (optional)
            add_noise: Whether to add noise to actions
            noise_std: Standard deviation of noise
            
        Returns:
            Dictionary with simulation results
        """
        if initial_state is not None:
            self.env.state = np.array(initial_state)
            state = self.env.state.copy()
        else:
            state = self.env.reset()
        
        # Storage for trajectory
        states = [state.copy()]
        actions = []
        rewards = []
        target_torques = []
        predicted_torques = []
        
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from agent
            action = self.agent.act(state)
            
            # Add noise if specified
            if add_noise:
                noise = np.random.normal(0, noise_std, action.shape)
                action = np.clip(action + noise, -1, 1)
            
            # Store predicted torque
            predicted_torques.append(action[0])
            
            # Calculate target torque for comparison
            pos_error, velocity, kp, kd, tau_ff = state
            target_torque = kp * pos_error + kd * velocity + tau_ff
            target_torques.append(target_torque)
            
            # Take step in environment
            next_state, reward, done, _ = self.env.step(action)
            
            # Store data
            actions.append(action[0])
            rewards.append(reward)
            states.append(next_state.copy())
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'target_torques': np.array(target_torques),
            'predicted_torques': np.array(predicted_torques),
            'total_reward': total_reward,
            'steps': len(actions)
        }
    
    def run_multiple_simulations(self, num_episodes=10, **kwargs):
        """Run multiple simulation episodes and collect statistics"""
        results = []
        
        print(f"Running {num_episodes} simulation episodes...")
        for i in range(num_episodes):
            result = self.simulate_episode(**kwargs)
            results.append(result)
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1}/{num_episodes} episodes")
        
        # Calculate statistics
        total_rewards = [r['total_reward'] for r in results]
        episode_lengths = [r['steps'] for r in results]
        
        stats = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'results': results
        }
        
        return stats
    
    def visualize_episode(self, result, save_path=None, show_plot=True):
        """Visualize the results of a single episode"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Actuator Control Simulation Results', fontsize=16)
        
        time_steps = np.arange(len(result['actions']))
        
        # Plot 1: Position Error
        axes[0, 0].plot(time_steps, result['states'][:-1, 0], 'b-', label='Position Error')
        axes[0, 0].set_title('Position Error Over Time')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Position Error')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Plot 2: Velocity
        axes[0, 1].plot(time_steps, result['states'][:-1, 1], 'g-', label='Velocity')
        axes[0, 1].set_title('Velocity Over Time')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Velocity')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Plot 3: Torque Comparison
        axes[0, 2].plot(time_steps, result['target_torques'], 'r-', label='Target Torque', alpha=0.7)
        axes[0, 2].plot(time_steps, result['predicted_torques'], 'b--', label='Predicted Torque', alpha=0.7)
        axes[0, 2].set_title('Torque Comparison')
        axes[0, 2].set_xlabel('Time Steps')
        axes[0, 2].set_ylabel('Torque')
        axes[0, 2].grid(True)
        axes[0, 2].legend()
        
        # Plot 4: Torque Error
        torque_error = np.abs(result['target_torques'] - result['predicted_torques'])
        axes[1, 0].plot(time_steps, torque_error, 'm-', label='Torque Error')
        axes[1, 0].set_title('Absolute Torque Error')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot 5: Rewards
        axes[1, 1].plot(time_steps, result['rewards'], 'orange', label='Reward')
        axes[1, 1].set_title('Rewards Over Time')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        # Plot 6: Control Parameters
        axes[1, 2].plot(time_steps, result['states'][:-1, 2], 'c-', label='Kp', alpha=0.7)
        axes[1, 2].plot(time_steps, result['states'][:-1, 3], 'y-', label='Kd', alpha=0.7)
        axes[1, 2].plot(time_steps, result['states'][:-1, 4], 'k-', label='τ_ff', alpha=0.7)
        axes[1, 2].set_title('Control Parameters')
        axes[1, 2].set_xlabel('Time Steps')
        axes[1, 2].set_ylabel('Parameter Values')
        axes[1, 2].grid(True)
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def test_different_scenarios(self):
        """Test the model on different predefined scenarios"""
        scenarios = [
            {
                'name': 'Zero Initial State',
                'initial_state': [0.0, 0.0, 1.0, 0.1, 0.0],
                'description': 'Starting from rest position'
            },
            {
                'name': 'High Position Error',
                'initial_state': [2.0, 0.0, 1.0, 0.1, 0.0],
                'description': 'Large initial position error'
            },
            {
                'name': 'High Initial Velocity',
                'initial_state': [0.0, 2.0, 1.0, 0.1, 0.0],
                'description': 'High initial velocity'
            },
            {
                'name': 'Low Kp',
                'initial_state': [1.0, 0.0, 0.1, 0.1, 0.0],
                'description': 'Low proportional gain'
            },
            {
                'name': 'High Kd',
                'initial_state': [1.0, 1.0, 1.0, 2.0, 0.0],
                'description': 'High derivative gain'
            },
            {
                'name': 'With Feedforward',
                'initial_state': [0.5, 0.5, 1.0, 0.1, 1.0],
                'description': 'Non-zero feedforward torque'
            }
        ]
        
        results = {}
        
        print("Testing different scenarios...")
        print("-" * 50)
        
        for scenario in scenarios:
            print(f"Testing: {scenario['name']} - {scenario['description']}")
            result = self.simulate_episode(initial_state=scenario['initial_state'])
            results[scenario['name']] = result
            
            print(f"  Total Reward: {result['total_reward']:.3f}")
            print(f"  Episode Length: {result['steps']} steps")
            print(f"  Final Position Error: {result['states'][-1, 0]:.3f}")
            print(f"  Final Velocity: {result['states'][-1, 1]:.3f}")
            print()
        
        return results
    
    def performance_analysis(self, stats):
        """Print detailed performance analysis"""
        print("=" * 60)
        print("PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(f"Number of Episodes: {len(stats['results'])}")
        print(f"Mean Total Reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
        print(f"Reward Range: [{stats['min_reward']:.3f}, {stats['max_reward']:.3f}]")
        print(f"Mean Episode Length: {stats['mean_episode_length']:.1f} steps")
        print()
        
        # Analyze torque accuracy
        all_torque_errors = []
        for result in stats['results']:
            errors = np.abs(result['target_torques'] - result['predicted_torques'])
            all_torque_errors.extend(errors.tolist())
        
        mean_torque_error = np.mean(all_torque_errors)
        std_torque_error = np.std(all_torque_errors)
        
        print(f"Torque Prediction Accuracy:")
        print(f"  Mean Absolute Error: {mean_torque_error:.4f}")
        print(f"  Standard Deviation: {std_torque_error:.4f}")
        print(f"  Max Error: {np.max(all_torque_errors):.4f}")
        print("=" * 60)

def main():
    """Main simulation function with interactive menu"""
    print("=" * 60)
    print("ACTUATOR CONTROL MODEL SIMULATOR")
    print("=" * 60)
    
    # Initialize simulator
    model_path = "checkpoints/ddpg_model.pth"  # Default model path
    simulator = ModelSimulator(model_path)
    
    while True:
        print("\nSelect simulation option:")
        print("1. Run single episode simulation")
        print("2. Run multiple episodes with statistics")
        print("3. Test different scenarios")
        print("4. Train model (limited training)")
        print("5. Save current model")
        print("6. Load different model")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            print("\nRunning single episode simulation...")
            result = simulator.simulate_episode(max_steps=100)
            print(f"Total Reward: {result['total_reward']:.3f}")
            print(f"Episode Length: {result['steps']} steps")
            
            # Ask if user wants to visualize
            if input("Visualize results? (y/n): ").lower() == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"simulation_plots/episode_{timestamp}.png"
                simulator.visualize_episode(result, save_path=save_path)
        
        elif choice == '2':
            num_episodes = int(input("Number of episodes to run (default 10): ") or "10")
            add_noise = input("Add noise to actions? (y/n): ").lower() == 'y'
            
            kwargs = {'add_noise': add_noise}
            if add_noise:
                noise_std = float(input("Noise standard deviation (default 0.1): ") or "0.1")
                kwargs['noise_std'] = noise_std
            
            stats = simulator.run_multiple_simulations(num_episodes, **kwargs)
            simulator.performance_analysis(stats)
        
        elif choice == '3':
            print("\nTesting different scenarios...")
            scenario_results = simulator.test_different_scenarios()
            
            if input("Visualize a specific scenario? (y/n): ").lower() == 'y':
                scenario_names = list(scenario_results.keys())
                print("\nAvailable scenarios:")
                for i, name in enumerate(scenario_names):
                    print(f"{i+1}. {name}")
                
                scenario_idx = int(input("Select scenario number: ")) - 1
                if 0 <= scenario_idx < len(scenario_names):
                    selected_name = scenario_names[scenario_idx]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"simulation_plots/scenario_{selected_name.replace(' ', '_')}_{timestamp}.png"
                    simulator.visualize_episode(scenario_results[selected_name], save_path=save_path)
        
        elif choice == '4':
            print("\nRunning limited training...")
            from utils.replay_buffer import ReplayBuffer
            
            replay_buffer = ReplayBuffer()
            num_episodes = int(input("Number of training episodes (default 50): ") or "50")
            
            for ep in range(num_episodes):
                state = simulator.env.reset()
                ep_reward = 0
                
                for step in range(100):
                    action = simulator.agent.act(state)
                    next_state, reward, done, _ = simulator.env.step(action)
                    replay_buffer.push(state, action, reward, next_state, done)
                    
                    if len(replay_buffer) > 64:
                        simulator.agent.train(replay_buffer)
                    
                    state = next_state
                    ep_reward += reward
                    
                    if done:
                        break
                
                if (ep + 1) % 10 == 0:
                    print(f"Episode {ep + 1} | Total Reward: {ep_reward:.2f}")
            
            print("Training completed!")
        
        elif choice == '5':
            save_path = input("Enter save path (default: checkpoints/ddpg_model.pth): ").strip()
            if not save_path:
                save_path = "checkpoints/ddpg_model.pth"
            simulator.save_model(save_path)
        
        elif choice == '6':
            load_path = input("Enter model path to load: ").strip()
            if simulator.load_model(load_path):
                print("Model loaded successfully!")
            else:
                print("Failed to load model!")
        
        elif choice == '7':
            print("Exiting simulator...")
            break
        
        else:
            print("Invalid choice! Please select 1-7.")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("simulation_plots", exist_ok=True)
    
    main()
