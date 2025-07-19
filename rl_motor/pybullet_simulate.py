import torch
import numpy as np
import matplotlib.pyplot as plt
from env.pybullet_env import PybulletActuatorEnv, PybulletRobotArmEnv
from model.agent import DDPGAgent
import os
import json
from datetime import datetime
import time

class PybulletSimulator:
    def __init__(self, model_path=None, render=False, env_type='single_joint'):
        """
        Initialize the PyBullet simulator
        
        Args:
            model_path: Path to saved model checkpoint (optional)
            render: Whether to show PyBullet GUI
            env_type: 'single_joint' or 'robot_arm'
        """
        self.render = render
        self.env_type = env_type
        
        # Create environment
        if env_type == 'single_joint':
            self.env = PybulletActuatorEnv(render=render)
        elif env_type == 'robot_arm':
            self.env = PybulletRobotArmEnv(render=render, num_joints=3)
        else:
            raise ValueError("env_type must be 'single_joint' or 'robot_arm'")
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        print(f"Environment: {env_type}")
        print(f"State dimension: {self.state_dim}")
        print(f"Action dimension: {self.action_dim}")
        
        # Create agent with appropriate dimensions
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
            'env_type': self.env_type,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
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
    
    def simulate_episode(self, max_steps=1000, target_position=None, add_noise=False, noise_std=0.1):
        """
        Simulate a single episode in PyBullet
        
        Args:
            max_steps: Maximum steps in episode
            target_position: Custom target position (single joint) or positions (multi-joint)
            add_noise: Whether to add noise to actions
            noise_std: Standard deviation of noise
            
        Returns:
            Dictionary with simulation results
        """
        # Reset environment
        state = self.env.reset()
        
        # Set custom target if provided
        if target_position is not None:
            if self.env_type == 'single_joint':
                self.env.set_target_position(target_position)
            elif self.env_type == 'robot_arm':
                self.env.target_positions = np.array(target_position)
        
        # Storage for trajectory
        states = [state.copy()]
        actions = []
        rewards = []
        infos = []
        
        total_reward = 0
        
        print(f"Starting simulation with target: {target_position}")
        
        for step in range(max_steps):
            # Get action from agent
            action = self.agent.act(state)
            
            # Add noise if specified
            if add_noise:
                noise = np.random.normal(0, noise_std, action.shape)
                action = np.clip(action + noise, 
                               self.env.action_space.low, 
                               self.env.action_space.high)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store data
            actions.append(action.copy())
            rewards.append(reward)
            states.append(next_state.copy())
            infos.append(info.copy() if info else {})
            
            total_reward += reward
            state = next_state
            
            # Render if enabled
            if self.render:
                self.env.render()
            
            # Print progress
            if step % 100 == 0:
                if self.env_type == 'single_joint':
                    pos_error = info.get('position_error', 0)
                    print(f"Step {step}: Reward={reward:.3f}, Position Error={pos_error:.3f}")
                else:
                    print(f"Step {step}: Reward={reward:.3f}")
            
            if done:
                print(f"Episode completed at step {step}")
                break
        
        print(f"Total reward: {total_reward:.3f}")
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos,
            'total_reward': total_reward,
            'steps': len(actions)
        }
    
    def visualize_episode(self, result, save_path=None, show_plot=True):
        """Visualize the results of a PyBullet simulation"""
        if self.env_type == 'single_joint':
            self._visualize_single_joint(result, save_path, show_plot)
        else:
            self._visualize_robot_arm(result, save_path, show_plot)
    
    def _visualize_single_joint(self, result, save_path=None, show_plot=True):
        """Visualize single joint simulation"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PyBullet Single Joint Actuator Simulation', fontsize=16)
        
        time_steps = np.arange(len(result['actions']))
        states = result['states'][:-1]  # Remove last state
        
        # Extract state components
        joint_positions = states[:, 0]
        joint_velocities = states[:, 1]
        target_positions = states[:, 2]
        position_errors = states[:, 3]
        applied_torques = states[:, 4] if states.shape[1] > 4 else result['actions'][:, 0]
        
        # Plot 1: Joint Position vs Target
        axes[0, 0].plot(time_steps, joint_positions, 'b-', label='Joint Position', linewidth=2)
        axes[0, 0].plot(time_steps, target_positions, 'r--', label='Target Position', linewidth=2)
        axes[0, 0].set_title('Joint Position Tracking')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Position (rad)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Position Error
        axes[0, 1].plot(time_steps, position_errors, 'r-', label='Position Error')
        axes[0, 1].set_title('Position Error Over Time')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Joint Velocity
        axes[0, 2].plot(time_steps, joint_velocities, 'g-', label='Joint Velocity')
        axes[0, 2].set_title('Joint Velocity')
        axes[0, 2].set_xlabel('Time Steps')
        axes[0, 2].set_ylabel('Velocity (rad/s)')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Applied Torque
        axes[1, 0].plot(time_steps, result['actions'][:, 0], 'm-', label='Applied Torque')
        axes[1, 0].set_title('Applied Torque')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Torque (Nm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Rewards
        axes[1, 1].plot(time_steps, result['rewards'], 'orange', label='Reward')
        axes[1, 1].set_title('Rewards Over Time')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: Performance Metrics
        abs_errors = np.abs(position_errors)
        cumulative_error = np.cumsum(abs_errors)
        axes[1, 2].plot(time_steps, abs_errors, 'r-', alpha=0.7, label='Abs Error')
        axes[1, 2].plot(time_steps, cumulative_error / (time_steps + 1), 'b-', label='Avg Abs Error')
        axes[1, 2].set_title('Error Analysis')
        axes[1, 2].set_xlabel('Time Steps')
        axes[1, 2].set_ylabel('Error')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _visualize_robot_arm(self, result, save_path=None, show_plot=True):
        """Visualize robot arm simulation"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PyBullet Robot Arm Simulation', fontsize=16)
        
        time_steps = np.arange(len(result['actions']))
        states = result['states'][:-1]
        
        num_joints = self.action_dim
        
        # Extract joint positions, velocities, targets, and errors
        joint_positions = states[:, :num_joints]
        joint_velocities = states[:, num_joints:2*num_joints]
        target_positions = states[:, 2*num_joints:3*num_joints]
        position_errors = states[:, 3*num_joints:4*num_joints]
        
        # Plot 1: Joint Positions
        for i in range(num_joints):
            axes[0, 0].plot(time_steps, joint_positions[:, i], label=f'Joint {i+1}')
        axes[0, 0].set_title('Joint Positions')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Position (rad)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Position Errors
        for i in range(num_joints):
            axes[0, 1].plot(time_steps, position_errors[:, i], label=f'Joint {i+1} Error')
        axes[0, 1].set_title('Position Errors')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Error (rad)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Applied Torques
        for i in range(num_joints):
            axes[1, 0].plot(time_steps, result['actions'][:, i], label=f'Joint {i+1} Torque')
        axes[1, 0].set_title('Applied Torques')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Torque (Nm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Rewards and Total Error
        ax4_twin = axes[1, 1].twinx()
        axes[1, 1].plot(time_steps, result['rewards'], 'orange', label='Reward')
        total_error = np.sum(np.abs(position_errors), axis=1)
        ax4_twin.plot(time_steps, total_error, 'red', label='Total Abs Error')
        
        axes[1, 1].set_title('Rewards and Total Error')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Reward', color='orange')
        ax4_twin.set_ylabel('Total Error (rad)', color='red')
        axes[1, 1].legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def test_scenarios(self):
        """Test different scenarios in PyBullet"""
        if self.env_type == 'single_joint':
            scenarios = [
                {'name': 'Small Step', 'target': 0.2, 'description': 'Small target angle'},
                {'name': 'Large Step', 'target': 1.0, 'description': 'Large target angle'},
                {'name': 'Negative Target', 'target': -0.8, 'description': 'Negative target angle'},
                {'name': 'Zero Target', 'target': 0.0, 'description': 'Return to zero'},
                {'name': 'Max Positive', 'target': 1.5, 'description': 'Maximum positive angle'},
                {'name': 'Max Negative', 'target': -1.5, 'description': 'Maximum negative angle'}
            ]
        else:
            scenarios = [
                {'name': 'All Zeros', 'target': [0.0, 0.0, 0.0], 'description': 'All joints to zero'},
                {'name': 'Positive Step', 'target': [0.5, 0.5, 0.5], 'description': 'All joints positive'},
                {'name': 'Mixed Targets', 'target': [0.8, -0.5, 0.3], 'description': 'Mixed positive/negative'},
                {'name': 'Large Motion', 'target': [1.2, -1.0, 0.8], 'description': 'Large joint movements'},
                {'name': 'Alternating', 'target': [1.0, -1.0, 1.0], 'description': 'Alternating positions'}
            ]
        
        results = {}
        
        print("Testing different scenarios in PyBullet...")
        print("-" * 60)
        
        for scenario in scenarios:
            print(f"Testing: {scenario['name']} - {scenario['description']}")
            print(f"Target: {scenario['target']}")
            
            result = self.simulate_episode(
                max_steps=500, 
                target_position=scenario['target']
            )
            
            results[scenario['name']] = result
            
            print(f"  Total Reward: {result['total_reward']:.3f}")
            print(f"  Episode Length: {result['steps']} steps")
            
            if self.env_type == 'single_joint':
                final_error = result['infos'][-1].get('position_error', 0)
                print(f"  Final Position Error: {final_error:.3f} rad")
            else:
                # Calculate final errors for multi-joint
                final_state = result['states'][-1]
                final_errors = final_state[3*self.action_dim:4*self.action_dim]
                print(f"  Final Joint Errors: {[f'{e:.3f}' for e in final_errors]}")
            
            print()
        
        return results
    
    def performance_analysis(self, results):
        """Analyze performance across multiple scenarios"""
        print("=" * 60)
        print("PYBULLET SIMULATION PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        total_rewards = [r['total_reward'] for r in results.values()]
        episode_lengths = [r['steps'] for r in results.values()]
        
        print(f"Number of Scenarios: {len(results)}")
        print(f"Mean Total Reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
        print(f"Reward Range: [{np.min(total_rewards):.3f}, {np.max(total_rewards):.3f}]")
        print(f"Mean Episode Length: {np.mean(episode_lengths):.1f} steps")
        print()
        
        # Success rate analysis
        if self.env_type == 'single_joint':
            success_count = 0
            for result in results.values():
                if result['infos']:
                    final_error = abs(result['infos'][-1].get('position_error', float('inf')))
                    if final_error < 0.1:  # Success threshold
                        success_count += 1
            
            success_rate = success_count / len(results) * 100
            print(f"Success Rate (error < 0.1 rad): {success_rate:.1f}%")
        
        print("=" * 60)
    
    def close(self):
        """Close the simulation environment"""
        self.env.close()


def main():
    """Main PyBullet simulation function"""
    print("=" * 60)
    print("PYBULLET ACTUATOR CONTROL SIMULATOR")
    print("=" * 60)
    
    # Choose environment type
    print("Select environment type:")
    print("1. Single Joint Actuator")
    print("2. Multi-Joint Robot Arm")
    env_choice = input("Enter choice (1 or 2): ").strip()
    
    env_type = 'single_joint' if env_choice == '1' else 'robot_arm'
    
    # Choose rendering
    render = input("Enable PyBullet GUI? (y/n): ").lower() == 'y'
    
    # Initialize simulator
    model_path = f"checkpoints/pybullet_{env_type}_model.pth"
    simulator = PybulletSimulator(model_path, render=render, env_type=env_type)
    
    try:
        while True:
            print(f"\nPyBullet {env_type.replace('_', ' ').title()} Simulator")
            print("1. Run single simulation")
            print("2. Test scenarios")
            print("3. Save model")
            print("4. Load model")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                if env_type == 'single_joint':
                    target = float(input("Target position in radians (default 0.5): ") or "0.5")
                else:
                    print("Enter target positions for 3 joints (space-separated):")
                    targets_input = input("Default: 0.5 -0.5 0.3: ") or "0.5 -0.5 0.3"
                    target = [float(x) for x in targets_input.split()]
                
                result = simulator.simulate_episode(target_position=target, max_steps=500)
                
                if input("Visualize results? (y/n): ").lower() == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"pybullet_plots/simulation_{env_type}_{timestamp}.png"
                    os.makedirs("pybullet_plots", exist_ok=True)
                    simulator.visualize_episode(result, save_path=save_path)
            
            elif choice == '2':
                results = simulator.test_scenarios()
                simulator.performance_analysis(results)
                
                if input("Visualize a scenario? (y/n): ").lower() == 'y':
                    scenario_names = list(results.keys())
                    print("\nAvailable scenarios:")
                    for i, name in enumerate(scenario_names):
                        print(f"{i+1}. {name}")
                    
                    scenario_idx = int(input("Select scenario: ")) - 1
                    if 0 <= scenario_idx < len(scenario_names):
                        selected_name = scenario_names[scenario_idx]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = f"pybullet_plots/scenario_{selected_name.replace(' ', '_')}_{timestamp}.png"
                        os.makedirs("pybullet_plots", exist_ok=True)
                        simulator.visualize_episode(results[selected_name], save_path=save_path)
            
            elif choice == '3':
                save_path = input(f"Save path (default: checkpoints/pybullet_{env_type}_model.pth): ").strip()
                if not save_path:
                    save_path = f"checkpoints/pybullet_{env_type}_model.pth"
                simulator.save_model(save_path)
            
            elif choice == '4':
                load_path = input("Model path to load: ").strip()
                if simulator.load_model(load_path):
                    print("Model loaded successfully!")
                else:
                    print("Failed to load model!")
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice!")
    
    finally:
        simulator.close()
        print("PyBullet simulation closed.")

if __name__ == "__main__":
    main()
