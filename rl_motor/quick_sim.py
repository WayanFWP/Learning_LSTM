import torch
import numpy as np
import matplotlib.pyplot as plt
from env.actuator_env import ActuatorEnv
from model.agent import DDPGAgent

def quick_simulation():
    """Quick and simple simulation function"""
    
    # Initialize environment and agent
    env = ActuatorEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = DDPGAgent(state_dim, action_dim)
    
    # Try to load trained model if exists
    model_path = "checkpoints/ddpg_model.pth"
    try:
        checkpoint = torch.load(model_path)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        print(f"Loaded trained model from {model_path}")
    except:
        print("No trained model found, using random weights")
    
    # Run simulation
    state = env.reset()
    states = [state.copy()]
    actions = []
    rewards = []
    target_torques = []
    predicted_torques = []
    
    print("Running simulation...")
    for step in range(100):
        # Get action from agent
        action = agent.act(state)
        
        # Calculate target torque
        pos_error, velocity, kp, kd, tau_ff = state
        target_torque = kp * pos_error + kd * velocity + tau_ff
        
        # Store data
        actions.append(action[0])
        target_torques.append(target_torque)
        predicted_torques.append(action[0])
        
        # Take step
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        states.append(next_state.copy())
        
        state = next_state
        
        if done:
            break
    
    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    target_torques = np.array(target_torques)
    predicted_torques = np.array(predicted_torques)
    
    print(f"Simulation completed in {len(actions)} steps")
    print(f"Total reward: {np.sum(rewards):.3f}")
    print(f"Mean absolute torque error: {np.mean(np.abs(target_torques - predicted_torques)):.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Quick Actuator Simulation Results', fontsize=14)
    
    time_steps = np.arange(len(actions))
    
    # Position and velocity
    axes[0, 0].plot(time_steps, states[:-1, 0], 'b-', label='Position Error')
    axes[0, 0].plot(time_steps, states[:-1, 1], 'r-', label='Velocity')
    axes[0, 0].set_title('Position Error & Velocity')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Torque comparison
    axes[0, 1].plot(time_steps, target_torques, 'r-', label='Target', alpha=0.7)
    axes[0, 1].plot(time_steps, predicted_torques, 'b--', label='Predicted', alpha=0.7)
    axes[0, 1].set_title('Torque Comparison')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Torque')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Torque error
    torque_error = np.abs(target_torques - predicted_torques)
    axes[1, 0].plot(time_steps, torque_error, 'm-')
    axes[1, 0].set_title('Absolute Torque Error')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Error')
    axes[1, 0].grid(True)
    
    # Rewards
    axes[1, 1].plot(time_steps, rewards, 'orange')
    axes[1, 1].set_title('Rewards')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('quick_simulation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'target_torques': target_torques,
        'predicted_torques': predicted_torques,
        'total_reward': np.sum(rewards),
        'steps': len(actions)
    }

if __name__ == "__main__":
    results = quick_simulation()
