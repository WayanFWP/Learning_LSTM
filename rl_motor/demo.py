"""
Demo script showing how to use the actuator control model simulator

This script demonstrates:
1. Quick simulation with visualization
2. Loading/saving models
3. Testing different scenarios
4. Performance analysis
"""

import os
import numpy as np
from simulate import ModelSimulator

def demo_basic_simulation():
    """Demo: Basic simulation with visualization"""
    print("=" * 60)
    print("DEMO 1: Basic Simulation")
    print("=" * 60)
    
    # Initialize simulator (will use trained model if available)
    simulator = ModelSimulator()
    
    # Run a single episode
    print("Running single episode simulation...")
    result = simulator.simulate_episode(max_steps=100)
    
    print(f"Results:")
    print(f"  Total Reward: {result['total_reward']:.3f}")
    print(f"  Episode Length: {result['steps']} steps")
    print(f"  Final Position Error: {result['states'][-1, 0]:.3f}")
    print(f"  Final Velocity: {result['states'][-1, 1]:.3f}")
    
    # Calculate torque accuracy
    torque_error = np.abs(result['target_torques'] - result['predicted_torques'])
    print(f"  Mean Torque Error: {np.mean(torque_error):.4f}")
    print(f"  Max Torque Error: {np.max(torque_error):.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    simulator.visualize_episode(result, save_path="demo_plots/basic_simulation.png", show_plot=False)
    print("Visualization saved to: demo_plots/basic_simulation.png")
    
    return simulator

def demo_multiple_episodes(simulator):
    """Demo: Multiple episodes with statistics"""
    print("\n" + "=" * 60)
    print("DEMO 2: Multiple Episodes Analysis")
    print("=" * 60)
    
    # Run multiple episodes
    print("Running 20 episodes for statistical analysis...")
    stats = simulator.run_multiple_simulations(num_episodes=20, max_steps=100)
    
    # Display performance analysis
    simulator.performance_analysis(stats)
    
    return stats

def demo_scenario_testing(simulator):
    """Demo: Testing different scenarios"""
    print("\n" + "=" * 60)
    print("DEMO 3: Scenario Testing")
    print("=" * 60)
    
    # Test predefined scenarios
    scenario_results = simulator.test_different_scenarios()
    
    # Visualize the most challenging scenario
    print("\nVisualizing 'High Position Error' scenario...")
    high_error_result = scenario_results['High Position Error']
    simulator.visualize_episode(
        high_error_result, 
        save_path="demo_plots/high_position_error_scenario.png", 
        show_plot=False
    )
    print("Scenario visualization saved to: demo_plots/high_position_error_scenario.png")
    
    return scenario_results

def demo_custom_scenario(simulator):
    """Demo: Custom scenario testing"""
    print("\n" + "=" * 60)
    print("DEMO 4: Custom Scenario")
    print("=" * 60)
    
    # Define a custom challenging scenario
    custom_initial_state = [3.0, -2.0, 0.5, 0.05, 0.5]  # High error, negative velocity, low gains
    
    print("Testing custom challenging scenario:")
    print(f"  Initial State: {custom_initial_state}")
    print("  Description: High position error, negative velocity, low gains")
    
    result = simulator.simulate_episode(
        initial_state=custom_initial_state,
        max_steps=150,  # Allow more steps for this challenging scenario
        add_noise=True,
        noise_std=0.05
    )
    
    print(f"\nCustom Scenario Results:")
    print(f"  Total Reward: {result['total_reward']:.3f}")
    print(f"  Episode Length: {result['steps']} steps")
    print(f"  Position Error Reduction: {custom_initial_state[0]:.3f} → {result['states'][-1, 0]:.3f}")
    print(f"  Velocity Change: {custom_initial_state[1]:.3f} → {result['states'][-1, 1]:.3f}")
    
    # Visualize custom scenario
    simulator.visualize_episode(
        result, 
        save_path="demo_plots/custom_challenging_scenario.png", 
        show_plot=False
    )
    print("Custom scenario visualization saved to: demo_plots/custom_challenging_scenario.png")
    
    return result

def demo_model_evaluation(simulator, stats):
    """Demo: Model evaluation and insights"""
    print("\n" + "=" * 60)
    print("DEMO 5: Model Evaluation Summary")
    print("=" * 60)
    
    # Overall performance assessment
    print("OVERALL MODEL PERFORMANCE ASSESSMENT:")
    print("-" * 40)
    
    # Reward analysis
    if stats['mean_reward'] > -0.5:
        reward_assessment = "EXCELLENT"
    elif stats['mean_reward'] > -1.0:
        reward_assessment = "GOOD"
    elif stats['mean_reward'] > -2.0:
        reward_assessment = "FAIR"
    else:
        reward_assessment = "NEEDS IMPROVEMENT"
    
    print(f"Reward Performance: {reward_assessment}")
    print(f"  Mean Reward: {stats['mean_reward']:.3f}")
    print(f"  Consistency (Std): {stats['std_reward']:.3f}")
    
    # Stability analysis
    convergence_episodes = sum(1 for r in stats['results'] if r['steps'] < 80)
    convergence_rate = convergence_episodes / len(stats['results']) * 100
    
    print(f"\nStability Analysis:")
    print(f"  Quick Convergence Rate: {convergence_rate:.1f}% (episodes < 80 steps)")
    print(f"  Mean Episode Length: {stats['mean_episode_length']:.1f} steps")
    
    # Torque accuracy across all episodes
    all_errors = []
    for result in stats['results']:
        errors = np.abs(result['target_torques'] - result['predicted_torques'])
        all_errors.extend(errors)
    
    mean_error = np.mean(all_errors)
    accuracy_percentage = max(0, (1 - mean_error) * 100)
    
    print(f"\nTorque Prediction Analysis:")
    print(f"  Mean Absolute Error: {mean_error:.4f}")
    print(f"  Accuracy Estimate: {accuracy_percentage:.1f}%")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print("-" * 20)
    if stats['mean_reward'] < -1.0:
        print("• Consider more training episodes")
        print("• Tune hyperparameters (learning rates, network architecture)")
    if stats['std_reward'] > 1.0:
        print("• Model predictions are inconsistent - consider larger network or more data")
    if mean_error > 0.3:
        print("• Torque prediction accuracy could be improved")
        print("• Consider adding more complex dynamics to the environment")
    if convergence_rate < 50:
        print("• Model takes too long to converge - consider different reward function")
    
    if stats['mean_reward'] > -0.5 and stats['std_reward'] < 0.5 and mean_error < 0.2:
        print("• Model performance is excellent! Ready for deployment.")

def main():
    """Main demo function"""
    print("ACTUATOR CONTROL MODEL - COMPREHENSIVE DEMO")
    print("=" * 60)
    print("This demo will:")
    print("1. Run basic simulations")
    print("2. Analyze performance across multiple episodes")
    print("3. Test various scenarios")
    print("4. Evaluate model capabilities")
    print("5. Provide recommendations")
    print()
    
    # Create output directory
    os.makedirs("demo_plots", exist_ok=True)
    
    # Run all demos
    simulator = demo_basic_simulation()
    stats = demo_multiple_episodes(simulator)
    scenario_results = demo_scenario_testing(simulator)
    custom_result = demo_custom_scenario(simulator)
    demo_model_evaluation(simulator, stats)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED!")
    print("=" * 60)
    print("Generated files:")
    print("• demo_plots/basic_simulation.png")
    print("• demo_plots/high_position_error_scenario.png") 
    print("• demo_plots/custom_challenging_scenario.png")
    print("\nTo run individual simulations, use:")
    print("• python quick_sim.py  (for quick testing)")
    print("• python simulate.py   (for interactive simulation)")
    print("• python train.py      (for training)")

if __name__ == "__main__":
    main()
