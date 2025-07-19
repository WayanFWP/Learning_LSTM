"""
Quick PyBullet Demo Script

This demonstrates the PyBullet physics simulation for actuator control.
Shows both single joint and multi-joint robot arm simulations.
"""

from pybullet_simulate import PybulletSimulator
import numpy as np
import os

def demo_single_joint():
    """Demo single joint actuator with PyBullet physics"""
    print("=" * 60)
    print("DEMO: Single Joint Actuator in PyBullet")
    print("=" * 60)
    
    # Create simulator with GUI
    simulator = PybulletSimulator(
        model_path=None,  # Use untrained model for demo
        render=True,      # Show PyBullet GUI
        env_type='single_joint'
    )
    
    print("Watch the PyBullet window - you'll see a rotating actuator!")
    print("The orange link should try to reach the green target sphere.")
    
    try:
        # Test different target positions
        targets = [0.5, -0.8, 1.2, 0.0, -1.0]
        
        for i, target in enumerate(targets):
            print(f"\nTest {i+1}: Moving to target position {target:.1f} radians")
            
            result = simulator.simulate_episode(
                max_steps=300,
                target_position=target
            )
            
            print(f"  Completed in {result['steps']} steps")
            print(f"  Total reward: {result['total_reward']:.2f}")
            
            if result['infos']:
                final_error = abs(result['infos'][-1].get('position_error', 0))
                print(f"  Final position error: {final_error:.3f} radians")
            
            input("  Press Enter to continue to next target...")
    
    finally:
        simulator.close()

def demo_robot_arm():
    """Demo multi-joint robot arm with PyBullet physics"""
    print("=" * 60)
    print("DEMO: Multi-Joint Robot Arm in PyBullet")
    print("=" * 60)
    
    # Create simulator with GUI
    simulator = PybulletSimulator(
        model_path=None,  # Use untrained model for demo
        render=True,      # Show PyBullet GUI
        env_type='robot_arm'
    )
    
    print("Watch the PyBullet window - you'll see a 3-joint robot arm!")
    print("Each joint will try to reach its target position.")
    
    try:
        # Test different configurations
        configurations = [
            [0.0, 0.0, 0.0],      # Home position
            [0.5, -0.5, 0.3],     # Mixed configuration
            [1.0, 0.0, -0.8],     # Extended reach
            [-0.8, 1.2, 0.5],     # Complex pose
            [0.2, 0.2, 0.2]       # Small angles
        ]
        
        for i, config in enumerate(configurations):
            print(f"\nTest {i+1}: Moving to configuration {config}")
            
            result = simulator.simulate_episode(
                max_steps=400,
                target_position=config
            )
            
            print(f"  Completed in {result['steps']} steps")
            print(f"  Total reward: {result['total_reward']:.2f}")
            
            # Calculate final errors
            final_state = result['states'][-1]
            final_errors = final_state[-3:]  # Last 3 elements are position errors
            print(f"  Final joint errors: {[f'{e:.3f}' for e in final_errors]}")
            
            input("  Press Enter to continue to next configuration...")
    
    finally:
        simulator.close()

def demo_comparison():
    """Compare simple environment vs PyBullet simulation"""
    print("=" * 60)
    print("DEMO: Environment Comparison")
    print("=" * 60)
    
    print("This demo shows the difference between:")
    print("1. Simple mathematical model (original)")
    print("2. PyBullet physics simulation (realistic)")
    print()
    
    # Simple environment simulation
    print("Running simple environment simulation...")
    from simulate import ModelSimulator
    simple_sim = ModelSimulator(model_path=None)
    simple_result = simple_sim.simulate_episode(max_steps=100)
    
    print(f"Simple Environment Results:")
    print(f"  Total reward: {simple_result['total_reward']:.2f}")
    print(f"  Episode length: {simple_result['steps']} steps")
    
    # PyBullet simulation (headless for comparison)
    print("\nRunning PyBullet simulation...")
    pybullet_sim = PybulletSimulator(
        model_path=None,
        render=False,  # No GUI for comparison
        env_type='single_joint'
    )
    
    pybullet_result = pybullet_sim.simulate_episode(max_steps=300, target_position=0.5)
    
    print(f"PyBullet Environment Results:")
    print(f"  Total reward: {pybullet_result['total_reward']:.2f}")
    print(f"  Episode length: {pybullet_result['steps']} steps")
    
    if pybullet_result['infos']:
        final_error = abs(pybullet_result['infos'][-1].get('position_error', 0))
        print(f"  Final position error: {final_error:.3f} radians")
    
    pybullet_sim.close()
    
    print("\nComparison:")
    print("- Simple environment uses basic mathematical models")
    print("- PyBullet provides realistic physics, inertia, friction, etc.")
    print("- PyBullet is more challenging but more realistic for real robots")

def main():
    """Main demo function"""
    print("PYBULLET ACTUATOR CONTROL DEMONSTRATION")
    print("=" * 60)
    print("This demo shows realistic physics simulation using PyBullet.")
    print("PyBullet provides:")
    print("• Realistic rigid body dynamics")
    print("• Gravity, friction, and inertia effects") 
    print("• Joint limits and collision detection")
    print("• Real-time 3D visualization")
    print()
    
    # Create output directories
    os.makedirs("pybullet_plots", exist_ok=True)
    
    while True:
        print("\nSelect demo:")
        print("1. Single Joint Actuator (with PyBullet GUI)")
        print("2. Multi-Joint Robot Arm (with PyBullet GUI)")
        print("3. Environment Comparison (Simple vs PyBullet)")
        print("4. Quick Performance Test")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            print("\nStarting single joint demo...")
            print("NOTE: PyBullet GUI window will open. Close it or press Ctrl+C to stop.")
            try:
                demo_single_joint()
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
            except Exception as e:
                print(f"\nDemo error: {e}")
        
        elif choice == '2':
            print("\nStarting robot arm demo...")
            print("NOTE: PyBullet GUI window will open. Close it or press Ctrl+C to stop.")
            try:
                demo_robot_arm()
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
            except Exception as e:
                print(f"\nDemo error: {e}")
        
        elif choice == '3':
            try:
                demo_comparison()
            except Exception as e:
                print(f"\nComparison error: {e}")
        
        elif choice == '4':
            print("\nRunning quick performance test...")
            try:
                # Quick test without GUI
                simulator = PybulletSimulator(render=False, env_type='single_joint')
                
                print("Testing 5 random targets...")
                total_rewards = []
                
                for i in range(5):
                    target = np.random.uniform(-1, 1)
                    result = simulator.simulate_episode(max_steps=200, target_position=target)
                    total_rewards.append(result['total_reward'])
                    print(f"  Test {i+1}: Target={target:.2f}, Reward={result['total_reward']:.2f}")
                
                avg_reward = np.mean(total_rewards)
                print(f"\nAverage reward: {avg_reward:.2f}")
                
                simulator.close()
                
            except Exception as e:
                print(f"\nPerformance test error: {e}")
        
        elif choice == '5':
            print("Exiting demo...")
            break
        
        else:
            print("Invalid choice! Please select 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo terminated by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("\nMake sure PyBullet is installed: pip install pybullet")
