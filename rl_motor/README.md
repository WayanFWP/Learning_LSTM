# Actuator Control with DDPG Reinforcement Learning

This project implements a Deep Deterministic Policy Gradient (DDPG) agent for actuator control in robotics applications. The model learns to predict optimal torque outputs based on position error, velocity, and control parameters.

## 🆕 **NEW: PyBullet Physics Simulation**

This project now includes **realistic physics simulation using PyBullet**! Experience true-to-life robotic actuator control with:

- **Realistic Physics**: Gravity, friction, inertia, and joint dynamics
- **3D Visualization**: Real-time rendering of robot motion
- **Multiple Environments**: Single joint actuators and multi-joint robot arms
- **Advanced Dynamics**: Collision detection, joint limits, and force/torque sensing

## Project Structure

```
rl_motor/
├── env/
│   ├── actuator_env.py          # Simple mathematical environment
│   └── pybullet_env.py          # 🆕 PyBullet physics environments
├── model/
│   └── agent.py                 # DDPG agent implementation
├── utils/
│   └── replay_buffer.py         # Experience replay buffer
├── train.py                     # Training with simple environment
├── pybullet_train.py           # 🆕 Training with PyBullet physics
├── simulate.py                  # Simple environment simulation
├── pybullet_simulate.py        # 🆕 PyBullet simulation and analysis
├── pybullet_demo.py            # 🆕 Interactive PyBullet demonstrations
├── quick_sim.py                # Quick testing script
├── demo.py                     # Comprehensive demonstrations
├── requirements.txt            # Updated with PyBullet dependencies
└── README.md                   # This file
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**New dependencies include:**
- `pybullet` - Physics simulation engine
- `pybullet-data` - Additional robot models and assets

### 2. 🚀 **Try PyBullet Demo First!**
```bash
python pybullet_demo.py
```
This provides an interactive demonstration of:
- Single joint actuator with 3D visualization
- Multi-joint robot arm simulation
- Comparison between simple and physics-based environments

### 3. Train with PyBullet Physics
```bash
python pybullet_train.py
```
Choose between:
- **Single Joint Actuator**: Learn precise position control
- **Multi-Joint Robot Arm**: Coordinate multiple joints

### 4. Test Your Trained Model
```bash
python pybullet_simulate.py
```
Interactive simulation with:
- Real-time 3D visualization
- Multiple test scenarios
- Performance analysis
- Model comparison tools

### 5. Original Simple Environment
```bash
python train.py        # Train with mathematical model
python simulate.py     # Test with simple environment
```

## 🎯 **Environment Comparison**

| Feature | Simple Environment | PyBullet Environment |
|---------|-------------------|---------------------|
| **Physics** | Mathematical model | Realistic rigid body dynamics |
| **Visualization** | 2D plots | Real-time 3D rendering |
| **Dynamics** | Linear approximation | Full inertia, friction, gravity |
| **Training Speed** | Fast | Moderate (more realistic) |
| **Realism** | Basic | High fidelity |
| **Use Case** | Algorithm development | Real robot preparation |

## 🤖 **PyBullet Environments**

### Single Joint Actuator (`PybulletActuatorEnv`)
- **Robot**: Revolute joint with rotating link
- **Observation**: [joint_pos, joint_vel, target_pos, pos_error, joint_torque]
- **Action**: Torque command [-10, 10] Nm
- **Visualization**: Green target sphere, orange rotating link

### Multi-Joint Robot Arm (`PybulletRobotArmEnv`)
- **Robot**: 3-joint articulated arm
- **Observation**: Joint positions, velocities, targets, and errors
- **Action**: Torque commands for all joints [-5, 5] Nm each
- **Visualization**: Full 3D robot arm with realistic dynamics

## Features

### 🤖 DDPG Agent
- **Actor Network**: Predicts continuous torque actions
- **Critic Network**: Evaluates state-action pairs
- **Target Networks**: Stable learning with soft updates
- **Experience Replay**: Efficient learning from past experiences

### 🎯 Multiple Environments
- **Simple Mathematical**: Fast prototyping and algorithm development
- **PyBullet Single Joint**: Realistic single actuator control
- **PyBullet Robot Arm**: Complex multi-joint coordination

### 📊 Comprehensive Analysis Tools
- **Interactive Simulation**: Menu-driven testing
- **Scenario Testing**: Predefined challenging situations
- **Performance Analysis**: Statistical evaluation and insights
- **3D Visualization**: Real-time PyBullet rendering
- **Comparison Tools**: Simple vs physics-based results

## 🚀 **PyBullet Usage Examples**

### Quick Single Joint Test
```python
from pybullet_simulate import PybulletSimulator

# Create simulator with GUI
simulator = PybulletSimulator(render=True, env_type='single_joint')

# Test specific target
result = simulator.simulate_episode(target_position=0.8, max_steps=500)
print(f"Final error: {result['infos'][-1]['position_error']:.3f} rad")

# Visualize results
simulator.visualize_episode(result, save_path="my_test.png")
simulator.close()
```

### Multi-Joint Robot Arm
```python
# Create robot arm simulator
simulator = PybulletSimulator(render=True, env_type='robot_arm')

# Test complex configuration
targets = [0.5, -0.8, 0.3]  # Joint targets in radians
result = simulator.simulate_episode(target_position=targets)

# Test predefined scenarios
scenario_results = simulator.test_scenarios()
simulator.performance_analysis(scenario_results)
```

### Training Custom Agent
```python
from pybullet_train import train_pybullet_agent

# Train for single joint control
results = train_pybullet_agent(
    env_type='single_joint',
    render=False,  # No GUI for faster training
    num_episodes=500
)

# Train for robot arm control  
results = train_pybullet_agent(
    env_type='robot_arm',
    render=True,   # Watch training progress
    num_episodes=800
)
```

## Model Architecture

### Actor Network
```
Input (5) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → Tanh → Output (1)
```

### Critic Network
```
Input (6) → FC(128) → ReLU → FC(64) → ReLU → FC(1) → Output (1)
```
*Note: Critic input is state (5) + action (1) = 6 dimensions*

## Training Configuration

### PyBullet Training Parameters
- **Single Joint Episodes**: 500 (recommended)
- **Robot Arm Episodes**: 800 (more complex)
- **Max Steps**: 500-800 per episode
- **Exploration Noise**: Decreasing from 1.0 to 0.1
- **Evaluation Frequency**: Every 25 episodes
- **Physics Time Step**: 1/240 seconds (realistic)

### Original Simple Environment
- **Episodes**: 1000 (adjustable)
- **Max Steps per Episode**: 100
- **Actor Learning Rate**: 1e-4
- **Critic Learning Rate**: 1e-3
- **Discount Factor (γ)**: 0.99
- **Soft Update Rate (τ)**: 0.005
- **Replay Buffer Size**: 100,000
- **Batch Size**: 64

## File Outputs

### PyBullet Training
- `checkpoints/pybullet_single_joint_model.pth` - Single joint model
- `checkpoints/pybullet_robot_arm_model.pth` - Robot arm model
- `pybullet_plots/training_*.png` - Training progress curves
- `pybullet_plots/simulation_*.png` - Simulation visualizations

### Training
- `checkpoints/ddpg_model.pth` - Final trained model
- `checkpoints/ddpg_model_ep*.pth` - Periodic checkpoints
- `training_plots/training_progress.png` - Training curves

### Simulation
- `simulation_plots/` - Individual episode visualizations
- `demo_plots/` - Demonstration outputs
- `quick_simulation.png` - Quick simulation results

## Performance Metrics

PyBullet environments track additional realistic metrics:

- **Physics Accuracy**: Realistic force/torque interactions
- **Settling Time**: Time to reach and maintain target position
- **Overshoot**: Maximum deviation from target during approach
- **Power Consumption**: Total energy used (torque × time)
- **Stability**: Variance in final position over time

## 🎮 **Interactive Controls**

When using PyBullet with GUI (`render=True`):

- **Mouse**: Rotate and zoom the 3D view
- **Keyboard**:
  - `Ctrl+Mouse`: Pan the view
  - `Shift+Mouse`: Zoom
  - `r`: Reset camera view
  - `g`: Toggle grid display
  - `w`: Toggle wireframe mode

## Troubleshooting

### PyBullet-Specific Issues

1. **PyBullet not found**
   ```bash
   pip install pybullet pybullet-data
   ```

2. **GUI not showing**
   - Check if running in headless environment
   - Try `render=False` for training, `render=True` for testing

3. **Slow performance with GUI**
   - Use `render=False` during training
   - Enable GUI only for testing and demonstrations

4. **Physics instability**
   - Check joint limits and torque bounds
   - Adjust time step in environment (default: 1/240s)
   - Verify collision detection settings

### Common Issues

1. **No module named 'torch'**
   ```bash
   pip install torch
   ```

2. **Model file not found**
   - Train the model first: `python train.py`
   - Or use untrained model (random weights)

3. **Poor performance**
   - Train for more episodes
   - Adjust learning rates in `agent.py`
   - Modify reward function in `actuator_env.py`

4. **Slow training**
   - Reduce number of episodes
   - Use smaller replay buffer
   - Simplify network architecture

### Tips for Better Performance

1. **Environment Tuning**:
   - Ensure reward function is well-scaled
   - Balance exploration vs exploitation
   - Consider reward shaping

2. **Network Architecture**:
   - Adjust layer sizes based on problem complexity
   - Consider batch normalization for stable training
   - Experiment with different activation functions

3. **Training Process**:
   - Monitor training curves for convergence
   - Save checkpoints frequently
   - Use multiple random seeds for robust evaluation

## 🔬 **Research Applications**

This PyBullet implementation is suitable for:

- **Algorithm Development**: Test RL algorithms on realistic physics
- **Sim-to-Real Transfer**: Prepare for real robot deployment
- **Robotic Research**: Study actuator control, dynamics, and learning
- **Educational Purposes**: Understand robotics and physics simulation
- **Benchmarking**: Compare different control approaches

## Contributing

To extend this project:

1. **Add New Robots**: Create custom URDF files and environments
2. **Implement Other Physics**: Try MuJoCo, Gazebo, or other simulators
3. **Advanced Algorithms**: Add PPO, SAC, or other RL methods
4. **Real Robot Integration**: Connect to actual hardware
5. **VR/AR Visualization**: Add immersive interfaces

## License

This project is open source and available under the MIT License.

## References

- [PyBullet Documentation](https://pybullet.org/)
- [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [PyTorch RL Examples](https://github.com/pytorch/examples/tree/master/reinforcement_learning)
