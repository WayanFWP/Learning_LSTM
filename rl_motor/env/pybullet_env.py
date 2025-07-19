import pybullet as p
import pybullet_data
import gym
import numpy as np
from gym import spaces
import time
import os

class PybulletActuatorEnv(gym.Env):
    def __init__(self, render=False, time_step=1/240):
        super(PybulletActuatorEnv, self).__init__()
        
        self.render_mode = render
        self.time_step = time_step
        self.max_steps = 1000
        self.current_step = 0
        
        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(time_step)
        
        # Action space: torque command [-10, 10] Nm
        self.action_space = spaces.Box(
            low=np.array([-10.0]), 
            high=np.array([10.0]), 
            dtype=np.float32
        )
        
        # Observation space: [joint_pos, joint_vel, target_pos, pos_error, joint_torque]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -50, -np.pi, -np.pi, -50]),
            high=np.array([np.pi, 50, np.pi, np.pi, 50]),
            dtype=np.float32
        )
        
        # Load robot and setup
        self.robot_id = None
        self.joint_id = 0
        self.target_position = 0.0
        self.previous_error = 0.0
        self.integral_error = 0.0
        
        self._setup_robot()
        
    def _setup_robot(self):
        """Create a simple robot with a revolute joint"""
        # Create base (fixed)
        base_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Create link (movable)
        link_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 0.05])
        link_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.05, 0.05], rgbaColor=[1, 0.5, 0, 1])
        
        # Create multi-body robot
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, 1],
            linkMasses=[0.5],
            linkCollisionShapeIndices=[link_collision],
            linkVisualShapeIndices=[link_visual],
            linkPositions=[[0.5, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1]]
        )
        
        # Set joint properties
        p.changeDynamics(self.robot_id, self.joint_id, 
                        linearDamping=0.1, 
                        angularDamping=0.1,
                        jointDamping=0.5,
                        restitution=0.5,
                        lateralFriction=1.0)
        
        # Enable joint force/torque sensing
        p.enableJointForceTorqueSensor(self.robot_id, self.joint_id, 1)
        
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.previous_error = 0.0
        self.integral_error = 0.0
        
        # Reset robot position
        initial_angle = np.random.uniform(-np.pi/4, np.pi/4)
        p.resetJointState(self.robot_id, self.joint_id, initial_angle, 0)
        
        # Set random target position
        self.target_position = np.random.uniform(-np.pi/2, np.pi/2)
        
        # Let simulation settle
        for _ in range(10):
            p.stepSimulation()
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        # Get joint state
        joint_state = p.getJointState(self.robot_id, self.joint_id)
        joint_pos = joint_state[0]
        joint_vel = joint_state[1]
        joint_torque = joint_state[3]  # Applied torque
        
        # Calculate position error
        pos_error = self.target_position - joint_pos
        
        # Normalize angles to [-pi, pi]
        pos_error = np.arctan2(np.sin(pos_error), np.cos(pos_error))
        joint_pos = np.arctan2(np.sin(joint_pos), np.cos(joint_pos))
        target_pos = np.arctan2(np.sin(self.target_position), np.cos(self.target_position))
        
        observation = np.array([
            joint_pos,      # Current joint position
            joint_vel,      # Current joint velocity  
            target_pos,     # Target position
            pos_error,      # Position error
            joint_torque    # Current applied torque
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        """Take a step in the environment"""
        # Apply torque command
        torque = np.clip(action[0], -10.0, 10.0)
        p.setJointMotorControl2(
            self.robot_id,
            self.joint_id,
            p.TORQUE_CONTROL,
            force=torque
        )
        
        # Step simulation
        p.stepSimulation()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, torque)
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Additional termination conditions
        joint_pos, joint_vel, target_pos, pos_error, joint_torque = obs
        if abs(joint_pos) > np.pi:  # Joint limit violation
            done = True
            reward -= 10.0
        
        if abs(joint_vel) > 50:  # Velocity limit violation
            done = True
            reward -= 10.0
        
        info = {
            'target_position': self.target_position,
            'position_error': pos_error,
            'applied_torque': torque,
            'joint_velocity': joint_vel
        }
        
        return obs, reward, done, info
    
    def _calculate_reward(self, obs, applied_torque):
        """Calculate reward based on performance"""
        joint_pos, joint_vel, target_pos, pos_error, joint_torque = obs
        
        # Position error penalty (main objective)
        position_reward = -abs(pos_error) * 10.0
        
        # Velocity penalty (encourage smooth motion)
        velocity_penalty = -abs(joint_vel) * 0.1
        
        # Torque efficiency penalty (encourage minimal torque)
        torque_penalty = -abs(applied_torque) * 0.01
        
        # Bonus for being close to target
        if abs(pos_error) < 0.1:  # Within 0.1 radians
            position_reward += 5.0
            
        if abs(pos_error) < 0.05:  # Very close
            position_reward += 10.0
        
        # Stability bonus (low velocity near target)
        if abs(pos_error) < 0.1 and abs(joint_vel) < 1.0:
            stability_bonus = 5.0
        else:
            stability_bonus = 0.0
        
        total_reward = position_reward + velocity_penalty + torque_penalty + stability_bonus
        
        return total_reward
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.render_mode:
            # Add target position visualization
            target_visual = p.createVisualShape(
                p.GEOM_SPHERE, 
                radius=0.05, 
                rgbaColor=[0, 1, 0, 0.5]
            )
            
            # Calculate target position in 3D space
            target_x = 0.5 * np.cos(self.target_position)
            target_y = 0.5 * np.sin(self.target_position)
            target_z = 1.0
            
            # Remove old target marker if exists
            if hasattr(self, 'target_marker_id'):
                p.removeBody(self.target_marker_id)
            
            # Create new target marker
            self.target_marker_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=target_visual,
                basePosition=[target_x, target_y, target_z]
            )
            
            time.sleep(self.time_step)
    
    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)
    
    def set_target_position(self, target):
        """Set a specific target position"""
        self.target_position = np.clip(target, -np.pi/2, np.pi/2)
    
    def get_joint_state(self):
        """Get detailed joint state information"""
        joint_state = p.getJointState(self.robot_id, self.joint_id)
        return {
            'position': joint_state[0],
            'velocity': joint_state[1],
            'reaction_forces': joint_state[2],
            'applied_torque': joint_state[3]
        }


class PybulletRobotArmEnv(gym.Env):
    """More complex robot arm environment with multiple joints"""
    
    def __init__(self, render=False, num_joints=3):
        super(PybulletRobotArmEnv, self).__init__()
        
        self.render_mode = render
        self.num_joints = num_joints
        self.time_step = 1/240
        self.max_steps = 1000
        self.current_step = 0
        
        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
        # Action space: torque commands for each joint
        self.action_space = spaces.Box(
            low=-5.0 * np.ones(num_joints), 
            high=5.0 * np.ones(num_joints), 
            dtype=np.float32
        )
        
        # Observation space: [joint_positions, joint_velocities, target_positions, position_errors]
        obs_dim = num_joints * 4  # pos, vel, target, error for each joint
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(obs_dim),
            high=np.inf * np.ones(obs_dim),
            dtype=np.float32
        )
        
        self.robot_id = None
        self.joint_ids = list(range(num_joints))
        self.target_positions = np.zeros(num_joints)
        
        self._setup_robot_arm()
        
    def _setup_robot_arm(self):
        """Create a multi-joint robot arm"""
        # Base
        base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length=0.2, rgbaColor=[0.5, 0.5, 0.5, 1])
        
        # Links
        link_masses = [0.5] * self.num_joints
        link_collisions = []
        link_visuals = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []
        
        for i in range(self.num_joints):
            # Create link
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.2])
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.2], 
                                       rgbaColor=[1, 0.5, 0, 1])
            
            link_collisions.append(collision)
            link_visuals.append(visual)
            link_positions.append([0, 0, 0.3 + i * 0.4])
            link_orientations.append([0, 0, 0, 1])
            link_parent_indices.append(i)  # Each link connected to previous
            link_joint_types.append(p.JOINT_REVOLUTE)
            
            # Alternate joint axes for more interesting motion
            if i % 2 == 0:
                link_joint_axes.append([0, 1, 0])  # Y-axis rotation
            else:
                link_joint_axes.append([1, 0, 0])  # X-axis rotation
        
        # Create robot
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[0, 0, 0.5],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collisions,
            linkVisualShapeIndices=link_visuals,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0, 0, 0]] * self.num_joints,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * self.num_joints,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes
        )
        
        # Set joint properties
        for joint_id in self.joint_ids:
            p.changeDynamics(self.robot_id, joint_id, 
                           linearDamping=0.1, 
                           angularDamping=0.1,
                           jointDamping=0.5)
            p.enableJointForceTorqueSensor(self.robot_id, joint_id, 1)
    
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        
        # Reset joint positions
        for joint_id in self.joint_ids:
            initial_angle = np.random.uniform(-np.pi/4, np.pi/4)
            p.resetJointState(self.robot_id, joint_id, initial_angle, 0)
        
        # Set random target positions
        self.target_positions = np.random.uniform(-np.pi/2, np.pi/2, self.num_joints)
        
        # Let simulation settle
        for _ in range(10):
            p.stepSimulation()
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation"""
        joint_positions = []
        joint_velocities = []
        position_errors = []
        
        for i, joint_id in enumerate(self.joint_ids):
            joint_state = p.getJointState(self.robot_id, joint_id)
            joint_pos = joint_state[0]
            joint_vel = joint_state[1]
            
            joint_positions.append(joint_pos)
            joint_velocities.append(joint_vel)
            
            # Calculate position error
            pos_error = self.target_positions[i] - joint_pos
            pos_error = np.arctan2(np.sin(pos_error), np.cos(pos_error))
            position_errors.append(pos_error)
        
        # Combine all observations
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            self.target_positions,
            position_errors
        ]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        """Take a step in the environment"""
        # Apply torque commands
        for i, joint_id in enumerate(self.joint_ids):
            torque = np.clip(action[i], -5.0, 5.0)
            p.setJointMotorControl2(
                self.robot_id,
                joint_id,
                p.TORQUE_CONTROL,
                force=torque
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Get new observation
        obs = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(obs, action)
        
        # Check if done
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return obs, reward, done, {}
    
    def _calculate_reward(self, obs, action):
        """Calculate reward for multi-joint arm"""
        # Extract position errors (last num_joints elements)
        position_errors = obs[-self.num_joints:]
        
        # Position error penalty
        position_reward = -np.sum(np.abs(position_errors)) * 5.0
        
        # Torque efficiency penalty
        torque_penalty = -np.sum(np.abs(action)) * 0.01
        
        # Bonus for all joints close to target
        if np.all(np.abs(position_errors) < 0.1):
            position_reward += 10.0
        
        return position_reward + torque_penalty
    
    def render(self, mode='human'):
        """Render the environment"""
        if self.render_mode:
            time.sleep(self.time_step)
    
    def close(self):
        """Close the environment"""
        p.disconnect(self.physics_client)
