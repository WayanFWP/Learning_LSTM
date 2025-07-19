import gym
import numpy as np
from gym import spaces

class ActuatorEnv(gym.Env):
    def __init__(self):
        super(ActuatorEnv, self).__init__()

        # Action space: output torque in range [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # Observation: [position_error, velocity, kp, kd, tau_ff]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf]*5),
            high=np.array([np.inf]*5),
            dtype=np.float32
        )

        # Initial state
        self.state = None
        self.time_step = 0
        self.max_steps = 100

    def reset(self):
        self.time_step = 0
        # Randomize initial state for training
        self.state = np.random.uniform(low=-1.0, high=1.0, size=(5,))
        return self.state

    def step(self, action):
        pos_error, velocity, kp, kd, tau_ff = self.state
        tau_out = action[0]

        # "Simulated" next state — in practice, use model/sensor data
        pos_error += velocity * 0.01  # Integrate velocity
        velocity += -tau_out * 0.01   # Simplified dynamic model

        self.state = np.array([pos_error, velocity, kp, kd, tau_ff])

        # Calculate target torque
        tau_target = kp * pos_error + kd * velocity + tau_ff

        # Reward: negative L2 error between estimated and true torque
        reward = -np.square(tau_out - tau_target)

        self.time_step += 1
        done = self.time_step >= self.max_steps

        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"State: {self.state}")
