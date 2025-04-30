import genesis as gs
import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import VecEnv

class SoftPendulumBalanceEnv(VecEnv):
    def __init__(self, vis, device='cpu',
                 max_steps=2000,
                 max_tension=25, stiffness=25e-1,
                 tendon_radius=0.01,
                 num_envs=10,
                 reset_on_completion=True,
                 dt=0.1):

        if str.lower(device) == 'cpu':
            gs.init(backend=gs.cpu, precision="32", logging_level='warning')
        else:
            print("ERROR! Current no other device than CPU supported")
        self.vis = vis

        self.max_tension = max_tension
        self.stiffness = stiffness
        self.tendon_radius = tendon_radius
        
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))  # Tension applied to the tendon
        observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,)) # EE position (x, y) and velocity (x_dot, y_dot)

        super().__init__(num_envs,
                         observation_space,
                         action_space)
        
        self.scene = gs.Scene(
            viewer_options=
            gs.options.ViewerOptions(
                camera_pos=(0.0, 20.0, 5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=dt),
            show_viewer=vis
        )

        self.cam = self.scene.add_camera(
                res    = (1280, 960),
                pos    = (0.0, 20.0, 5),
                lookat = (0.0, 0.0, 0.5),
                fov    = 30,
                GUI    = False
            )

        self.pendulum = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/flexy_arm.urdf",  # Path to your URDF file
                fixed=True,
                merge_fixed_links=False
            )
        )

        plane = self.scene.add_entity(gs.morphs.Plane())

        self.scene.build(n_envs=self.num_envs)
        self.envs_idx = np.arange(self.num_envs)

        self.actions = np.zeros([num_envs, 1], dtype=np.float32)
        self.step_counts = np.zeros(num_envs, dtype=np.int32)

        self.MAX_STEPS = max_steps
        self.POS_THRESHOLD = 0.05
        self.VEL_THRESHOLD = 0.01
        self.RESET_ON_COMPLETION = reset_on_completion

        self.set_positions()

    def reset_(self, dones):
        
        num_resets = dones.sum()

        position = np.deg2rad((np.random.uniform(low=-10, high=10, size=[num_resets, 1]))) * np.ones((num_resets, 10))
        velocity = np.zeros([num_resets, 10])
        # Set angular position
        self.pendulum.set_dofs_position(position, envs_idx=self.envs_idx[dones])
        # Set angular velocity
        self.pendulum.set_dofs_velocity(velocity, envs_idx=self.envs_idx[dones])
        self.step_counts[dones] = np.zeros(num_resets, dtype=np.int32) 

        return self._get_observation()

    def reset(self):
        return self.reset_(np.ones(self.num_envs, dtype=bool))

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):

        self.step_counts += 1

        # Get pendulum pre-step state
        pre_state = self._get_state()
        pre_theta = pre_state[:, :10]
        pre_theta_dot = pre_state[:, 10:]

        # Get efffects of action
        actuation = self.tendon_radius * np.ones(10) * np.clip(
            self.max_tension * self.actions,
            a_min=-self.max_tension * np.ones_like(self.actions),
            a_max= self.max_tension * np.ones_like(self.actions)
        ).reshape([self.num_envs, 1])  

        # Find stiffness contribution
        stiffness_contribution = -self.stiffness * pre_theta

        # Apply torque within limits
        self.pendulum.control_dofs_force(actuation + stiffness_contribution)
        self.scene.step()

        # Get pendulum post-step state
        state = self._get_ee_state()
        p = state[:, :2]
        v = state[:, 2:]

        # Episode ends if the pendulum falls
        max_steps_reached = self.step_counts > self.MAX_STEPS
        reached_goal = (  (np.linalg.norm(p - np.array([0.0, 0.3])) < self.POS_THRESHOLD)
                        & (np.linalg.norm(v) < self.VEL_THRESHOLD))
        dones = (max_steps_reached | (reached_goal & self.RESET_ON_COMPLETION) )

        # Compute reward
        upright_bonus = -np.linalg.norm(p - np.array([0.0, 0.29]), axis=1)     # Encourage to be upright
        velocity_bonus = -0.1 * np.linalg.norm(v, axis=1)                     # Discourage high velocities
        tension_bonus = -0.1 * np.square(self.actions.flatten())              # Penalize high actuation
        step_cost = np.zeros(self.num_envs)                                   # Encourage fast solutions
        reached_goal_bonus = reached_goal * 1e2                               # Encourage to reach the goal
        rewards = (upright_bonus + velocity_bonus                             # Combine Boni
                    + tension_bonus + reached_goal_bonus + step_cost)
        # Write info dicts
        infos = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            if dones[i]:
                infos[i]["terminal_observation"] = self._get_observation()
            if max_steps_reached[i]:
                infos[i]["TimeLimit.truncated"] = True

        # Reset done environments
        self.reset_(dones=dones)

        return self._get_observation(), rewards, dones, infos
    
    def _set_state(self, theta, theta_dot=None):

        if theta_dot is None:
            theta_dot = np.zeros([self.num_envs, 10])

        assert len(theta) == self.num_envs, "Wrong number of positions given!"
        assert len(theta_dot) == self.num_envs, "Wrong number of velocities given!"

        # Set angular position
        self.pendulum.set_dofs_position(theta, envs_idx=self.envs_idx)
        # Set angular velocity
        self.pendulum.set_dofs_velocity(theta_dot, envs_idx=self.envs_idx)

    def _get_state(self):
        theta = self.pendulum.get_dofs_position()
        theta = np.arctan2(np.sin(theta), np.cos(theta))
        theta_dot = self.pendulum.get_dofs_velocity()
        return np.concatenate((theta, theta_dot), axis=1)
    
    def _get_ee_state(self):
        ee_pos = self.pendulum.get_link("arm_sphere").get_pos() - self.pendulum.get_pos()
        ee_vel = self.pendulum.get_link("arm_sphere").get_vel()

        # Only return x and z coordinates
        obs = np.concatenate((ee_pos[:, ::2], ee_vel[:, ::2]), axis=1)
        return obs
    
    def _get_observation(self):
        return self._get_ee_state()
    
    def close(self):
        pass
    
    def seed(self):
        pass

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None for _ in range(self.num_envs)]
    
    def set_attr(self, attr_name, value, indices=None):
        pass
        
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]*self.num_envs
    
    def set_positions(self, spacing=1.5):
        num_per_side = int(np.ceil(np.sqrt(self.num_envs)))  # Number of rows/cols
        x_idx, y_idx = np.meshgrid(range(num_per_side), range(num_per_side))
        
        # Select the first num_envs positions
        grid_positions = np.stack([x_idx.ravel(), y_idx.ravel()], axis=1)[:self.num_envs]
        
        # Center the grid at the origin
        grid_positions = (grid_positions - np.mean(grid_positions, axis=0)) * spacing
        positions = np.column_stack([grid_positions, np.zeros(self.num_envs)])  # Add Z coordinate
        
        self.pendulum.set_pos(positions)
        self.scene.step()

    def simulate(self, steps=100, control_inputs=None):
        """
        Simulates the pendulum environment for a given number of steps.
        
        Args:
            steps (int): Number of simulation steps to run.
            control_inputs (np.array): Control torques to apply (optional, shape: [steps, num_envs, 1]).
        """
        for t in range(steps):
            # Apply control input if provided
            if control_inputs is not None:
                self.pendulum.control_dofs_force(control_inputs[t, :, :])

            # Step the simulation
            self.scene.step()

            # Retrieve and print the state for debugging (optional)
            angular_pos = self.pendulum.get_dofs_position()
            angular_vel = self.pendulum.get_dofs_velocity()
            print(f"Step {t}: Position={angular_pos}, Velocity={angular_vel}")
