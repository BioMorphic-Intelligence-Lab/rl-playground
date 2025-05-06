import os
import sys
import torch
from env import PendulumBalanceEnv
from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import VecMonitor

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # Assuming flat observation space
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def main(argv):

    models_dir = "models/PPO"
    logdir = "logs"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    env = VecMonitor(PendulumBalanceEnv(num_envs=100, vis=False, max_steps=3000))

    if "-c" in argv:
        argument = argv[argv.index("-c") + 1]
        if argument.isdigit():
            number = int(argument)
            model = PPO.load(f"models/PPO/{number}.0")  
            model.set_env(env)
        else:
            number = 0
            model = PPO.load(argument)  
            model.set_env(env)

    else:
        number = 0
        net_arch_dict = dict({"pi": [32, 32], "vf": [32, 32]})

        policy_kwargs = dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=64),
            activation_fn=torch.nn.Tanh,
            net_arch=net_arch_dict,
            log_std_init=0,
            ortho_init=True
        )
        
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir,
                     policy_kwargs=policy_kwargs)
    
    # Train the agent
    TIMESTEPS = 1e6
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS,
                    reset_num_timesteps=False,
                    tb_log_name="PPO",
                    progress_bar=True)
        model.save(f"{models_dir}/{number + iters*TIMESTEPS}")

if __name__ == "__main__":
    main(sys.argv)
