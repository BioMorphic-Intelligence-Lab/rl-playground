import argparse

import numpy as np
import genesis as gs
from stable_baselines3 import PPO
from env import SoftPendulumBalanceEnv 

def read_po():
    parser = argparse.ArgumentParser(description="Testing the pendulum swingup")
    parser.add_argument('--record', action='store_true', help="Flag on whether to record a video of the simulation")
    parser.add_argument('--n_envs', type=int, default=12, help="Number of parallel environmnents")
    return parser.parse_args()

def main():
    args = read_po()
    
    model = PPO.load("flexy_swingup_model.zip")
    env = SoftPendulumBalanceEnv(num_envs=args.n_envs, vis=True, max_steps=200, reset_on_completion=False, dt=0.01)
    obs = env.reset()
    a = np.zeros([env.num_envs, 1], dtype=np.float32)

    state = np.array(obs)

    if args.record:
        env.cam.start_recording()

    for _ in range(200):
        for j in range(env.num_envs):
            a[j] = model.predict(state[j, :])[0][0]
        state, rewards, dones, infos = env.step(a)
        #print(f"State: {state} ; rewards: {rewards}")
        if args.record:
            env.cam.render()

    if args.record:
        env.cam.stop_recording(save_to_filename='video.mp4', fps=60)

if __name__ == "__main__":
    main()