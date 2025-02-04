import argparse

import numpy as np
import genesis as gs
from stable_baselines3 import PPO
from env import PendulumBalanceEnv 

def read_po():
    parser = argparse.ArgumentParser(description="Testing the pendulum swingup")
    parser.add_argument('--record', action='store_true', help="Flag on whether to record a video of the simulation")
    parser.add_argument('--n_envs', type=int, default=12, help="Number of parallel environmnents")
    return parser.parse_args()

def baseline_control(pos, vel):
    ctrl = np.zeros_like(pos)
    
    # If we reached the upper position we do PD control
    ctrl[np.abs(pos) < np.deg2rad(15)] = 25 * (0 - pos) + 50 * (0 - vel)

    # Otherwise energy shaping
    e_pot = 0.5 * np.sin(pos)
    e_kin = 0.5 * np.square(vel)

    e_err = (1.0 - (e_pot + e_kin))

    vel_corr = vel
    if vel == 0:
        vel_corr = 0.001

    ctrl[np.abs(pos) >= np.deg2rad(15)] = 1000 * vel_corr * e_err

    return ctrl

def main():
    args = read_po()
    
    model = PPO.load("swingup_model")
    env = PendulumBalanceEnv(num_envs=args.n_envs, vis=True, max_steps=3000, reset_on_completion=False, dt=0.01)

    angles = np.deg2rad(np.random.uniform(low=-15 + 180, high=15 + 180, size=[env.num_envs, 1]))
    state = np.concatenate(
        [np.sin(angles), np.cos(angles),
        np.zeros([env.num_envs, 1])], axis=1
    )
    env._set_state(theta=angles.reshape([env.num_envs, 1]),
                theta_dot=state[:, 2].reshape([env.num_envs, 1]))

    a = np.zeros(env.num_envs)

    if args.record:
        env.cam.start_recording()

    for _ in range(1500):
        for j in range(env.num_envs):
            a[j] = model.predict(state[j, :])[0][0]
            #a[j] = baseline_control(pos[j], vel[j])
        state, rewards, dones, infos = env.step(a)
        if args.record:
            env.cam.render()

    if args.record:
        env.cam.stop_recording(save_to_filename='video.mp4', fps=60)

if __name__ == "__main__":
    main()