# RL - Playground

This tutorial gives some examples on simple RL environments. 
Hereby the focus lies on showcasing the technical details that lead to better performance, such as vectorized environments, GPU accelerated simulation, etc...

# How to Train and Test the Environments

The two environments implemented in this repository are ``PendulumBalanceEnv`` and ``SoftPendulumBalanceEnv``, an environment to train a model to swing up and balance a rigid and a flexible pendulum respectively. 

They are trained by running the scripts ``train_swingup.py`` and ``train_flexy_swingup.py``. 
Here the default command line arguments should work to successfully train the model.

When testing the environment you can run ``test_swingup.py`` and ``test_flexy_swingup.py`` respectively. 
Make sure you are loading the correct model in the script!
Furthermore you can customize the testing environment using the commandline arguments:


| Argument | Meaning |
|---|---|
|``--record`` | Flag to record the simulation to an `mp4` video |
|``--n_envs`` | How many environments do you want to simulate simulatenously? |

