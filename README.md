# General

An **introduction to this project** is provided in this [**Jupyter Notebook**](https://github.com/MarlonMueller/stable-baselines3-contrib/blob/feat/safety-wrappers/notebook.ipynb).<br>
Information and code on how to **tune the environment** can be found [**here**](https://github.com/MarlonMueller/math_pendulum_tuning)

This project is primarily based on
- [Safe Reinforcement Learning for Autonomous Lane Changing Using Set-Based Prediction](https://mediatum.ub.tum.de/doc/1548735/256213.pdf)
- [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://arxiv.org/abs/2006.14171)
- [Safe Reinforcement Learning via Shielding](https://arxiv.org/abs/1708.08611)
- [End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks](https://arxiv.org/abs/1903.08792)
- [Stable-baselines3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/25)
# Installation

```
git clone https://github.com/MarlonMueller/stable-baselines3-contrib.git
git checkout feat/safety-wrappers
```
We recommend to use a virtual environment - name it ``safetyWrappers``.
```
conda create -n safetyWrappers python=3.8
source activate safetyWrappers
```
Anaconda is only an example. Note that not all required packages are available in conda or conda-forge channels.
```
pip3 install -r requirements.txt
```
Note that some packages might require further system-wide functionality.<br>

The following is only necessary if you want to use e.g. ``compute_roa()``<br>in ``pendulum_roa.py`` (see 'Benchmark' below).<br>
To use all provided functionality which uses MATLAB make sure to install
- [the MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- [AROC](https://tumcps.github.io/AROC/) wich among others requires [CORA](https://tumcps.github.io/CORA/)

Make sure to follow the respective installation guidelines.
Add all MATLAB related files in ``./matlab/``.<br>
**Note**: Proprietary AROC version as of now.



<!---
On BigSur use pyglet==1.5.11 (https://github.com/openai/gym/issues/2101)
'stable-baselines3[extra]'
pypoman
scipy
cvxopt?
python3 main.py --flag 0
./train.sh 
--->

# Adapted Project Structure

## SB3 Contrib

The following code is embedded into SB3 Contrib.

```
contrib/
└── sb3_contrib/
    ├── a2c_mask <- Action masking for A2C
    │   ├── a2c_mask.py
    │   └── policies.py
    └── common
        ├── envs/
        │   └── pendulum/ <- Mathematical pendulum environment
        │       ├── assets/
        │       │   ├── README.md
        │       │   └── clockwise.png
        │       └── math_pendulum_env.py
        ├── safety/
        │   └── safe_region.py <- General safe region class
        └── wrappers/
           ├── safety_cbf.py <- Discrete-time CBF wrapper
           ├── safety_mask.py <- Action masking wrapper
           └── safety_shield.py <- Post-posed shield wrapper

```

## Auxiliary

The following code utilizes SB3-Contrib (including the wrappers etc.) to generate and evaluate benchmarks.<br>
Note that this part should only be used to reproduce the results of the benchmark conducted in main.py.<br>
Otherwise modify/generalize it accordingly.

```
./
├── callbacks/ <- Callbacks to extend Tensorboard logs
│   ├── pendulum_rollout.py
│   └── pendulum_train.py
├── contrib/ <- See above
├── gifs/
├── matlab/ <- Matlab code
│   ├── gainMatrix.m
│   ├── mathematicalPendulum.m
│   └── regionOfAttraction.m
├── models/example <- Pretrained models
├── pdfs/example <- Precomputed plots
├── .gitignore
├── README.md
├── main.py <- Controls and configures the benchmark
├── notebook.ipynb <- Introductory notebook to the SB3-Contrib modifications
├── pendulum_roa.py <- Subclass of the safe region to support the mathematical pendulum
├── requirements.txt
├── sb3_contrib/ <- Symbol link to work with SB3-Contrib as subdirectory
├── train.sh <- Distributes main calls to isolated hardware threads
├── util.py <- Auxiliary functions

```

# Benchmark

The benchmark trains and deploys policies on the inverted pendulum task. The safety constraint it set by a precomputed region of attraction (**ROA**). Specifically, three environment configurations are tested: the default one, initializing the pendulum at the equilibrium (often denoted as **0**) and reducing the available actions value-wise (often denoted as **SAS**). Training runs include default A2C & PPO runs, and, PPO runs with all safety wrappers applied. For each wrapper configuration, the wrappers are benchmarked without and with additional reward punishment (**PUN**). For the CBF wrapper, the gamma values 0.1, 0.5 and 0.95 are tested. Deployment is done in two different ways. Firstly, the trained models are deployed using the same configuration. In other words, the safety wrappers are still used in most cases (denoted as suffix **SAFE**). Furthermore, all models are deployed without safety wrappers (denoted as suffix **UNSAFE**). 

By calling ``./tmux.sh``, the default training benchmark will be performed. ``./tmux.sh`` distributes main calls to isolated hardware threads. Note that this might need adaption depending on the available threads. By default, each training is repeated five times, i.e., iteration is set to five. The trained models are saved to ``./models/``. An according ``./tensorboard/`` folder will store the logs. Pretrained models are included in the repository. To use the models, **extract** them into ``./models/``.
```
tensorboard --logdir tensorboard
```
Uncomment respective parts at the **bottom** of main.py to deploy trained models (by default 5 models for 5 runs each) or to automatically generate plots, which average over all logs in a directory. The plots are saved to ``./plots/``. Precomputed plots are included in the repository. Moreover, at the end of main.py, a code block to manually deploy specific configurations is provided.

Tags logged during training
| tag        | Description      | 
| ------------- |-------------| 
| episode_reward     | Cumulated reward |
| episode_length      | Episode length   |  
| episode_time | Measured by class Monitor(gym.Wrapper) **TBD**|
| avg_abs_theta     | Average absolute angular displacement throughout the episode  |
| avg_abs_thdot     | Average absolute angular velocity throughout the episode  |
| max_abs_theta     | Maximal absolute angular displacement throughout the episode  |
| max_abs_thdot     | Maximal absolute angular velocity throughout the episode |
| avg_abs_action_rl     |  Average absolute action of the policy throughout the episode|
| max_abs_action_rl     |  Maximal absolute action of the policy throughout the episode |
| avg_reward_rl     | Average reward (excluding reward punishment) |
| safe_episode     | True iff ROA is never left |
| safe_episode_excl_approx     | True iff ROA is only left whenever the fail-safe controller is active |
| avg_abs_safety_correction     | Average absolute safety correction by the wrappers |
| max_abs_safety_correction     | Maximal absolute safety correction by the wrappers |
| avg_abs_safety_correction_mask_lqr     | " by the LQR when action masking is used  |
| max_abs_safety_correction_mask_lqr     | " by the LQR when action masking is used |
| avg_punishment     | Average reward punishment |
| rel_abs_safety_correction     | total_abs_safety_correction/total_abs_action_rl |

Tags logged during deployment
| tag        | Description      | 
| ------------- |-------------| 
| theta     | Angular displacement |
| thdot     | Angular velocity |
| action_rl     | Action of the policy |
| reward_rl     | Reward (excluding reward punishment) |
| safe     | True iff state is inside ROA |
| safe_excl_approx     | True iff state is inside ROA or fail-safe controller is active |
| safety_correction     | Safety correction by the wrapper |
| safety_correction_mask_lqr     | " by the LQR when action masking is used |
| punishment     | Reward punishment |

<!---![Tensorboard](https://github.com/MarlonMueller/stable-baselines3-contrib/blob/feat/safety-wrappers/gifs/tensorboard.png?raw=true)--->

## Immediate Future Work
Fix Tensorboard episode_time measurement<br>
Adjust for VecEnv support<br>
Environment state is assumed to be exposed<br>
