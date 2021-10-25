# General


Information and code on how to tune the environment can be found here: https://github.com/MarlonMueller/math_pendulum_tuning

# Getting Started

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
To use all provided functionality which uses MATLAB make sure to install
- [the MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- [AROC](https://tumcps.github.io/AROC/) wich among others requires [CORA](https://tumcps.github.io/CORA/)

Make sure to follow the respective installation guidelines.
Add all MATLAB related files in ./matlab/.<br>
**Note**: Proprietary AROC version as of now.

On BigSur use pyglet==1.5.11 (https://github.com/openai/gym/issues/2101)

<!---
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
Note that this part should only be used to reproduce results. Otherwise modify/generalize it accordingly.

```
./
├── callbacks/ <- Callbacks to extend Tensorboard logs
│   ├── pendulum_rollout.py
│   └── pendulum_train.py
├── contrib/ <- See above
├── gifs/
├── matlab/ <- Matlab code primarily to compute the ROA
│   ├── gainMatrix.m
│   ├── mathematicalPendulum.m
│   └── regionOfAttraction.m
├── pdfs/ <- Generated plots
├── models/ <- Trained models
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

During training tensorboard logs

| tag        | Description      | 
| ------------- |-------------| 
| episode_reward     | Cumulated reward |
| episode_length      | Episode length   |  
| episode_time | Measured by class Monitor(gym.Wrapper) TBD|
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

An according tensorboard folder will generate.

```
tensorboard --logdir tensorboard
```

<!---![Tensorboard](https://github.com/MarlonMueller/stable-baselines3-contrib/blob/feat/safety-wrappers/gifs/tensorboard.png?raw=true)--->

By calling ./train.sh the training benchmark will be performed. train.sh distributes main calls to isolated hardware threads. Note that this might need adaption depending on the available threads. By default, each training is done five times. The trained models are saved to ./models. Uncomment respective parts in main.py to deploy trained models or to automatically generate averaged plots. The plots are saved to ./plots. For futher insights we refer to main.py.
