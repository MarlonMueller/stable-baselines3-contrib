# General


```
git clone https://github.com/MarlonMueller/stable-baselines3-contrib.git
git checkout feat/safety-wrappers
```
We recommend to use a virtual environment.
```
conda create -n safetyWrappers python=3.8
source activate safetyWrappers
```
Note that not all required packages are available in conda or conda-forge channels.
```
pip3 install -r requirements.txt
```
To use all provided functionality which uses MATLAB make sure to install
- [the MATLAB Engine API for Python](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
- [AROC](https://tumcps.github.io/AROC/) wich among others requires [CORA](https://tumcps.github.io/CORA/)

Make sure to follow the respective installation guideluines.
Add all MATLAB related files in ./matlab/.<br>
**Note** Proprietary AROC version as of now.

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

The following code utilizes SB3-Contrib (including the wrappers etc.) to generate and evaluate benchmarks.
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
Depending on the usage, additional folders will generate (ignored in git)

```
./
├── pdfs/ <- Generated plots
├── tensorboard/ <- Tensorboard logs
├── models/ <- Trained models

```



