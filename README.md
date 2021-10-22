# General

# Adapted Project Structure

## SB3

```
Stable Baselines3 Contrib/
└── sb3_contrib/
    ├── a2c_mask
    │   ├── a2c_mask.py
    │   └── policies.py
    └── common
        ├── envs/
        │   └── pendulum/
        │       ├── assets/
        │       │   ├── README.md
        │       │   └── clockwise.png
        │       └── math_pendulum_env.py
        ├── safety/
        │   └── safe_region.py
        └── wrappers/
           ├── safety_cbf.py
           ├── safety_mask.py
           └── safety_shield.py

```

## Auxiliary

```
./
├── callbacks/
│   ├── pendulum_rollout.py
│   └── pendulum_train.py
├── matlab/
│   ├── gainMatrix.m
│   ├── mathematicalPendulum.m
│   └── regionOfAttraction.m
├── .gitignore
├── README.md
├── main.py
├── notebook.ipynb
├── pendulum_roa.py
├── requirements.txt
├── train.sh
├── util.py

```
