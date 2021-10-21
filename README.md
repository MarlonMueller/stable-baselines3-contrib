# General

# Adapted Project Structure

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
