import os

from sb3_contrib.a2c_mask import MaskableA2C
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.wrappers import SafetyShield
from sb3_contrib.common.wrappers import SafetyMask
from sb3_contrib.common.wrappers import SafetyCBF
from sb3_contrib.common.safety import SafeRegion
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.tqc import TQC

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
