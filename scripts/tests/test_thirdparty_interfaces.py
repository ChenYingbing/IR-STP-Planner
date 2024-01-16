import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import thirdparty.config
print("> commonroad begin check: ", end="")
from commonroad.geometry.shape import Rectangle
print("pass.")

print("> nuscenes_devkit begin check: ", end="")
from nuscenes.utils.color_map import get_colormap
from nuscenes.prediction.models.backbone import ResNetBackbone
print("pass.")
