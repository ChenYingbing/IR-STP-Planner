import os
import thirdparty.__init__
THIRDPARTY_ROOT = os.path.dirname(thirdparty.__init__.__file__)

import sys
# commonroad-io
THIRDPARTY_COMMONROAD_IO_ROOT = os.path.join(THIRDPARTY_ROOT, "commonroad_io")
sys.path.append(THIRDPARTY_COMMONROAD_IO_ROOT)
# commonroad-interactive-scenarios
THIRDPARTY_COMMONROAD_IS_ROOT = os.path.join(THIRDPARTY_ROOT, "commonroad-interactive-scenarios")
sys.path.append(THIRDPARTY_COMMONROAD_IS_ROOT)
# nuscenes devkit
THIRDPARTY_NUSCENES_DEVKIT_ROOT = os.path.join(THIRDPARTY_ROOT, "nuscenes-devkit/python-sdk")
sys.path.append(THIRDPARTY_NUSCENES_DEVKIT_ROOT)
# interaction dataset
INTERACTION_DATASET_DEVKIT_ROOT = os.path.join(THIRDPARTY_ROOT, "interaction-dataset")
sys.path.append(INTERACTION_DATASET_DEVKIT_ROOT)
# l5kit
INTERACTION_DATASET_DEVKIT_ROOT = os.path.join(THIRDPARTY_ROOT, "l5kit-devkit/l5kit")
sys.path.append(INTERACTION_DATASET_DEVKIT_ROOT)
