#!/bin/bash

echo "Install thirdparty repositories."
if [ ! -d "./scripts/thirdparty/commonroad_io/" ]; then
  cd ./scripts/thirdparty/

  # git commonroad_io
  git clone https://gitlab.lrz.de/tum-cps/commonroad_io.git

  # git commonroad-interactive-scenarios
  git clone https://gitlab.lrz.de/tum-cps/commonroad-interactive-scenarios.git

  # kmeans-pytorch
  git clone https://github.com/subhadarship/kmeans_pytorch.git

  cd ../../
fi

echo "Install thirdparty dependencies."
# commonroad_io
pip install -r scripts/thirdparty/commonroad_io/requirements.txt

# others
pip install -r ./requirements.txt

