import os

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## BASIC FUNCTIONS
def get_root2folder(root_dir: str, folder_name: str) -> str:
  folder_dir = os.path.join(root_dir, folder_name)
  flag = os.path.exists(folder_dir)
  if not flag:
    os.makedirs(folder_dir) # create one empty folder

  return folder_dir

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
## ENVIRONMENTS
# COMMON
import envs.__init__
ENVS_ROOT = os.path.dirname(envs.__init__.__file__)

import conf.__init__
import yaml
get_configs = None
with open(os.path.join(os.path.dirname(conf.__init__.__file__), 'config.yaml')) as config_file:
   get_configs = yaml.safe_load(config_file)["configurations"]
print("run with get_configs=")
print(" ", get_configs)
print(" ")

## EXPERIMENTS
EXP_ROOT = get_configs["EXP_ROOT"]
if EXP_ROOT == "":
  raise ValueError("EXP_ROOT is nan, export EXP_ROOT before run the python code.")
COMMON_EXP_ROOT = get_root2folder(EXP_ROOT, 'common')

ENABLE_CLOSEDLOOP_SIMULATION_SAVE_PDF = get_configs['ENABLE_CLOSEDLOOP_SIMULATION_SAVE_PDF']

## DATASET ROOTS
# commonroad
AUTHOR_NAME=get_configs["AUTHOR_NAME"]
AUTHOR_AFFILIATION=get_configs["AUTHOR_AFFILIATION"]

# COMMONROAD: In interactive scenarios, other traffic participants react to the behavior of the ego vehicle. 
#             This is achieved by coupling CommonRoad with the traffic simulator SUMO. 
#             In our scenario database, we denote such scenarios by the suffix I in the scenario ID in 
#             contrast to scenarios with a fixed trajectory prediction T. 
#             To run these scenarios, please use the simulation scripts provided below under 
#             Interactive scenarios, which are based on the CommonRoad-SUMO interface.
#   scenario_T like DEU_A9-2_1_T-1 is scenario with true values
#   scenario_I like ZAM_Tjunction-1_65_I-1-1 is scenario supporting interactive simulation
#   scenario_S like ZAM_Urban-6_1_S-1 is scenario supporting set-based prediction
#   @note: we use hand-crafted/ folder in default.
COMMONROAD_DATA_ROOT=get_configs["COMMONROAD_DATA_ROOT"]
COMMONROAD_EXP_ROOT=get_root2folder(EXP_ROOT, 'commonroad')

import envs.commonroad.example_scenarios.__init__
COMMONROAD_EXAMPLE_SCENARIOS_PATH=os.path.dirname(envs.commonroad.example_scenarios.__init__.__file__)

## FUNCTIONS
def get_dataset_exp_folder(dataset_name: str, folder_name: str):
  if dataset_name == 'commonroad':
    return get_root2folder(COMMONROAD_EXP_ROOT, folder_name)
  else:
    raise ValueError("unsupported dataset name")
