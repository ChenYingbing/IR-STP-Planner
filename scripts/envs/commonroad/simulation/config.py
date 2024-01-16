import os
import envs.config

# SCENE_DIRS: fodlers containing scenes for simulation
SCENE_DIRS = [
  os.path.join(envs.config.COMMONROAD_DATA_ROOT, 'scenarios/interactive/hand-crafted'), 
  os.path.join(envs.config.COMMONROAD_DATA_ROOT, 'scenarios/interactive/scenario-factory'),
  os.path.join(envs.config.COMMONROAD_DATA_ROOT, 'scenarios/interactive/SUMO'), 
]

# INVALID_CITY_LIST contains the abandont cities being manually picked up.
#   the sumo simulation performances among them are bad with one or more reasons:
#   1. some agents are found that will run out of the lanes during simulation.
#   2. some agents often overlap with other agents during simulation.
#   3. the simulation end time is too short (<< 10.0s): although the setting in config is to simulate 20.0s.
INVALID_CITY_LIST = [
  'DEU_Moabit', 'DEU_Moelln', 'ESP_Cambre',
  'HRV_Pula', 'ITA_Adelfia', 'ITA_CarpiCentro',
  'ITA_Foggia', 'ITA_Siderno', 'ZAM_Tjunction',
]

# INVALID_CITY_KEY contains cities with this key string that may have the following simulation bug
#   ESP_XXX scenarios can be write to .xml scenario after simulation,
#   However, they can not be readed, beacuse the traffic light format error.
INVALID_CITY_KEY = 'ESP_'

# CITY_SCENE_MAX_NUM: limits the maximum number of scenes in one city, which can balance the data distribution.
CITY_SCENE_MAX_NUM = 4
