from typing import Dict, Any
from envs.format import DatasetBasicIO

class DatasetInteractionIO(DatasetBasicIO):
  def __init__(self):
    super().__init__(file_end_str='.interaction.bin')

  @staticmethod
  def read_data(read_dict_data: Dict) -> Dict[str, Any]:
    '''
    Return list of trajectories for each case.
    '''
    dict_data = read_dict_data
    # for case_str, dt in read_dict_data.items():

    return dict_data
