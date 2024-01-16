import numpy as np
from typing import Tuple

def boostrap_sampling(data_list: np.ndarray, 
                      confi_intervals: Tuple,
                      bootstrap_sample_times: int= 100, bootstrap_sample_size: int= 30) -> Tuple:
  '''
  Boostrap get median cofidence interval
  :param data_list: data array
  :param confi_intervals: confidence interval of the median
  :param bootstrap_sample_times: sample times
  :param bootstrap_sample_size: sample size at each sample
  '''
  confi_l = confi_intervals[0]
  confi_u = confi_intervals[1]

  sample_means = []
  for i in range(bootstrap_sample_times):
    _samples = np.random.choice(
      data_list, size=bootstrap_sample_size,
      replace=True)
    avg = np.mean(_samples)
    sample_means.append(avg)
  mean_mean = np.mean(sample_means)

  intervals = np.quantile(sample_means, [confi_l, confi_u])

  return mean_mean, intervals
