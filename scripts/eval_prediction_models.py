import argparse
import enum
import yaml
from train_eval.trajectory_prediction.train_eval import TrainAndEvaluator
from torch.utils.tensorboard import SummaryWriter
import os
from utils.file_io import extract_folder_file_list

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--models_path", help="path of model", required=True)
parser.add_argument("-pre_str", "--prefix_str", help="string to indicates model files", default="net_pgp")
args = parser.parse_args()

model_file_list = extract_folder_file_list(args.models_path)
print("Get model list num = {}.".format(len(model_file_list)))
result_list = []
for fi, model_file in enumerate(model_file_list):
  if not args.prefix_str in model_file:
    print("Skip file = {}.".format(model_file))
    continue

  model_path = os.path.join(args.models_path, model_file)
  print(f"[{fi}]: Read model from {model_file}.")

  # Load model default config
  with open(os.path.join(model_path, 'config.yaml'), 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)
  checkpoint_path = os.path.join(model_path, 'checkpoints/best.tar')

  # Reload eval relevant metrics in 'conf/eval_metrics.yml'
  import envs.__init__
  with open(os.path.join(os.path.dirname(envs.__init__.__file__), 
            'conf/eval_metrics.yml')) as yaml_file2:
    eval_cfg = yaml.safe_load(yaml_file2)
    cfg['val_metrics'] = eval_cfg['val_metrics']
    cfg['val_metric_args'] = eval_cfg['val_metric_args']

  # Evaler
  teval_tool = TrainAndEvaluator(cfg, checkpoint_path=checkpoint_path, writer=None)
  get_dict = teval_tool.eval()
  
  get_dict['proportion'] = float(model_file.split('_pp')[1].split('_')[0])
  get_dict['model_file'] = model_file

  result_list.append(get_dict)

non_metric_keys = ['minibatch_count', 'time_elapsed', 'proportion', 'model_file', ]
result_list = sorted(result_list, key=lambda e: -e.__getitem__('proportion'))
print("*"*30)
for dict_dt in result_list:
  proportion = dict_dt['proportion']
  batch_num = int(dict_dt['minibatch_count'])
  print("[{:.1f}%]: ".format(proportion*100.0), end="")
  for key in dict_dt.keys():
    if not key in non_metric_keys:
      metric_key = key + " "*(5 - len(key))
      metric_val = dict_dt[key] / batch_num
      print(metric_key + ':', format(metric_val, '0.2f'), end=", ")
  print(";")
