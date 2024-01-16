from multiprocessing import context
import paper_plot.utils as plot_utils
import numpy as np
from matplotlib import pyplot as plt 
import math
import collections
from typing import Tuple, Dict

from paper_plot.color_schemes import FIVE_COLOR_SCHEME1

##############################################################################################
def axis_feature(vlist, vreso):
  vreso_2 = vreso * 0.5
  v_min = np.min(vlist)
  v_max = np.max(vlist)
  v_min_id = math.floor((v_min + vreso_2) / vreso)
  v_max_id = math.floor((v_max + vreso_2) / vreso)
  vidlist = np.floor((vlist - v_min + vreso_2) / vreso)
  return {'vmin': v_min, 'vmax': v_max,
          'idmin': v_min_id, 'idmax': v_max_id}, np.array(vidlist, dtype=int)

def map_amount2color(amount):
  if amount <= 10:
    return FIVE_COLOR_SCHEME1[0]
  elif amount <= 100:
    return FIVE_COLOR_SCHEME1[1]
  elif amount <= 1000:
    return FIVE_COLOR_SCHEME1[2]
  elif amount <= 10000:
    return FIVE_COLOR_SCHEME1[3]
  else:
    return FIVE_COLOR_SCHEME1[4]

##############################################################################################
def plot_amount_bar1d(axis_example, xlist: np.ndarray, xreso: float):
  '''
  Compute the x values in x list in each grid (reso = xreso), and plot the bar
  :param axis_example, the axis being plotted
  '''
  xlist = np.array(xlist)
  total_num :int= xlist.shape[0]
  x_amounts = {}
  xreso_2 = xreso * 0.5
  
  for x in xlist:
    idx :int = math.floor((x + xreso_2) / xreso)
    
    if not idx in x_amounts.keys():
      x_amounts[idx] = 0
    x_amounts[idx] += 1

  x_amounts = sorted(list(x_amounts.items()), key=lambda p: p[0])
  x_amounts = np.array(x_amounts)
  # print("with x_amounts shape=", x_amounts.shape)
  axis_example.bar(x_amounts[:, 0] * xreso - xreso_2, x_amounts[:, 1] / total_num)

# @brief compute the amount of 2d-points in each grid2d
# scatter it with cmap=amount density
def plot_heatmap2d(xlist, xreso, ylist, yreso, 
                   full_color_num=5000.0, cmap_mode='Wistia'):
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)
  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)
  grid_xy_map = dict()
  grid_num = np.zeros([xsiz, ysiz])
  for xid, yid in zip(xidlist, yidlist):
    _xid = xid
    _yid = yid
    _x = _xid * xreso + xf['vmin']
    _y = _yid * yreso + yf['vmin']
    id = (_yid * xsiz + _xid)
    if not id in grid_xy_map:
      grid_xy_map[id] = [_x, _y, _xid, _yid]
    grid_num[_xid, _yid] = grid_num[_xid, _yid] + 1.

  grid_sumup = np.sum(grid_num)
  plt_data = []
  for _, dta in grid_xy_map.items():
    _x, _y, _xid, _yid = dta
    sample_num = grid_num[_xid, _yid]
    dense = 0.0
    if sample_num > 5:
      dense = float(sample_num) / grid_sumup
    plt_data.append([_x, _y, dense])

  plt_data = np.array(plt_data)
  heat_max = max(np.max(plt_data[:, 2]),full_color_num/grid_sumup)
  plt.scatter(plt_data[:, 0], plt_data[:, 1], c=plt_data[:, 2],
              vmin=0.0, vmax=heat_max, cmap=plt.cm.get_cmap(cmap_mode), 
              marker='s')

# @brief comput the amount of 2d-points in each column, plotbox it
def plot_boxplot(axis_example, xlist, xreso, ylist, 
                 enable_showfliers = True,fliercolor='blue', axis_coef=1.0) -> Tuple:
  xf, xidlist = axis_feature(xlist, xreso)

  box_dict = dict()
  for y, xid in zip(ylist, xidlist):
    if not xid in box_dict:
      box_dict[xid] = []
    box_dict[xid].append(y)

  od_box_dict = collections.OrderedDict(sorted(box_dict.items()))

  box_data = []
  x_ticks_labels = []
  x_ticks_poses = []
  common_max_v = -np.inf
  common_min_v = np.inf
  box_colors = []

  xy_datas = []
  for key_xid, content in od_box_dict.items():
    _x = key_xid * xreso + xf['vmin']
    box_data.append(content)
    num_sample = len(content)
    samples_maxv = np.max(np.array(content))

    # [[key_x_value, mean_y, std_y], ...]
    xy_datas.append([key_xid, _x, np.mean(content), np.std(content)])

    x_ticks_labels.append("{}".format(round(_x, 1)))
    x_ticks_poses.append(_x)
    common_max_v = max(common_max_v, samples_maxv)
    common_min_v = min(common_min_v, samples_maxv)
    box_colors.append(map_amount2color(num_sample))
  
  # common_max_v = common_max_v + 0.2
  # for xpos, num in zip(x_ticks_poses, num_samples):
  #   plt.text(xpos, common_max_v, "{}".format(num))
  boxpros = dict(linestyle='-', linewidth=0.25, color='k')
  medianprops = dict(linestyle='--', linewidth=1.0, color="g")
  meanprops = dict(linestyle='-', linewidth=1.0, color="g")
  flier_marker=dict(markeredgecolor=fliercolor, # markerfacecolor='red', 
                    marker='o', markersize=1)
  
  bps = axis_example.boxplot(box_data,
    boxprops=boxpros, 
    positions=x_ticks_poses,
    showfliers=enable_showfliers, # flier: abnormal value
    flierprops=flier_marker,
    meanprops=meanprops,
    medianprops=medianprops,
    # notch=1,
    patch_artist=True, # enable fill with color
    showmeans=True,
    meanline=True,
    widths=(1.0 * axis_coef * 0.4)
    )
  for patch, color in zip(bps['boxes'], box_colors):
    patch.set_facecolor(color)

  plot_utils.axis_set_xticks(axis_example, 
    tick_values=x_ticks_poses, tick_labels=x_ticks_labels)

  return x_ticks_poses, x_ticks_labels, np.array(xy_datas)

# @brief compute the amount of 3d-points in each grid3d
# scatter3d it with cmap=amount density
def plot_heatmap3d(axis3d_example,
                   xlist, xreso, ylist, yreso, 
                   zlist, zreso, cmap_mode='cool'):
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)
  zf, zidlist = axis_feature(zlist, zreso)
  # print("debug xf", xf)
  # print("debug yf", yf)
  # print("debug zf", zf)

  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)
  zsiz = int(zf['idmax']-zf['idmin'] + 1)
  xysiz = xsiz * ysiz
  grid_xyz_map = dict()
  grid_num = np.zeros([xsiz, ysiz, zsiz])
  for xid, yid, zid in zip(xidlist, yidlist, zidlist):
    _xid = xid
    _yid = yid
    _zid = zid
    _x = _xid * xreso + xf['vmin']
    _y = _yid * yreso + yf['vmin']
    _z = _zid * zreso + zf['vmin']
    id = (_zid * xysiz + _yid * xsiz + _xid)
    if not id in grid_xyz_map:
      grid_xyz_map[id] = [_x, _y, _z, _xid, _yid, _zid]
    # print(_xid, _yid, _zid)
    grid_num[_xid, _yid, _zid] = grid_num[_xid, _yid, _zid] + 1.

  grid_sum = dict()
  for zid in range(0, zsiz):
    grid_sum[zid] = np.sum(grid_num[:, :, zid])
      
  plt_data = []
  for _, dta in grid_xyz_map.items():
    _x, _y, _z, _xid, _yid, _zid = dta
    sample_num = grid_num[_xid, _yid, _zid]
    dense = 0.0
    if sample_num > 5:
      dense = float(sample_num) / grid_sum[zid]
    plt_data.append([_x, _y, _z, dense])

  plt_data = np.array(plt_data)
  axis3d_example.scatter(plt_data[:, 0], plt_data[:, 1], plt_data[:, 2],
                          c=plt_data[:, 3], vmin=0.0, vmax=1.0, 
                          cmap=plt.cm.get_cmap(cmap_mode), marker='s')

# @brief plot grid3d data, cmap=grid_value
def plot_grid3d_heatmap(axis3d_example, xyyawv_array, cid,
                        cvmin=0.0, cvmax=20.0, cmap_mode='coolwarm'):
  gridx_list = xyyawv_array[:, 1]
  gridy_list = xyyawv_array[:, 2]
  gridyaw_list = xyyawv_array[:, 3]
  gridmax_clist = xyyawv_array[:, cid]

  axis3d_example.scatter(gridx_list, gridy_list, gridyaw_list,
                          vmin=cvmin, vmax=cvmax,
                          c=gridmax_clist,
                          cmap=plt.cm.get_cmap(cmap_mode), marker='s')

# @brief group xyz points coording to z axis, and plot it in 3d, 
# and using grid to prevent the points is too dense
def plot_points3d(axis3d_example, 
                  xlist, xreso, ylist, yreso, zlist, zreso, 
                  xlabel, ylabel, zlabel,
                  txt_label="l"):
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)
  zf, zidlist = axis_feature(zlist, zreso)
  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)
  zsiz = int(zf['idmax']-zf['idmin'] + 1)
  xysiz = xsiz * ysiz

  data_dicts = dict()
  grid_num = np.zeros([xsiz, ysiz, zsiz])
  for x, y, z, xid, yid, zid in zip(xlist, ylist, zlist, xidlist, yidlist, zidlist):
    _xid = xid
    _yid = yid
    _zid = zid

    _z = _zid * zreso + zf['vmin']
    if not _zid in data_dicts:
      data_dicts[_zid] = []
    if grid_num[_xid, _yid, _zid] < 10: # limit rviz points
      data_dicts[_zid].append([x, y, _z])
    grid_num[_xid, _yid, _zid] = grid_num[_xid, _yid, _zid] + 1.

  plot_utils.axis_set_xlabel(axis3d_example, xlabel)
  plot_utils.axis_set_ylabel(axis3d_example, zlabel)
  plot_utils.axis_set_zlabel(axis3d_example, ylabel)
  for zid in zidlist:
    _zid = zid

    pcloud = np.array(data_dicts[_zid])
    _z = _zid * zreso + zf['vmin']
    axis3d_example.scatter(pcloud[:, 0], pcloud[:, 2], pcloud[:, 1],
                            label=txt_label+"="+str(_z), marker='.')

# @brief group xyz points coording to z axis, and plot it in 2d, 
# and using grid reso in x,y to prevent the points is too dense
#     using grid reso in z to determine number of figures
def plot_piecies_points3d(fig,
                          xlist, xreso, ylist, yreso, zlist, zreso, 
                          xlabel, ylabel, zlabel, inflation_y=1.0):
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)
  zf, zidlist = axis_feature(zlist, zreso)
  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)
  zsiz = int(zf['idmax']-zf['idmin'] + 1)
  xysiz = xsiz * ysiz

  data_dicts = dict()
  grid_num = np.zeros([xsiz, ysiz, zsiz])
  for x, y, z, xid, yid, zid in zip(xlist, ylist, zlist, xidlist, yidlist, zidlist):
    _xid = xid
    _yid = yid
    _zid = zid

    _z = _zid * zreso + zf['vmin']
    if not _zid in data_dicts:
      data_dicts[_zid] = []
    if grid_num[_xid, _yid, _zid] < 10: # limit rviz points
      data_dicts[_zid].append([x, y, _z])
    grid_num[_xid, _yid, _zid] = grid_num[_xid, _yid, _zid] + 1.

  fig_num = zsiz
  row_num = round(math.sqrt(float(fig_num)))
  col_num = row_num
  if (row_num * col_num) < fig_num:
    col_num = row_num + 1
  axess = []
  fid = 0
  for zid in range(0, zsiz):
    pcloud = np.array(data_dicts[zid])
    _z = zid * zreso + zf['vmin']
    fid = fid + 1
    axess.append(fig.add_subplot(row_num,col_num,fid))
    plot_utils.subfig_reset()
    plot_utils.axis_set_title(axess[-1], zlabel+"{:.1f}".format(_z))
    plot_utils.axis_set_xlabel(axess[-1], xlabel)
    plot_utils.axis_set_ylabel(axess[-1], ylabel)

    plt.scatter(pcloud[:, 0], pcloud[:, 1]*inflation_y, marker='.', s=1)

def plot_prob2d(axis_example,
                xlist, xreso, ylist, yreso, 
                flaglist,
                cond_num: int= 25,
                cmap_mode='winter') -> Dict:
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)

  _x = np.linspace(xf['vmin'], xf['vmax'], math.ceil((xf['vmax'] - xf['vmin']) / xreso))
  _y = np.linspace(yf['vmin'], yf['vmax'], math.ceil((yf['vmax'] - yf['vmin']) / yreso))
  _xx, _yy = np.meshgrid(_x, _y)
  _xlist, _ylist = _xx.ravel(), _yy.ravel()

  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)

  true_num = np.zeros([xsiz, ysiz])
  grid_num = np.zeros_like(true_num)
  for x, y, xid, yid, flag in zip(xlist, ylist, xidlist, yidlist, flaglist):
    _x = xid * xreso + xf['vmin']
    _y = yid * yreso + yf['vmin']

    if flag > 0.5:
      true_num[xid, yid] = true_num[xid, yid] + 1.

    grid_num[xid, yid] = grid_num[xid, yid] + 1.

  xreso_2 = xreso * 0.5
  yreso_2 = yreso * 0.5

  grid_xyc = []
  blanks = []
  cond_num = float(cond_num)
  for x, y in zip(_xlist, _ylist):
    xid :int = math.floor((x - xf['vmin'] + xreso_2) / xreso)
    yid :int = math.floor((y - yf['vmin'] + yreso_2) / yreso)

    if grid_num[xid, yid] >= cond_num:
      prob = true_num[xid, yid] / grid_num[xid, yid]
      grid_xyc.append([x, y, prob])
    else:
      blanks.append([x, y])

  grid_xyc = np.array(grid_xyc)
  blanks = np.array(blanks)

  axis_example.scatter(
    grid_xyc[:, 0], grid_xyc[:, 1], c=grid_xyc[:, 2],
    vmin=0.0, vmax=1.0, cmap=plt.cm.get_cmap(cmap_mode), 
    marker='s')
  axis_example.scatter(
    blanks[:, 0], blanks[:, 1], c='grey', marker='s')

# @brief collect the flags among x-y space and plot their probabilities using bar3d
def plot_prob_bar3d(axis3d_example,
                    xlist, xreso, ylist, yreso, 
                    flaglist,
                    cond_num: int= 25,
                    shape_flag: bool=True) -> Dict:
  xf, xidlist = axis_feature(xlist, xreso)
  yf, yidlist = axis_feature(ylist, yreso)

  _x = np.linspace(xf['vmin'], xf['vmax'], math.ceil((xf['vmax'] - xf['vmin']) / xreso))
  _y = np.linspace(yf['vmin'], yf['vmax'], math.ceil((yf['vmax'] - yf['vmin']) / yreso))
  _xx, _yy = np.meshgrid(_x, _y)
  _xlist, _ylist = _xx.ravel(), _yy.ravel()

  xsiz = int(xf['idmax']-xf['idmin'] + 1)
  ysiz = int(yf['idmax']-yf['idmin'] + 1)

  true_num = np.zeros([xsiz, ysiz])
  grid_num = np.zeros_like(true_num)
  for x, y, xid, yid, flag in zip(xlist, ylist, xidlist, yidlist, flaglist):
    _x = xid * xreso + xf['vmin']
    _y = yid * yreso + yf['vmin']

    if flag > 0.5:
      true_num[xid, yid] = true_num[xid, yid] + 1.

    grid_num[xid, yid] = grid_num[xid, yid] + 1.

  xreso_2 = xreso * 0.5
  yreso_2 = yreso * 0.5

  bar_xlist = []
  bar_ylist = []
  bar_top = []
  cond_num = float(cond_num)
  for x, y in zip(_xlist, _ylist):
    xid :int = math.floor((x - xf['vmin'] + xreso_2) / xreso)
    yid :int = math.floor((y - yf['vmin'] + yreso_2) / yreso)

    if grid_num[xid, yid] >= cond_num:
      prob = true_num[xid, yid] / grid_num[xid, yid]

      bar_xlist.append(x)
      bar_ylist.append(y)
      bar_top.append(prob)

  bar_bottom = np.zeros_like(bar_top)

  axis3d_example.bar3d(
    bar_xlist, bar_ylist,
    bar_bottom,
    xreso, yreso,
    bar_top, shade=shape_flag)

#######################################################################################
# TODO: add contour function
# where, examples are as follows
#
# overtake_features = plot_data[:, 0]
# giveway_features = plot_data[:, 1]

# overtake_fmin = np.min(overtake_features)
# overtake_fmax = np.max(overtake_features)
# giveway_fmin = np.min(giveway_features)
# giveway_fmax = np.max(giveway_features)

# overtake_space_range = (overtake_fmax - overtake_fmin)
# overtake_space_n = math.ceil(overtake_space_range / 0.1)
# giveway_space_range = (giveway_fmax - giveway_fmin)
# giveway_space_n = math.ceil(giveway_space_range / 0.1)

# overtake_space = np.linspace(overtake_fmin, overtake_fmax, overtake_space_n)
# giveway_space = np.linspace(giveway_fmin, giveway_fmax, giveway_space_n)
# mesh_overtake_fs, mesh_giveway_fs = np.meshgrid(overtake_space, giveway_space)

# loaded = ipmodel.load_model_from_file()
# print("load_model successfully? = {}".format(loaded))
# if not loaded:
#   raise ValueError("Did not have a valid ipModel")

# cache_inputs = []
# for of in overtake_space:
#   for gf in giveway_space:
#     cache_inputs.append([of, gf])
# cache_inputs = np.array(cache_inputs)
# cache_inputs = torch.from_numpy(cache_inputs).float()
# cache_outputs = ipmodel.forward(cache_inputs).cpu().detach().numpy()

# mesh_win_prop = []
# k = 0
# for of in overtake_space:
#   array_prop = []
#   for gf in giveway_space:
#     win_prop = cache_outputs[k, 0]
#     array_prop.append(win_prop)
#     k = k + 1
#   mesh_win_prop.append(array_prop)
# mesh_win_prop = np.array(mesh_win_prop).transpose() # ???

# fig = plt.figure()
# plot_utils.fig_reset()
# fig.canvas.set_window_title("Scene Interaction Visualization")
# title_text = fig.suptitle("")
# fig_ax1 = fig.add_subplot(1,1,1)
# plot_utils.subfig_reset()
# fig_ax1.set_title("Distribution")
# plot_utils.axis_set_xlabel(fig_ax1, "agent_i takeway ability")
# plot_utils.axis_set_ylabel(fig_ax1, "agent_j giveway ability")

# # print("shape=", mesh_overtake_fs.shape, mesh_giveway_fs.shape, mesh_win_prop.shape)
# win_samples = plot_data[plot_data[:, 2] > 0.5]
# loss_samples = plot_data[plot_data[:, 2] < 0.5]
# plt.plot(win_samples[:, 0], win_samples[:, 1], 'r.', label="agent i win", markersize=0.7)
# plt.plot(loss_samples[:, 0], loss_samples[:, 1], 'b.', label="agent i loss", markersize=0.7)
# plt.legend()

# dense_lvl = 5
# plt.contourf(mesh_overtake_fs, mesh_giveway_fs, mesh_win_prop, dense_lvl, alpha=0.75) # fill colors
# plt.colorbar()

# C = plt.contour(mesh_overtake_fs, mesh_giveway_fs, mesh_win_prop, dense_lvl, 
#                 linewidth=0.5, colors=('k',)) # add lines
# plt.clabel(C, inline=True, fontsize=12)
# plt.show()


#######################################################################################
# TODO: fill between function
# # Compute the mean of the sample. 
# y_mean = np.apply_over_axes(func=np.mean, a=y_samples, axes=0).squeeze()
# # Compute the standard deviation of the sample. 
# y_std = np.apply_over_axes(func=np.std, a=y_samples, axes=0).squeeze()
# ax1.fill_between(
#     x=x_array, 
#     y1=(y_mean - 2.0*y_std), 
#     y2=(y_mean + 2.0*y_std), 
#     color=get_color,
#     alpha=0.25, 
# )
# ax1.plot(x_array, y_mean, '-'+get_marker, 
#          color=get_color, label=scheme_tag)
# ax1.legend(loc=legend_loc)
