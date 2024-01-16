import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as pltcm
import matplotlib.pyplot as plt
from type_utils.trajectory_point import TrajectoryPoint
from sumocr.interface.ego_vehicle import EgoVehicle
from matplotlib.lines import Line2D
from matplotlib.text import Text
from typing import List, Dict
import copy
import paper_plot.utils as paper_utils
from paper_plot.color_schemes import COLOR_RGB_MAP, FOUR_COLOR_SCHEME1, FIVE_COLOR_SCHEME1
import matplotlib.colors as mcolors

import thirdparty.config
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import ObstacleRole, DynamicObstacle
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle

from preprocessor.commonroad.raster_renderer import RasterRenderer
from utils.transform import XYYawTransform
from envs.commonroad.utils import read_params, rgba2rgb
from .utility import limit_agent_length_width

from envs.config import ENABLE_CLOSEDLOOP_SIMULATION_SAVE_PDF

def read_raster_image_jsons():
  return {
    'ego': read_params('conf/ego_params.json'), 
    'pred': read_params('conf/rviz_raster_predictions.json'), 
    'plan': read_params('conf/rviz_raster_plan.json'),
  }

class SimulationVisualizer:
  def __init__(self):
    self.rviz_params = read_raster_image_jsons()
    self.ego_params = self.rviz_params['ego']
    self.rviz_inch2pixels = 100.0 # per inch is 100 pixels (default dpi in matplotlib)
    self.rviz_pixel2meters = 0.2  # per pixel = 0.2 meter
    self.rviz_one_inch2meters = self.rviz_inch2pixels * self.rviz_pixel2meters
    self.rviz_view_range_m = 100.0
    self.rviz_front_view_percentage = 0.8

    self.plan_max_v = 30.0
    self.plan_cmap=plt.cm.get_cmap('jet')

    self.__enable_ax2 = True
    self.__enable_paper_pdf_format = ENABLE_CLOSEDLOOP_SIMULATION_SAVE_PDF

    if self.__enable_paper_pdf_format:
      self.__enable_ax2 = False
      self.rviz_view_range_m = 60.0
      self.rviz_pixel2meters = 0.5
      self.rviz_one_inch2meters = self.rviz_inch2pixels * self.rviz_pixel2meters

      self.rviz_view_length_inch = self.rviz_view_range_m / self.rviz_one_inch2meters
      self.rviz_view_width_inch = self.rviz_view_length_inch / 0.75
    else:
      self.rviz_view_length_inch = self.rviz_view_range_m / self.rviz_one_inch2meters
      self.rviz_view_width_inch = self.rviz_view_length_inch * 1.0

  def _init_rnd(self, scenario: Scenario):
    paper_utils.fig_reset()
    if not self.__enable_ax2:
      # self.ax1 = plt.subplot(1, 1, 1)
      self.fig, self.ax1 = plt.subplots(1, 1)
      self.ax1.spines['top'].set_visible(False)
      self.ax1.spines['left'].set_visible(False)
      self.ax1.spines['right'].set_visible(False)
      self.ax1.spines['bottom'].set_visible(False)
    else:
      # self.ax2 = plt.subplot(1, 2, 1)
      # self.ax1 = plt.subplot(1, 2, 2)
      self.fig, (self.ax2, self.ax1) = plt.subplots(
        1, 2, gridspec_kw={'width_ratios': [1, 1]})

    paper_utils.subfig_reset()

    self.rnd1 = RasterRenderer(ax=self.ax1,
                               draw_params=self.rviz_params['pred'],
                               figsize=(self.rviz_view_length_inch, self.rviz_view_width_inch))
    self.rnd1.draw_params['time_begin'] = 0
    scenario.lanelet_network.draw(self.rnd1)

    if self.__enable_ax2:
      self.rnd2 = RasterRenderer(ax=self.ax2,
                                draw_params=self.rviz_params['plan'],
                                figsize=(self.rviz_view_length_inch, self.rviz_view_width_inch))
      self.rnd2.draw_params['time_begin'] = 0
      scenario.lanelet_network.draw(self.rnd2)

      # cmp_norm = mpl.colors.BoundaryNorm([0., 0.5, 1., 2., 4., 20.], self.plan_cmap.N)
      # plt.colorbar(mpl.cm.ScalarMappable(norm=cmp_norm, cmap=self.plan_cmap), 
      #              ax=self.ax1, orientation='vertical', cax=self.ax1)

  def get_xy_limits(self, ego: EgoVehicle):
    '''
    Return [[x_min, x_max], [y_min, y_max]]
    '''
    view_front_m = self.rviz_view_range_m * self.rviz_front_view_percentage
    view_back_m = view_front_m - self.rviz_view_range_m
    view_left_right = self.rviz_view_range_m * 0.35
    center = XYYawTransform(x=ego.current_state.position[0],
                            y=ego.current_state.position[1],
                            yaw_radian=ego.current_state.orientation)
    coner_x_list = []
    coner_y_list = []
    for dx in [view_front_m, view_back_m]:
      for dy in [-view_left_right, view_left_right]:
        tf2corner = center.multiply_from_right(
          XYYawTransform(x=dx, y=dy))
        coner_x_list.append(tf2corner._x)
        coner_y_list.append(tf2corner._y)
    get_xy_lims = [[min(coner_x_list), max(coner_x_list)], 
                   [min(coner_y_list), max(coner_y_list)]]

    return get_xy_lims

  def get_prediction_artists(self, agent_list: List, agents_pred_trajs: List, rviz_z_base: float) -> List:
    '''
    Return list of artists given agents' prediction trajectories
    :param rviz_z_base: z base of the rviz
    '''
    prob_cmap = pltcm.get_cmap('jet')
    pred_trajs_artists = []
    color_dict = mcolors.XKCD_COLORS 
    color_keys = list(color_dict.keys())
    color_num: int = len(color_dict)
    for agent_state, pred_info in zip(agent_list, agents_pred_trajs):
      # for each agent
      for traj_info in pred_info:
        agent_idx = traj_info['agent_idx']
        prob = traj_info['prob']
        traj = np.array(traj_info['trajectory'])
        if traj.shape[0] == 0:
          continue

        if self.__enable_paper_pdf_format:
          valid_list = [301654, 301656, 301660, 301661, 300852, 300849, 
                        300850, 300854, 305227, 305228, 305224,
                        303974, 303981, 303982]
          if agent_idx in valid_list:
            agent_idx_str = "{}".format(agent_idx)
            agent_idx_str = agent_idx_str[-3:]
            
            text_idx = Text(x=traj[0, 0]+2.0, y=traj[0, 1], text=f"[{agent_idx_str}]", 
              zorder=rviz_z_base+1.2, size=3, color='k')
            text_idx.set_bbox(dict(facecolor='w', edgecolor='none', pad=-0.05))

            text_v = Text(x=traj[0, 0]+2.0, y=traj[0, 1]+2.25, text=f"{round(agent_state['velocity'], 1)}m/s", 
              zorder=rviz_z_base+1.15, size=3, color='r')
            text_v.set_bbox(dict(facecolor='w', edgecolor='none', pad=-0.05))

            pred_trajs_artists.append(text_idx)
            pred_trajs_artists.append(text_v)
        else:
          pred_trajs_artists.append(
            Text(x=traj[0, 0], y=traj[0, 1], text=f"{agent_idx}", 
                zorder=rviz_z_base+1.2, size=4)
          )

        # >> color: probability map version
        # pred_trajs_artists.append(
        #   Line2D(traj[:, 0], traj[:, 1], 
        #          linestyle=':', linewidth=1,
        #          color=prob_cmap(prob), 
        #          zorder=rviz_z_base+0.1)
        # )
        # >> color: agent index map version
        color_idx: int= agent_idx % color_num
        pred_trajs_artists.append(
          Line2D(traj[:, 0], traj[:, 1], 
                 linestyle=':', linewidth=1,
                 color=color_dict[color_keys[color_idx]], #  color=prob_cmap(prob), 
                 zorder=rviz_z_base+0.1)
        )
    return pred_trajs_artists

  def get_behavior_artists(self, ref_behaviors: Dict) -> List:
    '''
    Return list of artists given reference behaviors
    '''
    zorder_base = 15.1
    get_artists = []

    rline_color = {
      'succ': COLOR_RGB_MAP['yellow'], 
      'left': COLOR_RGB_MAP['light_blue'],
      'right': COLOR_RGB_MAP['green'],
    }
    for bkey, content in ref_behaviors.items():
      for path in content['candidate_paths']:
        demo_xys = path['path_samples_xyyawcurs']
  
        get_artists.append(
          Line2D(demo_xys[:, 0], demo_xys[:, 1], 
                 linestyle='-', linewidth=1,
                 color=rline_color[bkey], zorder=zorder_base, 
                 markersize=0.1)
        )

    return get_artists

  def get_planning_artists(self, frame_index: int, axis: mpl.axes.Axes, planning_result: Dict) -> List:
    '''
    Return list of artists given planning result
    :axis: axis to plot
    '''
    # still has a lane-stop solution when it does not has a valid result.
    # if planning_result['has_result'] == False:
    #   return []

    if isinstance(planning_result['tva_xyyawcur_array'], np.ndarray) == False:
      return []

    zorder_base = 20.1

    itv :int= 10 # inverval
    stva_array = planning_result['stva_array']
    tva_xyyawcur_array = planning_result['tva_xyyawcur_array']
    speed_limits = planning_result['speed_limits']

    scatter_artist = axis.scatter(
      tva_xyyawcur_array[::itv, 3], tva_xyyawcur_array[::itv, 4],
      c='blue', marker='.', s=3.0, zorder=zorder_base)
    # vmin=0.0, vmax=self.plan_max_v, 
    # c=tva_xyyawcur_array[::itv, 1], cmap=self.plan_cmap, 

    get_artists = [scatter_artist]
    rviz_min_gap_s = 8.0
    last_rviz_s = -1e+10
    for s, tva_xyyawcur, v_u in zip(stva_array[::itv, 0], tva_xyyawcur_array[::itv, :], speed_limits[::itv]):
      if (s - last_rviz_s < rviz_min_gap_s):
        continue
      last_rviz_s = s
      if self.__enable_paper_pdf_format:
        text_v = Text(x=tva_xyyawcur[3]+3.0, y=tva_xyyawcur[4], 
          text=f" {round(tva_xyyawcur[1], 1)}m/s", 
          zorder=zorder_base+0.1, size=3)
        text_v.set_bbox(dict(facecolor='w', edgecolor='none', pad=-0.05))

        get_artists.append(text_v)
        break # only plot the initial speed
      else:
        get_artists.append(
          Text(x=tva_xyyawcur[3], y=tva_xyyawcur[4], 
              text=f"_{round(tva_xyyawcur[1], 1)}/{round(v_u, 1)} m/s", 
              zorder=zorder_base+0.1, size=4)
        )

    return get_artists

  def save_fig(self, video_path: str,
                     scenario: Scenario,
                     frame_index: int,
                     ego: EgoVehicle,
                     ego_view_limits: List,
                     ref_behaviors: Dict,
                     planning_result: Dict,
                     agent_list: List, 
                     agents_pred_trajs: List):
    '''
    Visualize the scenario with egoVechile, dynamic obstacles and their predicitons.
    :param video_path: video path to store images
    :param scenario: scenario of the simulation
    :param frame_index: frame index of the simulation
    :param ego: ego vehicle
    :param ego_view_limits: [[x_min, x_max], [y_min, y_max]]
    :param ref_behaviors: reference path and behaviors
    :param planning_result: dict of planning result
    :param agent_list: list of agent's state & information
    :param agents_pred_trajs: list of agent's prediction trajectories
           for example = [
             agent0's trajectories: [{prob, traj_i}, ...],
             agent1's trajectories: [{prob, traj_i}, ...],
             ...
           ]
    '''
    rviz_z_base = 19.0

    is_first_frame: bool = (frame_index == 0)
    if is_first_frame:
      plt.close('all')
      self._init_rnd(scenario)

    self.rnd1.draw_params['time_begin'] = frame_index
    self.rnd1.clear(keep_static_artists=True)

    if self.__enable_ax2:
      self.rnd2.draw_params['time_begin'] = frame_index
      self.rnd2.clear(keep_static_artists=True)

    # draw obstacles
    obs_list = scenario.obstacles_by_position_intervals(
      [Interval(ego_view_limits[0][0], ego_view_limits[0][1]), 
       Interval(ego_view_limits[1][0], ego_view_limits[1][1])], 
      tuple(ObstacleRole), time_step=frame_index)

    # obs_list = []
    # for o in cache_obs_list:
    #   nlength, nwidth = limit_agent_length_width(o.obstacle_shape.length, o.obstacle_shape.width)
    #   new_shape = Rectangle(length=nlength, width=nwidth, 
    #     center=o.obstacle_shape.center, orientation=o.obstacle_shape.orientation)
    #   obs_list.append(
    #     DynamicObstacle(o.obstacle_id, o.obstacle_type, new_shape, 
    #       o.initial_state, None, o.initial_center_lanelet_ids, 
    #       o.initial_shape_lanelet_ids, o.initial_signal_state, o.signal_series)
    #   )

    # print("rviz test", len(obs_list), len(scenario.dynamic_obstacles), len(scenario.obstacles))
    for o in obs_list:
      o.draw(self.rnd1)
    if self.__enable_ax2:
      for o in obs_list:
        o.draw(self.rnd2)

    if is_first_frame:
      ego.get_dynamic_obstacle().draw(self.rnd1, draw_params=self.ego_params)
      if self.__enable_ax2:
        ego.get_dynamic_obstacle().draw(self.rnd2, draw_params=self.ego_params)
    else:
      # unkown reason, when is_first_frame, get_dynamic_obstacle() return None.
      ego.get_dynamic_obstacle(time_step=frame_index).draw(
        self.rnd1, draw_params=self.ego_params)
      if self.__enable_ax2:
        ego.get_dynamic_obstacle(time_step=frame_index).draw(
          self.rnd2, draw_params=self.ego_params)

    # get prediction trajectory artists
    pred_trajs_artists = self.get_prediction_artists(
      agent_list, agents_pred_trajs, rviz_z_base=rviz_z_base)

    # add frame index
    ego_pos = ego.current_state.position
    frame_text = []
    # if self.__enable_paper_pdf_format:
    #   frame_text = [
    #     Text(x=ego_pos[0]-7.0, y=ego_pos[1]-30.0, text=f"frame={round((frame_index / 10.0), 1)}s", 
    #         zorder=rviz_z_base+10.0, size=4, color='k')
    #   ]

    # render, where first frame is abandont
    if not is_first_frame:
      # plot ax1 & ax2
      if self.__enable_ax2:
        self.rnd1.render(add_artists=frame_text+pred_trajs_artists, manual_xy_lims=ego_view_limits)
        self.rnd1.ax.set_facecolor('w')

        behavior_artists = self.get_behavior_artists(ref_behaviors)
        plan_artists = self.get_planning_artists(frame_index, self.rnd2.ax, planning_result)

        self.rnd2.render(add_artists=(frame_text + behavior_artists + plan_artists), 
                         manual_xy_lims=ego_view_limits)
        self.rnd2.ax.set_facecolor('w')
      else:
        behavior_artists = self.get_behavior_artists(ref_behaviors)
        plan_artists = self.get_planning_artists(frame_index, self.rnd1.ax, planning_result)

        self.rnd1.render(
          add_artists=(frame_text + pred_trajs_artists + plan_artists), 
          manual_xy_lims=ego_view_limits)
        self.rnd1.ax.set_facecolor('w')


      self.fig.set_facecolor('w')

      plt.subplots_adjust(wspace=0.025, hspace=0)
  
      self.fig.savefig(
        os.path.join(video_path, 'frame_{}.png'.format(frame_index)), 
        bbox_inches='tight', pad_inches=0, dpi=300)
      if self.__enable_paper_pdf_format:
        self.fig.savefig(
          os.path.join(video_path, '_frame_{}.pdf'.format(frame_index)), 
          bbox_inches='tight', pad_inches=0, dpi=300)
  
      plt.close()
