# modified from commonroad.visualization.mp_renderer
# where tag '[modify]' indicates the part being modified.

import math
import math
import os
from collections import defaultdict
from typing import Dict, Set, Any

import matplotlib as mpl
import matplotlib.artist as artists
import matplotlib.collections as collections
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.text as text
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb, to_hex
from matplotlib.path import Path

import thirdparty.config
import commonroad.geometry.shape
import commonroad.prediction.prediction
import commonroad.scenario.obstacle
from commonroad.common.util import Interval
from commonroad.geometry.shape import *
from commonroad.planning.goal import GoalRegion
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.prediction.prediction import Occupancy, TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork, LineMarking
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle, ObstacleRole, SignalState, PhantomObstacle, \
    EnvironmentObstacle, Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import TrafficLightState, TrafficLight, TrafficSign
from commonroad.scenario.trajectory import Trajectory, State
from commonroad.visualization.icons import supported_icons, get_obstacle_icon_patch
from commonroad.visualization.param_server import ParamServer
from commonroad.visualization.traffic_sign import draw_traffic_light_signs
from commonroad.visualization.util import LineDataUnits, collect_center_line_colors, get_arrow_path_at, colormap_idx, \
    line_marking_to_linestyle, traffic_light_color_dict, get_tangent_angle, approximate_bounding_box_dyn_obstacles, \
    get_vehicle_direction_triangle

__author__ = "Luis Gressenbuch"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = [""]
__version__ = "2022.1"
__maintainer__ = "Luis Gressenbuch"
__email__ = "commonroad@lists.lrz.de"
__status__ = "Released"

traffic_sign_path = os.path.join(os.path.dirname(__file__), 'traffic_signs/')


class ZOrders:
    # Map
    LANELET_POLY = 9.0
    INCOMING_POLY = 9.1
    CROSSING_POLY = 9.2
    CENTER_BOUND = 10.0
    LIGHT_STATE_OTHER = 10.0
    LIGHT_STATE_GREEN = 10.05
    DIRECTION_ARROW = 10.1
    SUCCESSORS = 11.0
    STOP_LINE = 11.0
    RIGHT_BOUND = 12.0
    LEFT_BOUND = 12.0
    # Obstacles
    OBSTACLES = 20.0
    CAR_PATCH = 20.0
    # Labels
    LANELET_LABEL = 30.2
    STATE = 100.0
    LABELS = 1000.0
    # Values added to base value from drawing parameters
    INDICATOR_ADD = 0.2
    BRAKING_ADD = 0.2
    HORN_ADD = 0.1
    BLUELIGHT_ADD = 0.1


class RasterRenderer(IRenderer):
    def __init__(self, draw_params: Union[ParamServer, dict, None] = None,
                 plot_limits: Union[List[Union[int, float]], None] = None, ax: Union[mpl.axes.Axes, None] = None,
                 figsize: Union[None, Tuple[float, float]] = None, focus_obstacle: Union[None, Obstacle] = None):
        """
        Creates an renderer for matplotlib

        :param draw_params: Default drawing params, if not supplied, default values are used.
        :param plot_limits: plotting limits. If not supplied, using `ax.autoscale()`.
        :param ax: Axis to use. If not supplied, `pyplot.gca()` is used.
        :param figsize: size of the figure
        :param focus_obstacle: if provided, the plot_limits are centered around center of obstacle at time_begin
        """

        self._plot_limits = None
        if draw_params is None:
            self.draw_params = ParamServer()
        elif isinstance(draw_params, dict):
            self.draw_params = ParamServer(params=draw_params)
        else:
            self.draw_params = draw_params

        self.plot_limits = plot_limits
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.f = self.ax.figure

        if figsize is not None:
            self.f.set_size_inches(*figsize)

        # Draw elements
        self.dynamic_artists = []
        self.dynamic_collections = []
        self.static_artists = []
        self.static_collections = []
        self.obstacle_patches = []
        self.traffic_sign_artists = []
        self.traffic_signs = []
        self.traffic_sign_call_stack = tuple()
        self.traffic_sign_draw_params = self.draw_params
        # labels of dynamic elements
        self.dynamic_labels = []

        # current center of focus obstacle
        self.plot_center = None
        self.callbacks = defaultdict(list)

        # [modified]
        self.ignore_obs_id = None

    @property
    def plot_limits(self):
        return self._plot_limits

    @plot_limits.setter
    def plot_limits(self, val: List[Union[float, int, List[Union[float, int]]]]):
        if val is not None and isinstance(val[0], List):
            self._plot_limits = val[0] + val[1]
        elif isinstance(val, List) or val == "auto":
            self._plot_limits = val
        elif val is not None:
            raise ValueError(f"Invalid plot_limit: {val}")

    @property
    def plot_limits_focused(self):
        """
        :returns: plot limits centered around focus_obstacle_id defined in draw_params
        """
        if self._plot_limits is not None and (self._plot_limits == "auto" or self.plot_center is None):
            return self._plot_limits
        elif self.plot_center is not None:
            plot_limits_f = np.array(self.plot_limits, dtype=int)
            plot_limits_f[:2] += int(self.plot_center[0])
            plot_limits_f[2:] += int(self.plot_center[1])
            return plot_limits_f

    def add_callback(self, event, func):
        self.callbacks[event].append(func)

    def draw_list(self, drawable_list: List[IDrawable], draw_params: Union[ParamServer, List[dict], dict, None] = None,
                  call_stack: Optional[Tuple[str, ...]] = tuple()) -> None:
        """
        Simple wrapper to draw a list of drawable objects

        :param drawable_list: Objects to draw
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        if not isinstance(draw_params, list):
            draw_params = [draw_params] * len(drawable_list)
        assert len(draw_params) == len(
            drawable_list), f"Number of drawables has to match number of draw params {len(drawable_list)} vs. " \
                            f"{len(draw_params)}!"
        for elem, params in zip(drawable_list, draw_params):
            elem.draw(self, self._get_draw_params(params), call_stack)

    def _get_draw_params(self, draw_params: Union[ParamServer, dict, None]) -> ParamServer:
        if draw_params is None:
            draw_params = self.draw_params
        elif isinstance(draw_params, dict):
            draw_params = ParamServer(params=draw_params, default=self.draw_params._params)
        return draw_params

    def clear(self, keep_static_artists=False) -> None:
        """
        Clears the internal drawing buffer

        :return: None
        """
        self.plot_center = None
        self.obstacle_patches.clear()
        self.traffic_signs.clear()
        self.traffic_sign_call_stack = tuple()
        self.traffic_sign_draw_params = self.draw_params
        self.dynamic_artists.clear()
        self.dynamic_collections.clear()
        self.traffic_sign_artists.clear()
        self.dynamic_labels.clear()
        if keep_static_artists is False:
            self.static_artists.clear()
            self.static_collections.clear()

    def remove_dynamic(self) -> None:
        """
        Remove the dynamic objects from their current axis

        :return: None
        """
        for art in self.dynamic_artists:
            art.remove()

        # text artists cannot be removed -> set invisble
        for t in self.dynamic_labels:
            t.set_visible(False)
        self.dynamic_labels.clear()

    def render_dynamic(self) -> List[artists.Artist]:
        """
        Only render dynamic objects from buffer

        :return: List of drawn object's artists
        """
        artist_list = []
        self.traffic_sign_artists = draw_traffic_light_signs(self.traffic_signs, self.traffic_sign_draw_params,
                                                             self.traffic_sign_call_stack, self)
        for art in self.dynamic_artists:
            self.ax.add_artist(art)
            artist_list.append(art)
        for art in self.traffic_sign_artists:
            self.ax.add_artist(art)
            artist_list.append(art)
        for col in self.dynamic_collections:
            self.ax.add_collection(col)
            artist_list.append(col)
        for t in self.dynamic_labels:
            self.ax.add_artist(t)

        self.obstacle_patches.sort(key=lambda x: x.zorder)
        patch_col = mpl.collections.PatchCollection(self.obstacle_patches, match_original=True,
                                                    zorder=ZOrders.OBSTACLES)
        self.ax.add_collection(patch_col)
        artist_list.append(patch_col)
        self.dynamic_artists = artist_list
        self._connect_callbacks()
        return artist_list

    def render_static(self) -> List[artists.Artist]:
        """
        Only render static objects from buffer

        :return: List of drawn object's artists
        """
        for col in self.static_collections:
            self.ax.add_collection(col)
        for art in self.static_artists:
            self.ax.add_artist(art)

        self._connect_callbacks()
        return self.static_collections + self.static_artists

    def render(self, show: bool = False, filename: str = None,
                     add_artists: List[artists.Artist] = [],
                     manual_xy_lims: List[Any] = None) -> List[artists.Artist]:
        """
        Render all objects from buffer

        :param show: Show the resulting figure
        :param filename: If provided, saves the figure to the provided file
        :return: List of drawn object's artists
        """
        self.ax.cla()
        artists_list = self.render_static()
        artists_list.extend(self.render_dynamic())

        for art in add_artists:
            self.ax.add_artist(art)
        artists_list.extend(add_artists)

        if manual_xy_lims is None:
            if self.plot_limits is None:
                self.ax.autoscale(True)
            else:
                self.ax.set_xlim(self.plot_limits_focused[:2])
                self.ax.set_ylim(self.plot_limits_focused[2:])
        else:
            # [modify]: add this part, which enable to dynamically change the xy_limits.
            self.ax.set_xlim(manual_xy_lims[0][0], manual_xy_lims[0][1])
            self.ax.set_ylim(manual_xy_lims[1][0], manual_xy_lims[1][1])

        # self.ax.set_aspect('equal') # promise different frame's figure size is equal
        if filename is not None:
            self.f.savefig(filename, bbox_inches='tight')
        if show:
            self.f.show()

        if self.draw_params.by_callstack(param_path="axis_visible", call_stack=()) is False:
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)

        # self.clear(keep_static_artists) # [modify]: annotate this part

        return artists_list

    def _connect_callbacks(self):
        """
        Connects collected callbacks with ax object.
        :return:
        """
        for event, funcs in self.callbacks.items():
            for fun in funcs:
                self.ax.callbacks.connect(event, fun)

        self.ax_updated = False

    def create_video(self, obj_lists: List[IDrawable], file_path: str, delta_time_steps: int = 1, plotting_horizon=0,
                     draw_params: Union[List[dict], dict, ParamServer, None] = None, fig_size: Union[list, None] = None,
                     dt=100, dpi=120) -> None:
        """
        Creates a video of one or multiple CommonRoad objects in mp4, gif,
        or avi format.

        :param obj_lists: list of objects to be plotted.
        :param file_path: filename of generated video (ends on .mp4/.gif/.avi, default mp4, when nothing is specified)
        :param delta_time_steps: plot every delta_time_steps time steps of scenario
        :param plotting_horizon: time steps of prediction plotted in each frame
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param fig_size: size of the video
        :param dt: time step between frames in ms
        :param dpi: resolution of the video
        :return: None
        """
        if not isinstance(draw_params, list):
            draw_params = [draw_params] * len(obj_lists)
        for i, p in enumerate(draw_params):
            draw_params[i] = self._get_draw_params(p)
        time_begin = draw_params[0]['time_begin']
        time_end = draw_params[0]['time_end']
        assert time_begin < time_end, '<video/create_scenario_video> ' \
                                      'time_begin=%i needs to smaller than ' \
                                      'time_end=%i.' % (time_begin, time_end)

        if fig_size is None:
            fig_size = [15, 8]

        self.ax.clear()
        self.f.set_size_inches(*fig_size)
        self.ax.set_aspect('equal')

        def init_frame():
            [p.update({'time_begin': time_begin, 'time_end': time_begin + delta_time_steps}) for p in draw_params]
            self.draw_list(obj_lists, draw_params=draw_params)
            self.render_static()
            artists = self.render_dynamic()
            if self.plot_limits is None:
                self.ax.autoscale()
            elif self.plot_limits == 'auto':
                limits = approximate_bounding_box_dyn_obstacles(obj_lists, time_begin)
                if limits is not None:
                    self.ax.xlim(limits[0][0] - 10, limits[0][1] + 10)
                    self.ax.ylim(limits[1][0] - 10, limits[1][1] + 10)
                else:
                    self.ax.autoscale()
            else:
                self.ax.set_xlim(self.plot_limits_focused[0], self.plot_limits_focused[1])
                self.ax.set_ylim(self.plot_limits_focused[2], self.plot_limits_focused[3])

            if draw_params[0].by_callstack(param_path="axis_visible", call_stack=()) is False:
                self.ax.axes.xaxis.set_visible(False)
                self.ax.axes.yaxis.set_visible(False)
            return artists

        def update(frame=0):
            [p.update({'time_begin': time_begin + delta_time_steps * frame,
                       'time_end': time_begin + min(frame_count, delta_time_steps * frame + plotting_horizon)}) for p in
             draw_params]
            self.remove_dynamic()
            self.clear()
            self.draw_list(obj_lists, draw_params=draw_params)
            artists = self.render_dynamic()
            if self.plot_limits is None:
                self.ax.autoscale()
            elif self.plot_limits == 'auto':
                limits = approximate_bounding_box_dyn_obstacles(obj_lists, time_begin)
                if limits is not None:
                    self.ax.xlim(limits[0][0] - 10, limits[0][1] + 10)
                    self.ax.ylim(limits[1][0] - 10, limits[1][1] + 10)
                else:
                    self.ax.autoscale()
            else:
                self.ax.set_xlim(self.plot_limits_focused[0], self.plot_limits_focused[1])
                self.ax.set_ylim(self.plot_limits_focused[2], self.plot_limits_focused[3])
            return artists

        # Min frame rate is 1 fps
        dt = min(1000.0, dt)
        frame_count = (time_end - time_begin) // delta_time_steps
        plt.ioff()
        # Interval determines the duration of each frame in ms
        anim = FuncAnimation(self.f, update, frames=frame_count, init_func=init_frame, blit=False, interval=dt)

        if not any([file_path.endswith('.mp4'), file_path.endswith('.gif'), file_path.endswith('.avi')]):
            file_path += '.mp4'
        fps = int(math.ceil(1000.0 / dt))
        interval_seconds = dt / 1000.0
        anim.save(file_path, dpi=dpi, writer='ffmpeg', fps=fps,
                  extra_args=["-g", "1", "-keyint_min", str(interval_seconds)])
        self.clear()
        self.ax.clear()

    def add_legend(self, legend: Dict[Tuple[str, ...], str],
                   draw_params: Union[ParamServer, dict, None] = None) -> None:
        """
        Adds legend with color of objects specified by legend.keys() and
        texts specified by legend.values().

        :param legend: color of objects specified by path in legend.keys() and texts specified by legend.values()
        :param draw_params: draw parameters used for plotting (color is extracted using path in legend.keys())
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        handles = []
        for obj_name, text in legend.items():
            try:
                color = draw_params[obj_name]
            except KeyError:
                color = None
            if color is not None:
                handles.append(mpl.patches.Patch(color=color, label=text))

        legend = self.ax.legend(handles=handles)
        legend.set_zorder(ZOrders.LABELS)

    def set_ignore_obs_id(self, obs_id=None):
        self.ignore_obs_id = obs_id

    def draw_scenario(self, obj: Scenario, draw_params: Union[ParamServer, dict, None],
                      call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['scenario'])
        obj.lanelet_network.draw(self, draw_params, call_stack)

        # draw only obstacles inside plot limits
        focus_obstacle_id = draw_params.by_callstack(call_stack, ('focus_obstacle_id',))
        if focus_obstacle_id is False and type(self.plot_limits) == list:
            time_begin = draw_params.by_callstack(call_stack, ('time_begin',))
            # dynamic obstacles
            obs = obj.obstacles_by_position_intervals([Interval(self.plot_limits[0], self.plot_limits[1]),
                                                       Interval(self.plot_limits[2], self.plot_limits[3])],
                                                      tuple(ObstacleRole), time_begin)
        else:
            obs = obj.obstacles
        # Draw all objects
        for o in obs:
            if (o.obstacle_id != self.ignore_obs_id):
                o.draw(self, draw_params, call_stack)

    def draw_static_obstacle(self, obj: StaticObstacle, draw_params: Union[ParamServer, dict, None],
                             call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        time_begin = draw_params.by_callstack(tuple(), ('time_begin',))
        call_stack = tuple(list(call_stack) + ['static_obstacle'])
        occ = obj.occupancy_at_time(time_begin)
        self._draw_occupancy(occ, obj.initial_state, draw_params, call_stack)

    def _draw_occupancy(self, occ: Occupancy, state: State, draw_params: Union[ParamServer, dict, None],
                        call_stack: Tuple[str, ...]) -> None:
        if occ is not None:
            occ.draw(self, draw_params, call_stack)
        if state is not None and state.is_uncertain_position:
            zorder_polygon = draw_params.by_callstack(call_stack, ("occupancy", "shape", "polygon", "zorder")) + 0.1
            zorder_rectangle = draw_params.by_callstack(call_stack, ("occupancy", "shape", "rectancle", "zorder")) + 0.1
            zorder_circle = draw_params.by_callstack(call_stack, ("occupancy", "shape", "circle", "zorder")) + 0.1
            draw_params = {"occupancy": {"uncertain_position": {
                "shape": {"polygon": {"zorder": zorder_polygon}, "rectangle": {"zorder": zorder_rectangle},
                          "circle": {"zorder": zorder_circle}}}}}
            state.position.draw(self, draw_params, call_stack + ('occupancy', 'uncertain_position'))

    def draw_dynamic_obstacle(self, obj: DynamicObstacle, draw_params: Union[ParamServer, dict, None],
                              call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)

        time_begin = draw_params.by_callstack(call_stack, ('time_begin',))
        time_end = draw_params.by_callstack(call_stack, ('time_end',))
        focus_obstacle_id = draw_params.by_callstack(call_stack, ('focus_obstacle_id',))
        call_stack = tuple(list(call_stack) + ['dynamic_obstacle'])
        draw_icon = draw_params.by_callstack(call_stack, 'draw_icon')
        show_label = draw_params.by_callstack(call_stack, 'show_label')
        draw_shape = draw_params.by_callstack(call_stack, 'draw_shape')
        draw_direction = draw_params.by_callstack(call_stack, 'draw_direction')
        draw_initial_state = draw_params.by_callstack(call_stack, 'draw_initial_state')
        draw_occupancies = draw_params.by_callstack(call_stack, ('occupancy', 'draw_occupancies'))
        draw_signals = draw_params.by_callstack(call_stack, 'draw_signals')
        draw_trajectory = draw_params.by_callstack(call_stack, ('trajectory', 'draw_trajectory'))

        draw_history = draw_params.by_callstack(call_stack, ('history', 'draw_history'))

        if obj.prediction is None \
                and obj.initial_state.time_step < time_begin or obj.initial_state.time_step > time_end:
            return
        elif (obj.prediction is not None and obj.prediction.final_time_step < time_begin) \
                or obj.initial_state.time_step > time_end:
            return

        if draw_history and isinstance(obj.prediction, commonroad.prediction.prediction.TrajectoryPrediction):
            self._draw_history(obj, call_stack, draw_params)

        # draw car icon
        if draw_icon and obj.obstacle_type in supported_icons() and type(
                obj.prediction) == commonroad.prediction.prediction.TrajectoryPrediction:

            try:
                length = obj.obstacle_shape.length
                width = obj.obstacle_shape.width
            except AttributeError:
                draw_shape = True
                draw_icon = False

            if draw_icon:
                draw_shape = False
                if time_begin == obj.initial_state.time_step:
                    inital_state = obj.initial_state
                else:
                    inital_state = obj.prediction.trajectory.state_at_time_step(time_begin)
                if inital_state is not None:
                    if isinstance(obj.obstacle_shape, (Rectangle, Circle, Polygon)):
                        shape_name = type(obj.obstacle_shape).__name__.lower()
                    else:
                        shape_name = "rectangle"
                    call_stack_tmp = call_stack + ('vehicle_shape', 'occupancy', 'shape', shape_name)

                    facecolor = draw_params.by_callstack(call_stack_tmp, 'facecolor')
                    edgecolor = draw_params.by_callstack(call_stack_tmp, 'edgecolor')
                    self.obstacle_patches.extend(get_obstacle_icon_patch(obj.obstacle_type, inital_state.position[0],
                                                                         inital_state.position[1],
                                                                         inital_state.orientation,
                                                                         vehicle_length=length, vehicle_width=width,
                                                                         vehicle_color=facecolor, edgecolor=edgecolor,
                                                                         zorder=ZOrders.CAR_PATCH))
        elif draw_icon is True:
            draw_shape = True

        # draw shape
        if draw_shape:
            veh_occ = obj.occupancy_at_time(time_begin)
            if veh_occ is not None:
                self._draw_occupancy(veh_occ, obj.initial_state, draw_params, call_stack + ('vehicle_shape',))
                if draw_direction and veh_occ is not None and type(veh_occ.shape) == Rectangle:
                    v_tri = get_vehicle_direction_triangle(veh_occ.shape)
                    self.draw_polygon(v_tri, draw_params, call_stack + ('vehicle_shape', 'direction'))

        # draw signals
        if draw_signals and (draw_shape or draw_icon):
            sig = obj.signal_state_at_time_step(time_begin)
            veh_occ = obj.occupancy_at_time(time_begin)
            if veh_occ is not None and sig is not None:
                self._draw_signal_state(sig, veh_occ, draw_params, call_stack)

        # draw occupancies
        if draw_occupancies or type(obj.prediction) == commonroad.prediction.prediction.SetBasedPrediction:
            if draw_shape:
                # occupancy already plotted
                time_begin_occ = time_begin + 1
            else:
                time_begin_occ = time_begin

            for time_step in range(time_begin_occ, time_end):
                state = None
                if isinstance(obj.prediction, TrajectoryPrediction):
                    state = obj.prediction.trajectory.state_at_time_step(time_step)
                occ = obj.occupancy_at_time(time_step)
                self._draw_occupancy(occ, state, draw_params, call_stack)

        # draw trajectory
        if draw_trajectory and type(obj.prediction) == commonroad.prediction.prediction.TrajectoryPrediction:
            obj.prediction.trajectory.draw(self, draw_params, call_stack)

        # get state
        state = None
        if time_begin == 0:
            state = obj.initial_state
        elif type(obj.prediction) == commonroad.prediction.prediction.TrajectoryPrediction:
            state = obj.prediction.trajectory.state_at_time_step(time_begin)

        # set plot center state
        if focus_obstacle_id == obj.obstacle_id and state is not None:
            self.plot_center = state.position

        # draw label
        if show_label:
            if state is not None:
                position = state.position
                self.dynamic_labels.append(text.Text(position[0] + 0.5, position[1], str(obj.obstacle_id), clip_on=True,
                                                     zorder=ZOrders.LABELS))

        # draw initial state
        if draw_initial_state and state is not None:
            state.draw(self, draw_params, call_stack)

    def draw_phantom_obstacle(self, obj: PhantomObstacle, draw_params: Union[ParamServer, dict, None],
                              call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)

        time_begin = draw_params.by_callstack(call_stack, ('time_begin',))
        time_end = draw_params.by_callstack(call_stack, ('time_end',))

        call_stack = call_stack + ('phantom_obstacle',)
        draw_shape = draw_params.by_callstack(call_stack, 'draw_shape')
        draw_occupancies = draw_params.by_callstack(call_stack, ('occupancy', 'draw_occupancies'))

        # draw shape
        if draw_shape:
            occ = obj.occupancy_at_time(time_begin)
            if occ is not None:
                occ.draw(self, draw_params, call_stack + ('vehicle_shape',))

        # draw occupancies
        if draw_occupancies == 1:
            if draw_shape:
                # Initial shape already drawn
                occ_time_begin = time_begin + 1
            else:
                occ_time_begin = time_begin
            for time_step in range(occ_time_begin, time_end):
                occ = obj.occupancy_at_time(time_step)
                if occ is not None:
                    occ.draw(self, draw_params, call_stack)

    def draw_environment_obstacle(self, obj: EnvironmentObstacle, draw_params: Union[ParamServer, dict, None],
                                  call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        time_begin = draw_params.by_callstack(tuple(), ('time_begin',))
        call_stack = call_stack + ('environment_obstacle',)
        obj.occupancy_at_time(time_begin).draw(self, draw_params, call_stack)

    def _draw_history(self, dyn_obs: DynamicObstacle, call_stack: Tuple[str, ...], draw_params: ParamServer):
        """
        Draws history occupancies of the dynamic obstacle

        :param call_stack: tuple of string containing the call stack,
        which allows for differentiation of plotting styles
               depending on the call stack of drawing functions
        :param draw_params: parameters for plotting given by a nested dict
        that recreates the structure of an object or a ParamServer object
        :param dyn_obs: the dynamic obstacle
        :return:
        """
        time_begin = draw_params['time_begin']
        history_base_color = draw_params.by_callstack(call_stack,
                                                      ('vehicle_shape', 'occupancy', 'shape', 'rectangle', 'facecolor'))
        history_callstack = call_stack + ('history',)
        history_steps = draw_params.by_callstack(history_callstack, 'steps')
        history_fade_factor = draw_params.by_callstack(history_callstack, 'fade_factor')
        history_step_size = draw_params.by_callstack(history_callstack, 'step_size')
        history_base_color = rgb_to_hsv(to_rgb(history_base_color))
        for history_idx in range(history_steps, 0, -1):
            time_step = time_begin - history_idx * history_step_size
            occ = dyn_obs.occupancy_at_time(time_step)
            if occ is not None:
                color_hsv_new = history_base_color.copy()
                color_hsv_new[2] = max(0, color_hsv_new[2] - history_fade_factor * history_idx)
                color_hex_new = to_hex(hsv_to_rgb(color_hsv_new))
                draw_params[history_callstack + ('occupancy', 'shape', 'rectangle', 'facecolor')] = color_hex_new
                draw_params[history_callstack + ('occupancy', 'shape', 'circle', 'facecolor')] = color_hex_new
                draw_params[history_callstack + ('occupancy', 'shape', 'polygon', 'facecolor')] = color_hex_new
                occ.draw(self, draw_params, history_callstack)

    def draw_trajectory(self, obj: Trajectory, draw_params: Union[ParamServer, dict, None],
                        call_stack: Tuple[str, ...]) -> None:
        """
        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)

        time_begin = draw_params.by_callstack(call_stack, 'time_begin')
        time_end = draw_params.by_callstack(call_stack, 'time_end')

        call_stack = call_stack + ('trajectory',)
        line_color = draw_params.by_callstack(call_stack, 'facecolor')

        line_width = draw_params.by_callstack(call_stack, 'line_width')
        draw_continuous = draw_params.by_callstack(call_stack, 'draw_continuous')
        z_order = draw_params.by_callstack(call_stack, 'z_order')

        if time_begin == time_end:
            return

        traj_states = [obj.state_at_time_step(t) for t in range(time_begin, time_end) if
                       obj.state_at_time_step(t) is not None]
        position_sets = [s.position for s in traj_states if s.is_uncertain_position]
        traj_points = [s.position for s in traj_states if not s.is_uncertain_position]

        traj_points = np.array(traj_points)

        # Draw certain states
        if draw_continuous:
            path = mpl.path.Path(traj_points, closed=False)
            self.obstacle_patches.append(
                    mpl.patches.PathPatch(path, color=line_color, lw=line_width, zorder=z_order, fill=False))
        else:
            self.dynamic_collections.append(
                    collections.EllipseCollection(np.ones([traj_points.shape[0], 1]) * line_width,
                                                  np.ones([traj_points.shape[0], 1]) * line_width,
                                                  np.zeros([traj_points.shape[0], 1]), offsets=traj_points, units='xy',
                                                  linewidths=0, zorder=z_order, transOffset=self.ax.transData,
                                                  facecolor=line_color))

        # Draw uncertain states
        for p in position_sets:
            p.draw(self, draw_params, call_stack)

    def draw_trajectories(self, obj: List[Trajectory], draw_params: Union[ParamServer, dict, None],
                          call_stack: Tuple[str, ...]) -> None:
        draw_params = self._get_draw_params(draw_params)
        unique_colors = draw_params.by_callstack(call_stack, ('trajectory', 'unique_colors'))
        if unique_colors:
            cmap = colormap_idx(len(obj))
            for i, traj in enumerate(obj):
                draw_params['trajectory', 'facecolor'] = mpl.colors.to_hex(cmap(i))
                traj.draw(self, draw_params, call_stack)
        else:
            self.draw_list(obj, draw_params)

    def draw_polygon(self, vertices, draw_params: Union[ParamServer, dict, None], call_stack: Tuple[str, ...]) -> None:
        """
        Draws a polygon shape

        :param vertices: vertices of the polygon
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['shape', 'polygon'])
        facecolor = draw_params.by_callstack(call_stack, 'facecolor')
        edgecolor = draw_params.by_callstack(call_stack, 'edgecolor')
        zorder = draw_params.by_callstack(call_stack, 'zorder')
        opacity = draw_params.by_callstack(call_stack, 'opacity')
        linewidth = draw_params.by_callstack(call_stack, 'linewidth')
        antialiased = draw_params.by_callstack(call_stack, 'antialiased')
        self.obstacle_patches.append(
                mpl.patches.Polygon(vertices, closed=True, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder,
                                    alpha=opacity, linewidth=linewidth, antialiased=antialiased))

    def draw_rectangle(self, vertices: np.ndarray, draw_params: Union[ParamServer, dict, None],
                       call_stack: Tuple[str, ...]) -> None:
        """
        Draws a rectangle shape

        :param vertices: vertices of the rectangle
        :param draw_params: parameters for plotting given by a nested dict that
            recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['shape', 'rectangle'])

        facecolor = draw_params.by_callstack(call_stack, 'facecolor')
        edgecolor = draw_params.by_callstack(call_stack, 'edgecolor')
        zorder = draw_params.by_callstack(call_stack, 'zorder')
        opacity = draw_params.by_callstack(call_stack, 'opacity')
        linewidth = draw_params.by_callstack(call_stack, 'linewidth')
        antialiased = draw_params.by_callstack(call_stack, 'antialiased')

        self.obstacle_patches.append(
                mpl.patches.Polygon(vertices, closed=True, zorder=zorder, facecolor=facecolor, edgecolor=edgecolor,
                                    alpha=opacity, antialiased=antialiased, linewidth=linewidth))

    def draw_ellipse(self, center: List[float], radius_x: float, radius_y: float,
                     draw_params: Union[ParamServer, dict, None], call_stack: Tuple[str, ...]) -> None:
        """
        Draws a circle shape

        :param ellipse: center position of the ellipse
        :param radius_x: radius of the ellipse along the x-axis
        :param radius_y: radius of the ellipse along the y-axis
        :param draw_params: parameters for plotting given by a nested dict that
            recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['shape', 'circle'])
        facecolor = draw_params.by_callstack(call_stack, 'facecolor')
        edgecolor = draw_params.by_callstack(call_stack, 'edgecolor')
        zorder = draw_params.by_callstack(call_stack, 'zorder')
        opacity = draw_params.by_callstack(call_stack, 'opacity')
        linewidth = draw_params.by_callstack(call_stack, 'linewidth')

        self.obstacle_patches.append(
                mpl.patches.Ellipse(center, 2 * radius_x, 2 * radius_y, facecolor=facecolor, edgecolor=edgecolor,
                                    zorder=zorder, linewidth=linewidth, alpha=opacity))

    def draw_state(self, state: State, draw_params: Union[ParamServer, dict, None],
                   call_stack: Tuple[str, ...] = None) -> None:
        """
        Draws a state as an arrow of its velocity vector

        :param state: state to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack += 'state',
        scale_factor = draw_params.by_callstack(call_stack, 'scale_factor')
        arrow_args = draw_params.by_callstack(call_stack, 'kwargs')
        draw_arrow = draw_params.by_callstack(call_stack, 'draw_arrow')
        radius = draw_params.by_callstack(call_stack, 'radius')
        facecolor = draw_params.by_callstack(call_stack, 'facecolor')
        zorder = draw_params.by_callstack(call_stack, 'zorder')
        if zorder is None:
            zorder = ZOrders.STATE
        self.obstacle_patches.append(
                mpl.patches.Circle(state.position, radius=radius, zorder=zorder, color=facecolor))

        if draw_arrow:
            cos = math.cos(state.orientation)
            sin = math.sin(state.orientation)
            x = state.position[0]
            y = state.position[1]
            self.obstacle_patches.append(mpl.patches.FancyArrow(x=x, y=y, dx=state.velocity * cos * scale_factor,
                                                                dy=state.velocity * sin * scale_factor,
                                                                zorder=zorder, **arrow_args))

    def draw_lanelet_network(self, obj: LaneletNetwork, draw_params: Union[ParamServer, dict, None],
                             call_stack: Tuple[str, ...]) -> None:
        """
        Draws a lanelet network

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict that
            recreates the structure of an object,
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['lanelet_network'])

        traffic_lights = obj.traffic_lights
        traffic_signs = obj.traffic_signs
        intersections = obj.intersections
        lanelets = obj.lanelets

        time_begin = draw_params.by_callstack(call_stack, ('time_begin',))
        if traffic_lights is not None:
            draw_traffic_lights = draw_params.by_callstack(call_stack, ('traffic_light', 'draw_traffic_lights'))

            traffic_light_colors = draw_params.by_callstack(call_stack, ('traffic_light'))
        else:
            draw_traffic_lights = False

        if traffic_signs is not None:
            draw_traffic_signs = draw_params.by_callstack(call_stack, ('traffic_sign', 'draw_traffic_signs'))
            show_traffic_sign_label = draw_params.by_callstack(call_stack, ('traffic_sign', 'show_label'))
        else:
            draw_traffic_signs = show_traffic_sign_label = False

        if intersections is not None and len(intersections) > 0:
            draw_intersections = draw_params.by_callstack(call_stack, ('intersection', 'draw_intersections'))
        else:
            draw_intersections = False

        if draw_intersections is True:
            draw_incoming_lanelets = draw_params.by_callstack(call_stack, ('intersection', 'draw_incoming_lanelets'))
            incoming_lanelets_color = draw_params.by_callstack(call_stack, ('intersection', 'incoming_lanelets_color'))
            draw_crossings = draw_params.by_callstack(call_stack, ('intersection', 'draw_crossings'))
            crossings_color = draw_params.by_callstack(call_stack, ('intersection', 'crossings_color'))
            draw_successors = draw_params.by_callstack(call_stack, ('intersection', 'draw_successors'))
            successors_left_color = draw_params.by_callstack(call_stack, ('intersection', 'successors_left_color'))
            successors_straight_color = draw_params.by_callstack(call_stack,
                                                                 ('intersection', 'successors_straight_color'))
            successors_right_color = draw_params.by_callstack(call_stack, ('intersection', 'successors_right_color'))
            show_intersection_labels = draw_params.by_callstack(call_stack, ('intersection', 'show_label'))
        else:
            draw_incoming_lanelets = draw_crossings = draw_successors = show_intersection_labels = False

        left_bound_color = draw_params.by_callstack(call_stack, ('lanelet', 'left_bound_color'))
        right_bound_color = draw_params.by_callstack(call_stack, ('lanelet', 'right_bound_color'))
        center_bound_color = draw_params.by_callstack(call_stack, ('lanelet', 'center_bound_color'))
        unique_colors = draw_params.by_callstack(call_stack, ('lanelet', 'unique_colors'))
        draw_stop_line = draw_params.by_callstack(call_stack, ('lanelet', 'draw_stop_line'))
        stop_line_color = draw_params.by_callstack(call_stack, ('lanelet', 'stop_line_color'))
        draw_line_markings = draw_params.by_callstack(call_stack, ('lanelet', 'draw_line_markings'))
        show_label = draw_params.by_callstack(call_stack, ('lanelet', 'show_label'))
        draw_border_vertices = draw_params.by_callstack(call_stack, ('lanelet', 'draw_border_vertices'))
        draw_left_bound = draw_params.by_callstack(call_stack, ('lanelet', 'draw_left_bound'))
        draw_right_bound = draw_params.by_callstack(call_stack, ('lanelet', 'draw_right_bound'))
        draw_center_bound = draw_params.by_callstack(call_stack, ('lanelet', 'draw_center_bound'))
        draw_start_and_direction = draw_params.by_callstack(call_stack, ('lanelet', 'draw_start_and_direction'))
        draw_linewidth = draw_params.by_callstack(call_stack, ('lanelet', 'draw_linewidth'))
        fill_lanelet = draw_params.by_callstack(call_stack, ('lanelet', 'fill_lanelet'))
        facecolor = draw_params.by_callstack(call_stack, ('lanelet', 'facecolor'))
        antialiased = draw_params.by_callstack(call_stack, 'antialiased')

        draw_lanlet_ids = draw_params.by_callstack(call_stack, 'draw_ids')

        colormap_tangent = draw_params.by_callstack(call_stack, ('lanelet', 'colormap_tangent'))

        # Collect lanelets
        incoming_lanelets = set()
        incomings_left = {}
        incomings_id = {}
        crossings = set()
        all_successors = set()
        successors_left = set()
        successors_straight = set()
        successors_right = set()
        if draw_intersections:
            # collect incoming lanelets
            if draw_incoming_lanelets:
                incomings: List[set] = []
                inc_2_intersections = obj.map_inc_lanelets_to_intersections
                for intersection in intersections:
                    for incoming in intersection.incomings:
                        incomings.append(incoming.incoming_lanelets)
                        for l_id in incoming.incoming_lanelets:
                            incomings_left[l_id] = incoming.left_of
                            incomings_id[l_id] = incoming.incoming_id
                incoming_lanelets: Set[int] = set.union(*incomings)

            if draw_crossings:
                tmp_list: List[set] = [intersection.crossings for intersection in intersections]
                crossings: Set[int] = set.union(*tmp_list)

            if draw_successors:
                tmp_list: List[set] = [incoming.successors_left for intersection in intersections for incoming in
                                       intersection.incomings]
                successors_left: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [incoming.successors_straight for intersection in intersections for incoming in
                                       intersection.incomings]
                successors_straight: Set[int] = set.union(*tmp_list)
                tmp_list: List[set] = [incoming.successors_right for intersection in intersections for incoming in
                                       intersection.incomings]
                successors_right: Set[int] = set.union(*tmp_list)
                all_successors = set.union(successors_straight, successors_right, successors_left)

        # select unique colors from colormap for each lanelet's center_line

        incoming_vertices_fill = list()
        crossing_vertices_fill = list()
        succ_left_paths = list()
        succ_straight_paths = list()
        succ_right_paths = list()

        vertices_fill = list()
        coordinates_left_border_vertices = np.empty((0, 2))
        coordinates_right_border_vertices = np.empty((0, 2))
        direction_list = list()
        center_paths = list()
        left_paths = list()
        right_paths = list()

        if draw_traffic_lights:
            center_line_color_dict = collect_center_line_colors(obj, traffic_lights, time_begin)

        cmap_lanelet = colormap_idx(len(lanelets))

        # collect paths for drawing
        for i_lanelet, lanelet in enumerate(lanelets):
            if isinstance(draw_lanlet_ids, list) and lanelet.lanelet_id not in draw_lanlet_ids:
                continue

            # left bound
            if draw_border_vertices or draw_left_bound:
                if draw_border_vertices:
                    coordinates_left_border_vertices = np.vstack(
                            (coordinates_left_border_vertices, lanelet.left_vertices))

                if draw_line_markings and lanelet.line_marking_left_vertices is not LineMarking.UNKNOWN and \
                        lanelet.line_marking_left_vertices is not LineMarking.NO_MARKING:
                    linestyle, dashes, linewidth_metres = line_marking_to_linestyle(lanelet.line_marking_left_vertices)
                    if lanelet.distance[-1] <= linewidth_metres:
                        left_paths.append(Path(lanelet.left_vertices, closed=False))
                    else:
                        tmp_left = lanelet.left_vertices.copy()
                        tmp_left[0, :] = lanelet.interpolate_position(linewidth_metres / 2)[2]
                        tmp_left[-1, :] = lanelet.interpolate_position(lanelet.distance[-1] - linewidth_metres / 2)[2]
                        line = LineDataUnits(tmp_left[:, 0], tmp_left[:, 1], zorder=ZOrders.LEFT_BOUND,
                                             linewidth=linewidth_metres, alpha=1.0, color=left_bound_color,
                                             linestyle=linestyle, dashes=dashes)
                        self.static_artists.append(line)
                else:
                    left_paths.append(Path(lanelet.left_vertices, closed=False))

            # right bound
            if draw_border_vertices or draw_right_bound:
                if draw_border_vertices:
                    coordinates_right_border_vertices = np.vstack(
                            (coordinates_right_border_vertices, lanelet.right_vertices))

                if draw_line_markings and lanelet.line_marking_right_vertices is not LineMarking.UNKNOWN and \
                        lanelet.line_marking_right_vertices is not LineMarking.NO_MARKING:
                    linestyle, dashes, linewidth_metres = line_marking_to_linestyle(lanelet.line_marking_right_vertices)
                    if lanelet.distance[-1] <= linewidth_metres:
                        right_paths.append(Path(lanelet.right_vertices, closed=False))
                    else:
                        tmp_right = lanelet.right_vertices.copy()
                        tmp_right[0, :] = lanelet.interpolate_position(linewidth_metres / 2)[1]
                        tmp_right[-1, :] = lanelet.interpolate_position(lanelet.distance[-1] - linewidth_metres / 2)[1]
                        line = LineDataUnits(tmp_right[:, 0], tmp_right[:, 1], zorder=ZOrders.RIGHT_BOUND,
                                             linewidth=linewidth_metres, alpha=1.0, color=right_bound_color,
                                             linestyle=linestyle, dashes=dashes)
                        self.static_artists.append(line)
                else:
                    right_paths.append(Path(lanelet.right_vertices, closed=False))

            # stop line
            if draw_stop_line and lanelet.stop_line:
                stop_line = np.vstack([lanelet.stop_line.start, lanelet.stop_line.end])
                linestyle, dashes, linewidth_metres = line_marking_to_linestyle(lanelet.stop_line.line_marking)
                # cut off in the beginning, because linewidth_metres is added
                # later
                vec = stop_line[1, :] - stop_line[0, :]
                tangent = vec / np.linalg.norm(vec)
                stop_line[0, :] += linewidth_metres * tangent / 2
                stop_line[1, :] -= linewidth_metres * tangent / 2
                line = LineDataUnits(stop_line[:, 0], stop_line[:, 1], zorder=ZOrders.STOP_LINE,
                                     linewidth=linewidth_metres, alpha=1.0, color=stop_line_color, linestyle=linestyle,
                                     dashes=dashes)
                self.static_artists.append(line)

            if unique_colors:
                # set center bound color to unique value
                center_bound_color = cmap_lanelet(i_lanelet)

            # direction arrow
            if draw_start_and_direction:
                center = lanelet.center_vertices[0]
                tan_vec = np.array(lanelet.right_vertices[0]) - np.array(lanelet.left_vertices[0])
                path = get_arrow_path_at(center[0], center[1], math.atan2(tan_vec[1], tan_vec[0]) + 0.5 * np.pi)
                if unique_colors:
                    direction_list.append(mpl.patches.PathPatch(path, color=center_bound_color, lw=0.5,
                                                                zorder=ZOrders.DIRECTION_ARROW,
                                                                antialiased=antialiased))
                else:
                    direction_list.append(path)

            # visualize traffic light state through colored center bound
            has_traffic_light = draw_traffic_lights and lanelet.lanelet_id in center_line_color_dict
            if has_traffic_light:
                light_state = center_line_color_dict[lanelet.lanelet_id]

                if light_state is not TrafficLightState.INACTIVE:
                    linewidth_metres = 0.75
                    # dashed line for red_yellow
                    linestyle = '--' if light_state == TrafficLightState.RED_YELLOW else '-'
                    dashes = (5, 5) if linestyle == '--' else (None, None)

                    # cut off in the beginning, because linewidth_metres is
                    # added
                    # later
                    tmp_center = lanelet.center_vertices.copy()
                    if lanelet.distance[-1] > linewidth_metres:
                        tmp_center[0, :] = lanelet.interpolate_position(linewidth_metres)[0]
                    zorder = ZOrders.LIGHT_STATE_GREEN if light_state == TrafficLightState.GREEN else \
                        ZOrders.LIGHT_STATE_OTHER
                    line = LineDataUnits(tmp_center[:, 0], tmp_center[:, 1], zorder=zorder, linewidth=linewidth_metres,
                                         alpha=0.7, color=traffic_light_color_dict(light_state, traffic_light_colors),
                                         linestyle=linestyle, dashes=dashes)
                    self.dynamic_artists.append(line)

            # draw colored center bound. Hierarchy or colors: successors > usual
            # center bound
            is_successor = draw_intersections and draw_successors and lanelet.lanelet_id in all_successors
            if is_successor:
                if lanelet.lanelet_id in successors_left:
                    succ_left_paths.append(Path(lanelet.center_vertices, closed=False))
                elif lanelet.lanelet_id in successors_straight:
                    succ_straight_paths.append(Path(lanelet.center_vertices, closed=False))
                else:
                    succ_right_paths.append(Path(lanelet.center_vertices, closed=False))

            elif draw_center_bound:
                if unique_colors:
                    center_paths.append(mpl.patches.PathPatch(Path(lanelet.center_vertices, closed=False),
                                                              edgecolor=center_bound_color, facecolor='none',
                                                              lw=draw_linewidth, zorder=ZOrders.CENTER_BOUND,
                                                              antialiased=antialiased))
                elif colormap_tangent:
                    relative_angle = draw_params['relative_angle']
                    points = lanelet.center_vertices.reshape(-1, 1, 2)
                    angles = get_tangent_angle(points[:, 0, :], relative_angle)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    norm = plt.Normalize(0, 360)
                    lc = collections.LineCollection(segments, cmap='hsv', norm=norm, lw=draw_linewidth,
                                                    zorder=ZOrders.CENTER_BOUND, antialiased=antialiased)
                    lc.set_array(angles)
                    self.static_collections.append(lc)

            is_incoming_lanelet = draw_intersections and draw_incoming_lanelets and (
                    lanelet.lanelet_id in incoming_lanelets)
            is_crossing = draw_intersections and draw_crossings and (lanelet.lanelet_id in crossings)

            # Draw lanelet area
            if fill_lanelet:
                if not is_incoming_lanelet and not is_crossing:
                    vertices_fill.append(np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0))))

            # collect incoming lanelets in separate list for plotting in
            # different color
            if is_incoming_lanelet:
                incoming_vertices_fill.append(
                        np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0))))
            elif is_crossing:
                crossing_vertices_fill.append(
                        np.concatenate((lanelet.right_vertices, np.flip(lanelet.left_vertices, 0))))

            # Draw labels
            if show_label or show_intersection_labels or draw_traffic_signs:
                strings = []
                if show_label:
                    strings.append(str(lanelet.lanelet_id))
                if is_incoming_lanelet and show_intersection_labels:
                    strings.append(f'int_id: {inc_2_intersections[lanelet.lanelet_id].intersection_id}')
                    strings.append('inc_id: ' + str(incomings_id[lanelet.lanelet_id]))
                    strings.append('inc_left: ' + str(incomings_left[lanelet.lanelet_id]))
                if draw_traffic_signs and show_traffic_sign_label:
                    traffic_signs_tmp = [obj._traffic_signs[id] for id in lanelet.traffic_signs]
                    if traffic_signs_tmp:
                        # add as text to label
                        str_tmp = 'sign: '
                        add_str = ''
                        for sign in traffic_signs_tmp:
                            for el in sign.traffic_sign_elements:
                                # TrafficSignIDGermany(
                                # el.traffic_sign_element_id).name would give
                                # the
                                # name
                                str_tmp += add_str + el.traffic_sign_element_id.value
                                add_str = ', '

                        strings.append(str_tmp)

                label_string = ', '.join(strings)
                if len(label_string) > 0:
                    # compute normal angle of label box
                    clr_positions = lanelet.interpolate_position(0.5 * lanelet.distance[-1])
                    normal_vector = np.array(clr_positions[1]) - np.array(clr_positions[2])
                    angle = np.rad2deg(math.atan2(normal_vector[1], normal_vector[0])) - 90
                    angle = angle if Interval(-90, 90).contains(angle) else angle - 180

                    self.static_artists.append(text.Text(clr_positions[0][0], clr_positions[0][1], label_string,
                                                         bbox={'facecolor': center_bound_color, 'pad': 2},
                                                         horizontalalignment='center', verticalalignment='center',
                                                         rotation=angle, zorder=ZOrders.LANELET_LABEL))

        # draw paths and collect axis handles
        if draw_right_bound:
            self.static_collections.append(
                    collections.PathCollection(right_paths, edgecolor=right_bound_color, facecolor='none',
                                               lw=draw_linewidth, zorder=ZOrders.RIGHT_BOUND, antialiased=antialiased))
        if draw_left_bound:
            self.static_collections.append(
                    collections.PathCollection(left_paths, edgecolor=left_bound_color, facecolor='none',
                                               lw=draw_linewidth, zorder=ZOrders.LEFT_BOUND, antialiased=antialiased))
        if unique_colors:
            if draw_center_bound:
                if draw_center_bound:
                    self.static_collections.append(
                            collections.PatchCollection(center_paths, match_original=True, zorder=ZOrders.CENTER_BOUND,
                                                        antialiased=antialiased))
                if draw_start_and_direction:
                    self.static_collections.append(collections.PatchCollection(direction_list, match_original=True,
                                                                               zorder=ZOrders.DIRECTION_ARROW,
                                                                               antialiased=antialiased))

        elif not colormap_tangent:
            if draw_center_bound:
                self.static_collections.append(
                        collections.PathCollection(center_paths, edgecolor=center_bound_color, facecolor='none',
                                                   lw=draw_linewidth, zorder=ZOrders.CENTER_BOUND,
                                                   antialiased=antialiased))
            if draw_start_and_direction:
                self.static_collections.append(
                        collections.PathCollection(direction_list, color=center_bound_color, lw=0.5,
                                                   zorder=ZOrders.DIRECTION_ARROW, antialiased=antialiased))

        if successors_left:
            self.static_collections.append(
                    collections.PathCollection(succ_left_paths, edgecolor=successors_left_color, facecolor='none',
                                               lw=draw_linewidth * 3.0, zorder=ZOrders.SUCCESSORS,
                                               antialiased=antialiased))
        if successors_straight:
            self.static_collections.append(
                    collections.PathCollection(succ_straight_paths, edgecolor=successors_straight_color,
                                               facecolor='none', lw=draw_linewidth * 3.0, zorder=ZOrders.SUCCESSORS,
                                               antialiased=antialiased))
        if successors_right:
            self.static_collections.append(
                    collections.PathCollection(succ_right_paths, edgecolor=successors_right_color, facecolor='none',
                                               lw=draw_linewidth * 3.0, zorder=ZOrders.SUCCESSORS,
                                               antialiased=antialiased))

        # fill lanelets with facecolor
        self.static_collections.append(
                collections.PolyCollection(vertices_fill, transOffset=self.ax.transData, zorder=ZOrders.LANELET_POLY,
                                           facecolor=facecolor, edgecolor='none', antialiased=antialiased))
        if incoming_vertices_fill:
            self.static_collections.append(
                    collections.PolyCollection(incoming_vertices_fill, transOffset=self.ax.transData,
                                               facecolor=incoming_lanelets_color, edgecolor='none',
                                               zorder=ZOrders.INCOMING_POLY, antialiased=antialiased))
        if crossing_vertices_fill:
            self.static_collections.append(
                    collections.PolyCollection(crossing_vertices_fill, transOffset=self.ax.transData,
                                               facecolor=crossings_color, edgecolor='none',
                                               zorder=ZOrders.CROSSING_POLY, antialiased=antialiased))

        # draw_border_vertices
        if draw_border_vertices:
            # left vertices
            self.static_collections.append(
                    collections.EllipseCollection(np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                                                  np.ones([coordinates_left_border_vertices.shape[0], 1]) * 1.5,
                                                  np.zeros([coordinates_left_border_vertices.shape[0], 1]),
                                                  offsets=coordinates_left_border_vertices, color=left_bound_color,
                                                  transOffset=self.ax.transData, zorder=ZOrders.LEFT_BOUND + 0.1, ))

            # right_vertices
            self.static_collections.append(
                    collections.EllipseCollection(np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                                                  np.ones([coordinates_right_border_vertices.shape[0], 1]) * 1.5,
                                                  np.zeros([coordinates_right_border_vertices.shape[0], 1]),
                                                  offsets=coordinates_right_border_vertices, color=right_bound_color,
                                                  transOffset=self.ax.transData, zorder=ZOrders.LEFT_BOUND + 0.1, ))

        if draw_traffic_signs:
            # draw actual traffic sign
            for sign in traffic_signs:
                sign.draw(self, draw_params, call_stack)

        if draw_traffic_lights:
            # draw actual traffic light
            for light in traffic_lights:
                light.draw(self, draw_params, call_stack)

    def draw_planning_problem_set(self, obj: PlanningProblemSet, draw_params: Union[ParamServer, dict, None],
                                  call_stack: Tuple[str, ...]) -> None:
        """
        Draws all or selected planning problems from the planning problem set. Planning problems can be selected by
        providing IDs in`drawing_params[planning_problem_set][draw_ids]`

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = tuple(list(call_stack) + ['planning_problem_set'])
        draw_ids = draw_params.by_callstack(call_stack, 'draw_ids')

        for pp_id, problem in obj.planning_problem_dict.items():
            if draw_ids == 'all' or pp_id in draw_ids:
                self.draw_planning_problem(problem, draw_params, call_stack)

    def draw_planning_problem(self, obj: PlanningProblem, draw_params: Union[ParamServer, dict, None],
                              call_stack: Tuple[str, ...]) -> None:
        """
        Draw initial state and goal region of the planning problem

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = call_stack + ('planning_problem',)
        self.draw_initital_state(obj.initial_state, draw_params, call_stack)
        self.draw_goal_region(obj.goal, draw_params, call_stack)

    def draw_initital_state(self, obj: State, draw_params: Union[ParamServer, dict, None],
                            call_stack: Tuple[str, ...]) -> None:
        """
        Draw initial state with label

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = call_stack + ('initial_state',)
        zorder = draw_params.by_callstack(call_stack, 'label_zorder')
        label = draw_params.by_callstack(call_stack, 'label')

        obj.draw(self, draw_params, call_stack)
        self.static_artists.append(
                text.Annotation(label, xy=(obj.position[0] + 1, obj.position[1]), textcoords='data', zorder=zorder))

    def draw_goal_region(self, obj: GoalRegion, draw_params: Union[ParamServer, dict, None],
                         call_stack: Tuple[str, ...]) -> None:
        """
        Draw goal states from goal region

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        if call_stack == ():
            call_stack = tuple(['planning_problem_set'])
        call_stack = tuple(list(call_stack) + ['goal_region'])
        for goal_state in obj.state_list:
            self.draw_goal_state(goal_state, draw_params, call_stack)

    def draw_goal_state(self, obj: State, draw_params: Union[ParamServer, dict, None],
                        call_stack: Tuple[str, ...]) -> None:
        """
        Draw goal states

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        if hasattr(obj, 'position'):
            if type(obj.position) == list:
                for pos in obj.position:
                    pos.draw(self, draw_params, call_stack)
            else:
                obj.position.draw(self, draw_params, call_stack)

    def draw_traffic_light_sign(self, obj: Union[TrafficLight, TrafficSign],
                                draw_params: Union[ParamServer, dict, None], call_stack: Tuple[str, ...]):
        """
        Draw traffic sings and lights

        :param obj: object to be plotted
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack,
            which allows for differentiation of plotting styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        # traffic signs and lights have to be collected and drawn only right
        # before rendering to allow correct grouping
        self.traffic_sign_call_stack = call_stack
        self.traffic_sign_draw_params = draw_params
        self.traffic_signs.append(obj)

    def _draw_signal_state(self, sig: SignalState, occ: Occupancy, draw_params: Union[ParamServer, dict, None],
                           call_stack: Tuple[str, ...]):
        """
        Draw the vehicle signals

        :param sig: signal state to be drawn
        :param occ: Occupancy at current time step
        :param draw_params: parameters for plotting given by a nested dict
            that recreates the structure of an object or a ParamServer object
        :param call_stack: tuple of string containing the call stack, which allows for differentiation of plotting
            styles depending on the call stack
        :return: None
        """
        draw_params = self._get_draw_params(draw_params)
        call_stack = call_stack + ('signals',)
        signal_radius = draw_params.by_callstack(call_stack, 'signal_radius')

        indicators = []
        braking = []

        # indicators
        if isinstance(occ.shape, Rectangle):
            if hasattr(sig, 'hazard_warning_lights') and sig.hazard_warning_lights is True:
                indicators.extend(
                        [occ.shape.vertices[0], occ.shape.vertices[1], occ.shape.vertices[2], occ.shape.vertices[3]])
            else:
                if hasattr(sig, 'indicator_left') and sig.indicator_left is True:
                    indicators.extend([occ.shape.vertices[1], occ.shape.vertices[2]])
                if hasattr(sig, 'indicator_right') and sig.indicator_right is True:
                    indicators.extend([occ.shape.vertices[0], occ.shape.vertices[3]])

            for e in indicators:
                self.draw_ellipse(e, signal_radius, signal_radius, draw_params, call_stack + ('indicator',))

            # braking lights
            if hasattr(sig, 'braking_lights') and sig.braking_lights is True:
                braking.extend([occ.shape.vertices[0], occ.shape.vertices[1]])

            for e in braking:
                self.draw_ellipse(e, signal_radius * 1.5, signal_radius * 1.5, draw_params, call_stack + ('braking',))

            # blue lights
            if hasattr(sig, 'flashing_blue_lights') and sig.flashing_blue_lights is True:
                pos = occ.shape.center
                self.draw_ellipse(pos, signal_radius, signal_radius, draw_params, call_stack + ('bluelight',))

            # horn
            if hasattr(sig, 'horn') and sig.horn is True:
                pos = occ.shape.center
                self.draw_ellipse(pos, signal_radius * 1.5, signal_radius * 1.5, draw_params, call_stack + ('horn',))

        else:
            warnings.warn('Plotting signal states only implemented for '
                          'obstacle_shapes Rectangle.')
