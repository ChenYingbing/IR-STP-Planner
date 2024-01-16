import copy
from typing import Dict, Tuple

from IPython import display
from commonroad.common.solution import PlanningProblemSolution, Solution, CommonRoadSolutionWriter, VehicleType, \
    VehicleModel, CostFunction
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from sumocr.interface.ego_vehicle import EgoVehicle

def agent_limited_length() -> float:
    return 6.5

def agent_limited_width() -> float:
    return 1.8

def limit_agent_length_width(length: float, width: float) -> Tuple:
    '''limit length/width of truck-like vehicles to improve simulation performance
       Because some agents shape are too large to avoid from the route of the AV.
    '''
    l = min(length, 6.5)
    w = min(width, 1.8)
    return l, w

def save_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, ego_vehicles: Dict[int, EgoVehicle],
                  vehicle_type: VehicleType,
                  vehicle_model: VehicleModel,
                  cost_function: CostFunction,
                  output_path: str = './',
                  filename: str = None, 
                  overwrite: bool = False):
    """Saves the given trajectory as a solution to the planning problem"""

    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in planning_problem_set.planning_problem_dict
        state_initial = copy.deepcopy(planning_problem_set.planning_problem_dict[pp_id].initial_state)
        set_attributes_state_initial = set(state_initial.attributes)
        list_states_trajectory_full = [state_initial]

        # set missing attributes to correctly construct solution file
        for state in ego_vehicle.driven_trajectory.trajectory.state_list:
            set_attributes_state = set(state.attributes)

            set_attributes_in_state_extra = set_attributes_state.difference(set_attributes_state_initial)
            if set_attributes_in_state_extra:
                for attribute in set_attributes_in_state_extra:
                    setattr(state_initial, attribute, 0)

            set_attributes_in_state_initial_extra = set_attributes_state_initial.difference(set_attributes_state)
            if set_attributes_in_state_initial_extra:
                for attribute in set_attributes_in_state_initial_extra:
                    setattr(state, attribute, 0)

            list_states_trajectory_full.append(state)

        trajectory_full = Trajectory(initial_time_step=0, state_list=list_states_trajectory_full)
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=vehicle_type,
                                           vehicle_model=vehicle_model,
                                           cost_function=cost_function,
                                           trajectory=trajectory_full))

    solution = Solution(scenario.scenario_id, pps)

    # write solution
    csw = CommonRoadSolutionWriter(solution)
    csw.write_to_file(
        output_path=output_path, filename=filename, overwrite=overwrite)
    print("Trajectory saved to solution file.")
