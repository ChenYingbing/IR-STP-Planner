from enum import IntEnum
from typing import List, Tuple, Dict, Union
import heapq
import copy

class AstarNodeState(IntEnum):
    UNVISITED = -1
    OPEN_LIST = 0
    CLOSE_LIST =1

class AstarNode:
  def __init__(self, parent_index: Tuple, index: Tuple, g: float=0.0, h: float=0.0) -> None:
    '''
    Init astar node with g, h values
    '''
    self.parent_index: Union[int, Tuple] = parent_index
    self.index: Union[int, Tuple] = index
    self.state: AstarNodeState =AstarNodeState.UNVISITED

    self.g: float = g
    self.h: float = h

  def f(self):
    '''
    Return f value of this node
    '''
    return (self.g + self.h)

class AStarSearch:
  def __init__(self) -> None:
    self.node_set: Dict[Union[int, Tuple], AstarNode] = {}
    self.open_list: List = []

    self.ReInit()

  def ReInit(self) -> None:
    '''
    Reinit the class for next operation
    '''
    self.node_set.clear()
    self.open_list.clear()

  def node_set_size(self) -> int:
    '''
    Return the amount of atar node
    '''
    return len(self.node_set)

  def get_node(self, index: Union[int, Tuple]) -> AstarNode:
    '''
    Return the astar node given index
    '''
    return self.node_set[index]

  def is_at_closed_list(self, node_index: Union[int, Tuple]) -> bool:
    '''
    Return true if the given node (with node index = node_index) is at closed list
    '''
    if node_index in self.node_set.keys():
      return (self.node_set[node_index].state == AstarNodeState.CLOSE_LIST)
    return False

  def Add2Openlist(self, parent_index: Union[int, Tuple], 
                         index: Union[int, Tuple],
                         g: float, h: float) -> Tuple[bool, Tuple[int, int]]:
    '''
    Add node to open list
    :param parent_index: index of the parent node
    :param index: index of the child node, which will be added to the openlist
    :param g: g cost of the child node
    :param h: h value of the child node
    :return: [flag, abandoned edge], where flag == true when 
      the node is successfully added, and the abandoned edge represents
      the edge being abandoned in a-star searching
    '''
    if not index in self.node_set.keys():
      # AstarNodeState::UNVISITED
      self.node_set[index] = AstarNode(
        parent_index=parent_index, index=index, g=g, h=h)
      self.node_set[index].state = AstarNodeState.OPEN_LIST

      f_value = self.node_set[index].f()
      heapq.heappush(self.open_list, (f_value, index))

      return True, None
    else:
      get_node = self.node_set[index]
      if (get_node.state == AstarNodeState.OPEN_LIST) and (g < get_node.g):
        # AstarNodeState.OPEN_LIST
        old_f_value = get_node.f()

        old_edge = (get_node.parent_index, get_node.index)

        # print("new old parent", [get_node.parent_index, parent_index])
        self.node_set[index].parent_index = parent_index
        self.node_set[index].g = g

        # update node with new values
        if (old_f_value, index) in self.open_list:
          self.open_list.remove((old_f_value, index))
        else:
          raise ValueError("This should not occurs as get_node is at open_list")

        f_value = self.node_set[index].f()
        heapq.heappush(self.open_list, (f_value, index))

        return True, old_edge
      # else:
      #   AstarNodeState.CLOSE_LIST
      # >> skip when set_node.state == AstarNodeState::CLOSE_LIST
    
    return False, None

  def PopOpenList(self) -> Tuple[bool, Union[int, Tuple]]:
    '''
    Pop node from the open list
    :return: flag, pop_node_index, when flag==true, the open_list is empty
    '''
    if len(self.open_list) == 0:
      return True, None

    _f_value, index = heapq.heappop(self.open_list)

    get_node = self.node_set[index]
    if (get_node.state == AstarNodeState.OPEN_LIST):
      get_node.state = AstarNodeState.CLOSE_LIST
      return False, index
    else:
      raise ValueError("Should not occurs")

  def ExtractPath(self, maximum_iterations: int, 
                  start_index: Union[int, Tuple], goal_index: Union[int, Tuple]) -> List:
    '''
    Found and return a sequence of node index
    :param maximum_iterations: maximum iteration amount 
    :return: list of nodes when a path to goal from start is found, else empty.
    '''
    found_goal: bool = False
    this_index: int = goal_index
    indexs_path: List = []

    for i in range(0, maximum_iterations):
      get_node = self.node_set[this_index]
      indexs_path.append(get_node.index)

      if (this_index == start_index):
        found_goal = True
        break

      if get_node.parent_index != None:
        this_index = get_node.parent_index
      else:
        break

    if found_goal == True:
      indexs_path.reverse()
    else:
      indexs_path.clear()

    return indexs_path
