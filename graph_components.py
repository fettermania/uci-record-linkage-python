import pandas as pd
import numpy as np

# graph_components.py

# Basic idea: component ids are the same space as node ids,
# and are the same id as the lowest node_id in that component.
class GraphComponents(object):

  def __init__(self, df_ids):
    self.edges = {}
    self.nodes = []

    self.node_to_component_map = {} # int -> int
    self.component_to_ids_map = {} # int -> set

    df_ids_array = df_ids.values
    # Fettermania; Could possibly remove this, doing it inline
    for edge in df_ids_array:
      # print("ADDING NODE %d" % edge[0])
      self.add_node(edge[0])
      # print("ADDING NODE %d" % edge[1])
      self.add_node(edge[1])

    # print("ID -> COMPONENT")
    # print(self.node_to_component_map);

    # print("COMPONENT -> IDS")
    # print(self.component_to_ids_map);

    for edge in df_ids_array:
      self.add_edge(edge[0], edge[1])

  def get_components(self):
    return self.component_to_ids_map

  def get_nodes_in_component(self, component):
    return self.component_to_ids_map(component)

  def get_component_for_node(self, node_id):
    return self.node_to_component_map[node]

  def add_node(self, node_id):
    if node_id not in self.node_to_component_map:
      self._add_node_to_component(node_id, node_id)
      self.nodes.append(node_id)

  def add_edge(self, node_id_1, node_id_2):
    if node_id_1 in self.edges:
      self.edges[node_id_1].add(node_id_2)
    else:
      self.edges[node_id_1] = set([node_id_2])


    if node_id_2 in self.edges:
      self.edges[node_id_2].add(node_id_1)
    else:
      self.edges[node_id_2] = set([node_id_1])
    

    # print("ADDING EDGE %d -- %d" % (node_id_1, node_id_2))
    component_1 = self.node_to_component_map[node_id_1]
    component_2 = self.node_to_component_map[node_id_2]
    min_component = min(component_1, component_2)

    # print (" .... MIN COMPONENT IS %d" % min_component)

    if component_1 != min_component:

      self._add_node_to_component(node_id_1, min_component)
      self._remap_component_to_new_id(component_1, min_component)

    if component_2 != min_component:
      self._add_node_to_component(node_id_2, min_component)
      self._remap_component_to_new_id(component_2, min_component)

  # Invariant: Node_id is in exactly one component
  def _add_node_to_component(self, node_id, component):
    # remove node_id from set in component -> {node_ids_set}
    if node_id in self.node_to_component_map:
      old_component = self.node_to_component_map[node_id]
      if old_component != component:
        self._helper_remove_id_from_component(node_id, old_component)

    # (re)assign node_id -> component
    self.node_to_component_map[node_id] = component
    
    # add node_id to component
    # add: if component's node_id set exists, add.  Else create singleton set.
    if component in self.component_to_ids_map:
      self.component_to_ids_map[component].add(node_id)
    else:
      self.component_to_ids_map[component] = set([node_id])
  
  # Helper: Only called by _add_node_to_component
  # Fettermania: Basically "remove node from graph"
  def _helper_remove_id_from_component(self, node_id, component_id):
    if component_id in self.component_to_ids_map:
      self.component_to_ids_map[component_id].remove(node_id)
      if len(self.component_to_ids_map[component_id]) == 0:
        # print("Remove: COMPONENT %d is " % component_id)
        # print(self.component_to_ids_map[component_id])
        # print("Remove: DESTROYING COMPONENT %d" % component_id)
        del self.component_to_ids_map[component_id]

    if node_id in self.node_to_component_map:
      del self.node_to_component_map[node_id]

  def _remap_component_to_new_id(self, component_id, new_component_id):
    # print ("REMAPPING COMPONENT %d to %d" % (component_id, new_component_id))
    if component_id in self.component_to_ids_map: # Fettermania TODO gross
      node_ids = self.component_to_ids_map[component_id].copy()

      for node_id in node_ids:
        self._add_node_to_component(node_id, new_component_id)

def get_test_data_frame():
  # s1 = pd.Series([1,1,2,3,5])
  # s2 = pd.Series([2,3,4,6,5])
  s1 = pd.Series( [np.random.random_integers(1,30) for i in range(10)])
  s2 = pd.Series( [np.random.random_integers(1,30) for i in range(10)])
  df = pd.DataFrame({'id_1': s1, 'id_2': s2})
  return df

