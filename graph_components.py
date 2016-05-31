# graph_components.py
class GraphComponents(object):

  def __init__(self, df_ids):
    self.id_to_component_map = {} # int -> int
    self.component_to_ids_map = {} # int -> set
    self.next_component_id = 0;
    df_ids_array = df_ids.values
    
    # Fettermania; Could possibly remove this, doing it inline
    for ids in df_ids_array:
      self._add_id_to_component(ids[0], ids[0])
      self._add_id_to_component(ids[1], ids[1])

    for ids in df_ids_array:
      self._add_edge(ids[0], ids[1])

    # Fettermania TODO: possibly could walk backwards instead of doing ALL
    for ids in df_ids_array:
      self._resolve_component_for_id(ids[0])
      self._resolve_component_for_id(ids[1])

  def get_ids_in_component(component):
    return self.component_to_ids_map(component)

  def get_component_for_id(id):
    return self.id_to_component_map[id]

  def _add_id_to_component(id, component):
    # remove id from set in component -> {ids_set}
    if id in self.id_to_component_map:
      old_component = self.id_to_component[id]
      self._remove_id_from_component(old_component)

    # (re)assign id -> component
    id_to_component_map[id] = component
    
    # add: if component exists, add.  Else create singleton set.
    if component in component_to_ids_map:
      component_to_ids[component].add(id)
    else:
      component_to_ids[component] = set([id])
  
  def _remove_id_from_component(id, component):
    if component in component_to_ids_map:
      self.component_to_ids_map[component].remove(id)

  def _add_edge(id1, id2):
    component_1 = self.id_to_component_map[id1]
    component_2 = self.id_to_component_map[id2]
    min_component = min(component_1, component_2)

    if component_1 != min_component:
      self._remove_id_from_component(id_1, component_1)

    if component_2 != min_component:
      self._remove_id_from_component(id_2, component_2)

    self._add_id_to_component(id1, min_component)
    self._add_id_to_component(id2, min_component)

  def _resolve_component_for_id(id):
    to_resolve = []
    last_component_id = id
    next_component_id = self.id_to_component_map[last_component_id]

    while (next_component_id != last_component_id):
      to_resolve.append(last_component_id)
      last_component_id = next_component_id
      next_component_id = self.id_to_component_map[next_component_id]

    for id_to_resolve in to_resolve:
      self._add_id_to_component(id_to_resolve, last_component_id)



