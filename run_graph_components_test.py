import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import graph_components


# Import data

import import_data
print('Importing data from block_*.csv...')
df = import_data.df_from_known_csvs()
import_data.dataframe_clean_columns_in_place(df)

true_df = df[df['is_match'] == 1]
# X_true = true_df[import_data.feature_columns_from_df(true_df)]
# y_true = true_df[import_data.target_column_from_df(true_df)]

false_df = df[df['is_match'] == 0]
# X_false = true_df[import_data.feature_columns_from_df(false_df)]
# y_false = true_df[import_data.target_column_from_df(false_df)]



def component_histogram(gc):
  component_map = gc.get_components()
  component_lengths = list(map(lambda s: len(component_map[s]), component_map))
  # Fettermania: Note that UCI data set is all non-loop edges, so no singleton nodes
  return np.histogram(component_lengths, bins=np.arange(2, max(component_lengths) + 1))

def graph_stats(gc):
  edge_count = sum(map(lambda x: len(gc.edges[x]), gc.edges)) / 2
  node_count = len(gc.nodes)
  component_count = len(gc.get_components())
  return (edge_count, node_count, component_count)

# Graph of true links
true_gc = graph_components.GraphComponents(true_df[['id_1', 'id_2']])
print ("TRUE GRAPH")
print ("E: %d, N: %d, C: %d" % graph_stats(true_gc))
print ("HISTOGRAM:")
print(component_histogram(true_gc))

false_gc = graph_components.GraphComponents(false_df[['id_1', 'id_2']])
print ("FALSE GRAPH")
print ("E: %d, N: %d, C: %d" % graph_stats(false_gc))
print ("HISTOGRAM:")
print(component_histogram(false_gc))

print("CHECK: Shared edges?  Should be empty: %r" % graph_components.check_shared_edges(true_gc, false_gc))




