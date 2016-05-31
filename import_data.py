#import_data.py

import pandas as pd
import os
import re
import numpy as np

# Fettermania libraries

def df_from_known_csvs():
  dfs = []
  for i in range(1, 11):
    dfs.append(pd.read_csv("./data/block_{}.csv".format(i)))
  return pd.concat(dfs)

def feature_columns_from_df(df):
  columns = list(df.columns)
  columns.remove('id_1')
  columns.remove('id_2')
  columns.remove('is_match')
  return columns

def target_column_from_df(df):
  return 'is_match'


# TODO: Fettermania: Drop other data column
def dataframe_clean_columns_in_place(df):
  for col_name in df.columns:
#    print ("CHECKING COLUMN " + col_name)
    if (df[col_name].dtype == np.dtype('O')):
      null_col_name = col_name + "_is_null"
      # Fettermania: Trying to "add feature" to missing column???
      df[null_col_name] = df[col_name].apply(lambda x : 1 if x == "?" else 0)
      #np.zeros(df[col_name].shape[0])
      #df[null_col_name][df[col_name] == "?"] = 1.0
      df[col_name] = df[col_name].apply(lambda x: 0.0 if x == "?" else x)
      df[col_name] = df[col_name].astype(float)
  # TODO: Call this "target column" in config or something
  df['is_match'] = df['is_match'].astype(int)

