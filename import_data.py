#import_data.py

import pandas as pd
import os
import re
import numpy as np


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


def dataframe_clean_columns_in_place(df):
  for col_name in df.columns:
    if (df[col_name].dtype == np.dtype('O')):
      null_col_name = col_name + "_is_null"
      df[null_col_name] = df[col_name].apply(lambda x : 1 if x == "?" else 0)
      df[col_name] = df[col_name].apply(lambda x: 0.0 if x == "?" else x)
      df[col_name] = df[col_name].astype(float)
  target_column = target_column_from_df(df)
  df[target_column] = df[target_column].astype(int)

